import os

os.environ["KERAS_BACKEND"] = "torch"
import threading
import serial
import keyboard
import time
import pickle
import keras
import torch
import numpy as np
from src.toy_problem.kb_tools import VariationalAutoEncoder, MultivariateNormal, sample, search_kb, js_divergence
from src.toy_problem.toy_tools import TargetAdapter, RMSERegularizer, EpochLogger, prep_data
from keras import ops, optimizers, losses
from torch.utils.data import TensorDataset, DataLoader, random_split

running = True

arduino = None
buffer_lock = None
update_lock = None

target = -5
true_target = -5
new_state = 0
new_omega = 0

freq = 1 / 50

recorded_data = []
buffer = []

# Load vanilla model
prediction_window = 10
adapter = keras.models.load_model(f'./Models/adapter_allround.keras')

# Load deployed model weights
adapter.load_weights(f'./Models/deployed_adapter_{prediction_window}.weights.h5')

# Load VAE model
autoencoder = VariationalAutoEncoder(5, 2)  # keras.models.load_model("./Models/vae_skip_a.keras")
autoencoder.load_weights("./Models/vae_skip_s.weights.h5")


def save_recorded_data(name):
    print(f"\n{32 * '='}\n===   Saving recorded data   ===\n{32 * '='}")
    np.save(f"./Dynamics data/75-75 perforation/{name}.npy", recorded_data)


def user_save():
    save_recorded_data("user_save")
    with update_lock:
        adapter.save_weights(f'./Models/deployed_adapter_{prediction_window}.weights.h5')
    time.sleep(.5)


def send_value():
    while running:
        arduino.write(f'{target}\n'.encode())
        time.sleep(freq)
    print("\nClosing `send_value` thread")


def listen_echo():
    global recorded_data
    global buffer
    global new_state
    global new_omega

    time.sleep(1)
    print_time = time.time()

    while running:
        t0 = time.time()
        if arduino.in_waiting > 0:
            # Read all data in the recorded_data
            data = arduino.readlines(arduino.in_waiting)
            # Split the data into lines and take the last line
            if not data:
                continue

            latest_value = data[-1].decode('utf-8').strip().split()
            # latest value: old beta, old omega, control action, new beta, new omega
            counter = int(latest_value[0])
            old_state = float(latest_value[1])
            old_omega = float(latest_value[2])
            control_action = float(latest_value[3])
            new_state = float(latest_value[-3])
            new_omega = float(latest_value[-2])
            read_target = float(latest_value[-1])

            if time.time() - print_time > 0.5:
                print(
                    f"\rState: {old_state:.1f}, {old_omega:.1f}\tControl action: {control_action:.0f}"
                    f"\tNew state: {new_state}, {new_omega:.1f}\tTarget: {read_target:.1f}", end="")
                print_time = time.time()

            recorded_data.append(
                [counter, old_state, old_omega, control_action, new_state, new_omega, read_target, true_target])

            with buffer_lock:
                buffer.append([old_state, old_omega, control_action, new_state, new_omega, read_target, true_target])
                if len(buffer) > 3000:  # about a minute of data
                    buffer.pop(0)
        dt = time.time() - t0
        time.sleep(max(0, freq - dt))

    print("\nClosing `listen_echo` thread")


def change_value():
    global target
    global true_target
    global recorded_data

    # kb setup
    use_kb = False

    if use_kb:
        try:
            with open('kb.pkl', 'rb') as f:
                kb = pickle.load(f)
                reference = None
                print(f"\n{len(kb[0])} entries loaded from the KB")
        except FileNotFoundError:
            reference_data = np.load(f"./Dynamics data/75-75 perforation/deployed_adapter_{prediction_window}.npy")
            x_reference = ops.array(reference_data)[..., [1, 2, 3, 4, 5]].reshape(-1, 5)
            z_mean, z_log_var = autoencoder.dynamics(x_reference)
            cov = sample(z_mean, z_log_var)[1]
            reference = MultivariateNormal(z_mean, cov)
            kb = [[reference], [adapter.get_weights()]]

        running_distribution = None
        updated_reference = None
        backup_updated_reference = None
        current_kb_idx = None

        thres = np.log(2) / 2

        first_selection = True

    # set epoch
    start_time = time.time()
    target_t = start_time
    save_t = start_time
    kb_t = start_time
    update_t = start_time

    count = 0
    n = 0

    print(f"\n{10 * '='} TRUE TARGET: {true_target} {10 * '='}")
    while running:
        t0 = time.time()
        to_reach = max(true_target, new_state - 13) if true_target < new_state else min(true_target, new_state + 17)
        adapter_input = ops.array([new_state, new_omega, to_reach, 0.])[None]
        delta_target = adapter.predict(adapter_input, verbose=0)[0][0]
        target = delta_target

        if time.time() - target_t > 5:
            true_target = np.random.randint(-23, -2)
            # true_target = -20 if n % 2 == 0 else -5  # oscillate
            # n += 1
            # target = true_target
            print(f"\n{10 * '='} TRUE TARGET: {true_target} {10 * '='}")
            target_t = time.time()

        if time.time() - save_t > 300:
            save_recorded_data(f"auto_save_{count}")
            with update_lock:
                adapter.save_weights(f'./Models/deployed_adapter_{prediction_window}.weights.h5')
            count += 1
            save_t = time.time()

        # KB step
        if not use_kb:
            dt = time.time() - t0
            time.sleep(max(0, freq - dt))
            continue

        if time.time() - kb_t > 5:
            kb_t = time.time()

            # Get embeddings
            with buffer_lock:
                z_mean, z_log_var = autoencoder.dynamics(ops.array(buffer)[..., [0, 1, 2, 3, 4]])
            embeddings, cov = sample(z_mean, z_log_var, samples_per_centroid=1)

            # Pre-60 seconds
            if time.time() - start_time < 60:
                running_distribution = MultivariateNormal(z_mean, cov)
                current_kb_idx, js_divs = search_kb(running_distribution, kb[0])
                continue

            if first_selection:
                # Select KB entry
                reference = kb[0][current_kb_idx]
                updated_reference = reference.copy()
                backup_updated_reference = reference.copy()
                with update_lock:
                    adapter.set_weights(kb[1][current_kb_idx])
                print(f"\nSelected KB entry {current_kb_idx} as reference")
                first_selection = False

            # Post-60 seconds
            # Update running distribution
            running_distribution.update(z_mean, cov, weight=5 / (60 - 5))

            # KL losses
            kl_updated_dist = js_divergence(updated_reference, running_distribution)
            kl_reference_updated = js_divergence(reference, updated_reference)
            # kl_loss.append([kl_updated_dist.item(), kl_reference_updated.item()])

            # Check for shift
            if kl_updated_dist > thres or kl_reference_updated > .75 * thres:
                # retain last or leave it as it was?
                # kb[0][kb_idx] = backup_updated_reference.copy()
                # kb[1][kb_idx] = adapter.get_weights()
                best_idx, js_divs = search_kb(running_distribution, kb[0])
                if best_idx != current_kb_idx and js_divs[best_idx] < thres:
                    print(f"\n{34 * '='}\n"
                          f"||        SHIFT DETECTED        ||\n"
                          f"||  Restore KB entry reference  ||\n"
                          f"||   Check KB for better match  ||\n"
                          f"|| Use KB entry {best_idx} as reference{(3 - len(str(best_idx))) * ' '}||\n"
                          f"{34 * '='}")
                    reference = kb[0][best_idx]
                    updated_reference = reference.copy()
                    backup_updated_reference = reference.copy()
                    current_kb_idx = best_idx
                    with update_lock:
                        adapter.set_weights(kb[1][best_idx])
                else:
                    print(f"\n{34 * '='}\n"
                          f"||        SHIFT DETECTED        ||\n"
                          f"||  Restore KB entry reference  ||\n"
                          f"||  Check KB for better match   ||\n"
                          f"||        No match found        ||\n"
                          f"||     Initiate new KB entry    ||\n"
                          f"{34 * '='}")
                    reference = running_distribution.copy()
                    updated_reference = reference.copy()
                    kb[0].append(reference)
                    with update_lock:
                        kb[1].append(adapter.get_weights())
                    current_kb_idx = len(kb[0]) - 1

        # KB update step
        if time.time() - update_t > 60 and not first_selection:
            torch.cuda.empty_cache()
            best_idx, js_divs = search_kb(running_distribution, kb[0])
            if best_idx != current_kb_idx and js_divs[best_idx] < thres:
                print(f"\n{34 * '='}\n"
                      f"||         UPDATE STEP          ||\n"
                      f"||  Check KB for better match   ||\n"
                      f"|| Use KB entry {best_idx} as reference ||\n"
                      f"{34 * '='}")
                kb[0][current_kb_idx] = backup_updated_reference  # restore the distribution
                with update_lock:
                    kb[1][current_kb_idx] = adapter.get_weights()  # update the weights
                reference = kb[0][best_idx]
                updated_reference = reference.copy()
                backup_updated_reference = reference.copy()
                current_kb_idx = best_idx
                with update_lock:
                    adapter.set_weights(kb[1][best_idx])
                continue

            # Update reference
            print(f"\n{34 * '='}\n"
                  f"||         UPDATE STEP          ||\n"
                  f"||  Check KB for better match   ||\n"
                  f"||        No match found        ||\n"
                  f"||      Updated reference       ||\n"
                  f"{34 * '='}")
            backup_updated_reference = updated_reference.copy()
            updated_reference.update(z_mean, cov, weight=0.1)

            update_t = time.time()

        dt = time.time() - t0
        time.sleep(max(0, freq - dt))

    # Save the last updated reference
    if use_kb:
        kb[0][current_kb_idx] = backup_updated_reference
        with update_lock:
            kb[1][current_kb_idx] = adapter.get_weights()

        with open('kb.pkl', 'wb') as f:
            pickle.dump(kb, f)

        print(f"\n{len(kb[0])} entries saved in the KB\nClosing `change_value` thread")
    with update_lock:
        adapter.save_weights(f'./Models/deployed_adapter_{prediction_window}.weights.h5')
    save_recorded_data(f"deployed_adapter_{prediction_window}")


def update_adapter():
    global adapter
    global buffer

    # set epoch
    start_time = time.time()
    train_t = start_time

    temp_adapter = TargetAdapter(state_size=2)
    # temp_adapter.regularizer.add(RMSERegularizer(weight=.1))
    temp_adapter.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5),
                         loss=keras.losses.MeanSquaredError())

    while running:
        if time.time() - train_t > 15:
            print(f"\n{26 * '='}\n===   Updating model   ===\n{26 * '='}")

            temp_adapter.set_weights(adapter.get_weights())

            with buffer_lock:
                update_buffer = ops.array(buffer)[..., :-1]
                true_target_list = ops.array(buffer)[..., -1:].tolist()

            features, labels = prep_data(update_buffer, prediction_window, state_size=2, target_size=1,
                                         true_target_list=true_target_list)
            train_dataset, val_dataset = random_split(TensorDataset(features, labels),
                                                      [int(0.8 * len(features)),
                                                       len(features) - int(0.8 * len(features))])
            train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)
            callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       mode='min',
                                                       min_delta=1e-4,
                                                       patience=5,
                                                       restore_best_weights=True,
                                                       verbose=1),
                         EpochLogger()
                         ]
            temp_adapter.fit(train_dataloader,
                             epochs=1000,
                             callbacks=callbacks,
                             validation_data=val_dataloader,
                             verbose=0,
                             )
            with update_lock:
                adapter.set_weights(temp_adapter.get_weights())

            train_t = time.time()

        time.sleep(freq)
    print("\nClosing `update_adapter` thread")


def main():
    global buffer_lock
    global update_lock
    global arduino
    arduino = serial.Serial('COM5', 115200)

    keyboard.add_hotkey('ctrl+s', user_save)
    keyboard.add_hotkey('ctrl+q', lambda: globals().update(running=False))

    buffer_lock = threading.Lock()
    update_lock = threading.Lock()

    # Create threads for sending and receiving data
    send_thread = threading.Thread(target=send_value, daemon=True)
    listen_thread = threading.Thread(target=listen_echo, daemon=True)
    target_thread = threading.Thread(target=change_value, daemon=True)
    update_thread = threading.Thread(target=update_adapter, daemon=True)

    # Start threads
    send_thread.start()
    target_thread.start()
    listen_thread.start()
    update_thread.start()

    # Ensure the main thread waits for the completion of other threads
    send_thread.join()
    target_thread.join()
    listen_thread.join()
    update_thread.join()


if __name__ == "__main__":
    print("Using backend " + keras.backend.backend())
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    if torch.cuda.is_available():
        print("Using CUDA")
        with torch.cuda.device(0):
            main()
    else:
        main()
