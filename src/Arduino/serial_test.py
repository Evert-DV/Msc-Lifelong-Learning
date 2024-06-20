import os

os.environ["KERAS_BACKEND"] = "torch"
import threading
import serial
# import keyboard
import time
from torch.utils.data import DataLoader
from src.toy_problem.toy_tools import *

target = -5
true_target = -5
arduino = None
lock = None
new_state = None
new_omega = None
freq = 1 / 50
recorded_data = []
buffer = []


def save_recorded_data(name):
    print(f"\nSaving recorded_data")
    np.save(f"./Dynamics data/75-75 perforation/{name}.npy", recorded_data)


def send_value():
    global freq
    global target
    while True:
        arduino.write(f'{target}\n'.encode())
        time.sleep(freq)


def listen_echo():
    global freq
    global recorded_data
    global buffer
    global new_state
    global new_omega

    time.sleep(.5)

    while True:
        if arduino.in_waiting > 0:
            # Read all data in the recorded_data
            data = arduino.readlines(arduino.in_waiting)

            # Split the data into lines and take the last line
            if not data:
                continue

            latest_value = data[-1].decode('utf-8').strip().split()
            # latest value: old beta, old omega, control action, new beta, new omega
            old_state = float(latest_value[0])
            old_omega = float(latest_value[1])
            control_action = float(latest_value[2])
            new_state = float(latest_value[-2])
            new_omega = float(latest_value[-1])

            if time.time() % (1 / 4) < 0.03:
                print(
                    f"\rState: {old_state:.1f}, {old_omega:.1f}\tControl action: {control_action:.0f}"
                    f"\tNew state: {new_state}, {new_omega:.1f}\tTarget: {target:.1f}", end="")
            recorded_data.append([old_state, old_omega, control_action, new_state, new_omega, target])
            with lock:
                buffer.append([old_state, old_omega, control_action, new_state, new_omega, target, true_target])
        # if keyboard.is_pressed('s'):  # If 's' is pressed, save the recorded_data
        #     save_recorded_data("user_save")

        time.sleep(freq)


def change_value():
    global target
    global true_target
    global new_state
    global new_omega
    global freq
    global recorded_data
    global buffer

    # set epoch
    start_time = time.time()
    target_t = start_time
    train_t = start_time

    count = 0
    new_state = 0
    new_omega = 0
    true_target = -5
    prediction_window = 3

    adapter = keras.models.load_model('./Models/adapter_5.keras')

    print(f"\n{10 * '='} TRUE TARGET: {true_target} {10 * '='}")
    while True:
        adapter_input = ops.array([new_state, new_omega, true_target, 0.])[None]
        delta_target = adapter.predict(adapter_input, verbose=0)[0][0]
        target = delta_target

        if time.time() - target_t > 7.5:
            true_target = np.random.randint(-20, -5)
            print(f"\n{10 * '='} TRUE TARGET: {true_target} {10 * '='}")
            # true_target = -6 if true_target == -17 else -17  # oscillate
            # target = true_target
            target_t = time.time()

        if (time.time() - start_time) % 300 < 0.025 and (time.time() - start_time) > 5:
            save_recorded_data(f"b_auto_save_{count}")
            # recorded_data = []
            count += 1

        if (time.time() - train_t) > 15:
            print(f"\n{10 * '='} Updating model {10 * '='}")
            adapter.optimizer.lr = 1.e-4

            with lock:
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
            adapter.fit(train_dataloader,
                        epochs=100,
                        callbacks=callbacks,
                        validation_data=val_dataloader,
                        verbose=0,
                        )
            buffer = []
            train_t = time.time()

        time.sleep(freq)


def main():
    global lock
    global arduino
    arduino = serial.Serial('COM5', 115200)

    lock = threading.Lock()

    # Create threads for sending and receiving data
    send_thread = threading.Thread(target=send_value, daemon=True)
    listen_thread = threading.Thread(target=listen_echo, daemon=True)
    target_thread = threading.Thread(target=change_value, daemon=True)

    # Start threads
    send_thread.start()
    listen_thread.start()
    target_thread.start()

    # Ensure the main thread waits for the completion of other threads
    send_thread.join()
    listen_thread.join()
    target_thread.join()


if __name__ == "__main__":
    print("Using backend " + keras.backend.backend())
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    if torch.cuda.is_available():
        print("Using CUDA")
        with torch.cuda.device(0):
            main()
    else:
        main()
