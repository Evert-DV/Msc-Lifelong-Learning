import os

os.environ["KERAS_BACKEND"] = "torch"
import threading
import serial
import keyboard
import time
from torch.utils.data import DataLoader
from src.toy_problem.toy_tools import *

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
adapter = keras.models.load_model(f'./Models/adapter_{prediction_window}.keras')
adapter.load_weights(f'./Models/deployed_adapter_{prediction_window}.weights.h5')


def save_recorded_data(name):
    print(f"\n{32 * '='}\n===   Saving recorded data   ===\n{32 * '='}")
    np.save(f"./Dynamics data/75-75 perforation/{name}.npy", recorded_data)


def user_save():
    save_recorded_data("user_save")
    with update_lock:
        adapter.save_weights(f'./Models/deployed_adapter_{prediction_window}.weights.h5')
    time.sleep(.5)



def send_value():
    while True:
        arduino.write(f'{target}\n'.encode())
        time.sleep(freq)


def listen_echo():
    global recorded_data
    global buffer
    global new_state
    global new_omega

    time.sleep(1)
    print_time = time.time()

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
            new_state = float(latest_value[-3])
            new_omega = float(latest_value[-2])
            read_target = float(latest_value[-1])

            if time.time() - print_time > 0.5:
                print(
                    f"\rState: {old_state:.1f}, {old_omega:.1f}\tControl action: {control_action:.0f}"
                    f"\tNew state: {new_state}, {new_omega:.1f}\tTarget: {read_target:.1f}", end="")
                print_time = time.time()

            recorded_data.append([old_state, old_omega, control_action, new_state, new_omega, read_target, true_target])

            with buffer_lock:
                buffer.append([old_state, old_omega, control_action, new_state, new_omega, read_target, true_target])
                if len(buffer) > 3000:  # about a minute of data
                    buffer.pop(0)

        # time.sleep(freq/4)


def change_value():
    global target
    global true_target
    global recorded_data

    # set epoch
    start_time = time.time()
    target_t = start_time
    save_t = start_time

    count = 0
    n = 0

    print(f"\n{10 * '='} TRUE TARGET: {true_target} {10 * '='}")
    while True:
        t0 = time.time()

        adapter_input = ops.array([new_state, new_omega, true_target, 0.])[None]
        delta_target = adapter.predict(adapter_input, verbose=0)[0][0]
        target = delta_target

        if time.time() - target_t > 5:
            true_target = np.random.randint(-23, -2)
            # if n % 2 == 0:
            #     true_target = np.random.randint(-23, -15)
            # else:
            #     true_target = np.random.randint(-10, -2)
            # n += 1

            print(f"\n{10 * '='} TRUE TARGET: {true_target} {10 * '='}")
            # true_target = -6 if true_target == -17 else -17  # oscillate
            # target = true_target
            target_t = time.time()

        if time.time() - save_t > 300:
            save_recorded_data(f"auto_save_{count}")
            with update_lock:
                adapter.save_weights(f'./Models/deployed_adapter_{prediction_window}.weights.h5')
            # recorded_data = []
            count += 1
            save_t = time.time()

        dt = time.time() - t0
        time.sleep(max(0, freq - dt))


def update_adapter():
    global adapter
    global buffer

    # set epoch
    start_time = time.time()
    train_t = start_time

    temp_adapter = TargetAdapter(state_size=2)

    while True:
        if time.time() - train_t > 15:
            print(f"\n{26 * '='}\n===   Updating model   ===\n{26 * '='}")

            temp_adapter.set_weights(adapter.get_weights())
            temp_adapter.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                                 loss=keras.losses.MeanSquaredError())

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


def main():
    global buffer_lock
    global update_lock
    global arduino
    arduino = serial.Serial('COM5', 115200)

    keyboard.add_hotkey('ctrl+s', user_save)

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
