import threading
import serial
import keyboard
import time
import numpy as np
from torch import ops
from src.toy_problem.toy_tools import TargetAdapter

target = -5
arduino = None
new_state = None
freq = 1 / 50
buffer = []


def save_buffer(name):
    print("Saving buffer")
    np.save(f"./Dynamics data/75-75 perforation/{name}.npy", buffer)


def send_value():
    global freq
    global target
    while True:
        arduino.write(f'{target}\n'.encode())
        time.sleep(freq)  # Send value at 50 Hz


def listen_echo():
    global freq
    global buffer
    global new_state

    time.sleep(.5)

    while True:
        if arduino.in_waiting > 0:
            # Read all data in the buffer
            data = arduino.readlines(arduino.in_waiting)

            # Split the data into lines and take the last line
            if not data:
                continue

            latest_value = data[-1].decode('utf-8').strip().split()
            old_state = float(latest_value[0])
            control_action = float(latest_value[1])
            new_state = float(latest_value[2])

            if time.time() % (1 / 4) < 0.03:
                print(
                    f"State: {old_state:.2f}\tControl action: {control_action:.0f}\tNew state: {new_state}\tTarget: {target:.0f}")
            buffer.append([old_state, control_action, new_state, target])

        # if keyboard.is_pressed('s'):  # If 's' is pressed, save the buffer
        #     save_buffer("user_save")

        time.sleep(freq)
        # time.sleep(freq)  # Listen at 50 Hz


def change_value():
    global target
    global new_state
    global freq
    global buffer

    start_time = time.time()
    old_t = time.time()
    count = 0
    new_state = 30

    adapter = TargetAdapter(state_size=1)
    adapter.load_weights(f"./Models/adapter_weights.weights.h5")

    while True:
        if time.time() - old_t > 2:
            # true_target = np.random.randint(-22, 0)
            true_target = -3 if true_target == -21 else -21
            old_t = time.time()

        if (time.time() - start_time) % 300 < 0.025:
            save_buffer(f"fatigue_auto_save_{count}")
            buffer = []
            count += 1

        adapter_input = ops.array([*new_state, *true_target])[None]
        delta_target = adapter.predict(adapter_input, verbose=0)[0]
        target = target + delta_target

        time.sleep(freq)


def main():
    global arduino
    arduino = serial.Serial('COM5', 115200)

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


if __name__ == '__main__':
    main()
