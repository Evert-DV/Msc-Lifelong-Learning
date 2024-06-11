import threading
import serial
import time

target = -5
arduino = None
new_state = None
freq = 1 / 50
buffer = []


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

            if time.time() % (1 / 2) < 0.03:
                print(f"State: {old_state}\tControl action: {control_action}\tNew state: {new_state}\tTarget: {target}")
            buffer.append([old_state, control_action, new_state, target])

        time.sleep(freq)
        # time.sleep(freq)  # Listen at 50 Hz


def change_value():
    global target
    global new_state
    global freq

    while True:
        time.sleep(freq)
        target = int(input("Enter new target: "))
    # count = 0
    # new_state = 30
    #
    # while True:
    #     time.sleep(freq)
    #     if abs(new_state - target) > .1:
    #         count = 0
    #         continue
    #     elif count == 5:
    #         target = 23 if target == 28 else 28
    #         count = 0
    #         continue
    #     count += 1


def main():
    global arduino
    arduino = serial.Serial('COM5', 115200)

    # Create threads for sending and receiving data
    send_thread = threading.Thread(target=send_value)
    listen_thread = threading.Thread(target=listen_echo)
    target_thread = threading.Thread(target=change_value)

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
