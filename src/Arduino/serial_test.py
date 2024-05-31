import threading
import serial
import time

target = 28
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
    first_reading = True
    while True:
        if arduino.in_waiting > 0:
            # Read all data in the buffer
            data = arduino.read(arduino.in_waiting).decode('utf-8').strip()

            # Split the data into lines and take the last line
            lines = data.split('\n')
            if not lines:
                continue

            latest_value = lines[-1].strip().split()
            if first_reading:
                first_reading = False
                old_state = float(latest_value[1])
                continue

            new_state = float(latest_value[1])
            control_action = float(latest_value[0])
            if time.time() % (1 / 2) < 0.03:
                print(f"State: {old_state}\tControl action: {control_action}\tNew state: {new_state}\tTarget: {target}")
            buffer.append([old_state, control_action, new_state, target])
            old_state = new_state

        time.sleep(freq)
        # time.sleep(freq)  # Listen at 50 Hz


def change_value():
    global target
    global new_state
    global freq
    count = 0

    while True:
        time.sleep(freq)
        if new_state != float(target):
            count = 0
            continue
        elif count == 5:
            target = 28 if target == 22 else 22
            count = 0
            continue
        count += 1


def main():
    global arduino
    arduino = serial.Serial('COM4', 9600)

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
