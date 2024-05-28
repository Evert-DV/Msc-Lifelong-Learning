import threading
import serial
import time

value = 29
arduino = None


def send_value():
    global value
    while True:
        arduino.write(f'{value}\n'.encode())
        time.sleep(1 / 50)  # Send value at 50 Hz


def listen_echo():
    while True:
        if arduino.in_waiting > 0:
            # Read all data in the buffer
            data = arduino.read(arduino.in_waiting).decode('utf-8').strip()
            # Split the data into lines and take the last line
            lines = data.split('\n')
            if lines:
                latest_value = lines[-1].strip()
                print(f"Arduino says: {latest_value}")
        time.sleep(1/2)  # Check for new data at 50 Hz


def change_value():
    global value
    while True:
        new_value = input("\nEnter new value: ")
        if new_value.isdigit():
            value = int(new_value)
        else:
            print("Please enter an integer.")


def main():
    global arduino
    arduino = serial.Serial('COM4', 9600)  # Replace 'COM3' with your Arduino's serial port

    # Create threads for sending and receiving data
    send_thread = threading.Thread(target=send_value)
    listen_thread = threading.Thread(target=listen_echo)
    input_thread = threading.Thread(target=change_value)

    # Start threads
    send_thread.start()
    listen_thread.start()
    input_thread.start()

    # Ensure main thread waits for the completion of other threads
    send_thread.join()
    listen_thread.join()
    input_thread.join()


if __name__ == '__main__':
    main()
