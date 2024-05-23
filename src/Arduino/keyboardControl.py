import serial
import keyboard
import time
import threading


def send_command(command):
    arduino.write(command.encode())


def read_serial():
    global servo_position
    while True:
        if arduino.in_waiting > 0:  # Check if there's data to read
            line = arduino.readline().decode('utf-8').rstrip()  # Read the line, decode to string, and strip newlines
            with position_lock:
                servo_position = line  # Update the shared variable
            print(f"\r{line}", end="")  # Print the received data
        time.sleep(0.1 / 50)  # Small delay to prevent CPU overuse


def ask_angle():
    while True:
        if keyboard.is_pressed('a'):
            send_command('a')  # Wait for key release with a short timeout
            time.sleep(.2)

def main():
    # print("Press the up/down arrow keys to control the servo. Press 'q' to quit.")
    wait = 1 / 60  # 60 Hz

    try:
        while True:
            if keyboard.is_pressed('up'):
                send_command('u')
                # print("Sending 'u' command to increase position")
                time.sleep(wait)  # Send command at intervals (adjust as needed)
            elif keyboard.is_pressed('down'):
                send_command('d')
                #                 print("Sending 'd' command to decrease position")
                time.sleep(wait)  # Send command at intervals (adjust as needed)
            elif keyboard.is_pressed('q'):
                send_command('q')
                print("\nExiting...")
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        arduino.close()


if __name__ == '__main__':
    # Set up the serial connection (change 'COM3' to your actual Arduino port)
    arduino = serial.Serial('COM4', 9600, timeout=1)

    # Shared variable and lock
    servo_position = None
    position_lock = threading.Lock()

    # Create and start the serial reading thread
    serial_thread = threading.Thread(target=read_serial)
    serial_thread.daemon = True  # This makes the thread exit when the main program exits
    serial_thread.start()

    angle_thread = threading.Thread(target=ask_angle)
    angle_thread.daemon = True  # This makes the thread exit when the main program exits
    angle_thread.start()

    try:
        # Run the main task
        main()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        arduino.close()  # Close the serial connection when done
