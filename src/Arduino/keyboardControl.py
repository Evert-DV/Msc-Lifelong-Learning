import serial
import keyboard
import time

# Set up the serial connection (change 'COM3' to your actual Arduino port)
arduino = serial.Serial('COM4', 9600, timeout=1)

def send_command(command):
    arduino.write(command.encode())

def main():
    print("Press the up/down arrow keys to control the servo. Press 'q' to quit.")
    wait = 1/60  # 60 Hz

    try:
        while True:
            if keyboard.is_pressed('up'):
                send_command('u')
                print("Sending 'u' command to increase position")
                time.sleep(wait)  # Send command at intervals (adjust as needed)
            elif keyboard.is_pressed('down'):
                send_command('d')
                print("Sending 'd' command to decrease position")
                time.sleep(wait)  # Send command at intervals (adjust as needed)
            elif keyboard.is_pressed('q'):
                print("Exiting...")
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        arduino.close()

if __name__ == '__main__':
    main()
