import serial
import time

# Replace 'COM3' with your Arduino port
ser = serial.Serial('COM4', 9600)
time.sleep(2)  # Wait for the serial connection to initialize


def send_command(command):
    ser.write(command.encode())


print("Use the left and right arrow keys to control the servo.")
angles = range(0, 75, 5)
i = 0

try:
    while True:
        command = input("Enter command: ")
        send_command(command)
        print(f"Sent command: {command}")
finally:
    ser.close()
