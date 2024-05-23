#include <Arduino.h>
#include <Servo.h>

Servo servo_9;  // Create a servo object to control a servo
int servoPos = 0; // Start position of the servo at 90 degrees

void setup() {
    Serial.begin(9600); // Start the serial communication
    servo_9.attach(9, 500, 2500); // Attach the servo to pin 9 with min and max pulse widths
    servo_9.write(servoPos); // Initialize the servo position
}

void loop() {
    if (Serial.available() > 0) {
        char command = Serial.read(); // Read the incoming byte

        if (command == 'u') { // 'u' for up arrow key
            servoPos += 2; // Increase the servo position
            if (servoPos > 85) servoPos = 85; // Limit to maximum angle
        } else if (command == 'd') { // 'd' for down arrow key
            servoPos -= 2; // Decrease the servo position
            if (servoPos < 5) servoPos = 5; // Limit to minimum angle
        } else if (command == 'q') { // 'q' for quit
            servo_9.write(5);
            Serial.println("Quitting the program");
            delay(1000); // Wait for 1 second
            exit(0); // Exit the program
        } else if (command == 'a') { // 'a' for angle
            Serial.print("Servo Position: ");
            Serial.println(servoPos);
        }

        servo_9.write(servoPos); // Move the servo to the new position
    }
}
