#include <Arduino.h>
#include <Servo.h>

Servo servo_9;
int pos = 0;  // variable to store the servo position

void setup() {
    servo_9.attach(9, 500, 2500);  // attaches the servo on pin 9 to the servo object
    Serial.begin(9600);            // initialize serial communication at 9600 bits per second
    servo_9.write(pos);
}

void loop() {
    // Check if any data is available to read from the serial buffer
    if (Serial.available() > 0) {
        // Read the incoming byte as an integer
        int incomingByte = Serial.parseInt();

        // Check if the input is within the valid range
        if (incomingByte >= 0 && incomingByte <= 180) {
            pos = incomingByte;  // Set the servo position to the input value
            servo_9.write(pos);  // Move the servo to the specified position
            Serial.print("Moving to position: ");
            Serial.println(pos);  // Print the position to the Serial Monitor
        } else {
            Serial.println("Invalid input. Please enter a value between 0 and 180.");
        }
    }
    delay(15);
}
