#include "keyboardControl.h"

Servo servo_9;  // Create a servo object to control a servo
int servoPos = 0; // Start position of the servo at 90 degrees

void setup() {
    Serial.begin(9600); // Start the serial communication
    servo_9.attach(9, 500, 2500); // Attach the servo to pin 9 with min and max pulse widths
    servo_9.write(servoPos); // Initialize the servo position
}

void loop() {
    servoPos = keyboard(servo_9, servoPos);
}
