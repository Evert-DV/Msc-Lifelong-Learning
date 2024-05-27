//
// Created by evert on 27/05/2024.
//

#include "keyboardControl.h"


int keyboard(Servo servo, int servoPos) {
    if (Serial.available() > 0) {
        char command = Serial.read(); // Read the incoming byte

        if (command == 'u') { // 'u' for up arrow key
            servoPos = 75; // Increase the servo position
        } else if (command == 'd') { // 'd' for down arrow key
            servoPos = 5; // Decrease the servo position
        } else if (command == 'q') { // 'q' for quit
            servo.write(5);
            Serial.println("Quitting the program");
            delay(1000); // Wait for 1 second
            exit(0); // Exit the program
        } else if (command == 'a') { // 'a' for angle
            Serial.print("Servo Position: ");
            Serial.println(servoPos);
        }

        servo.write(servoPos); // Move the servo to the new position
    }
    return servoPos;
}