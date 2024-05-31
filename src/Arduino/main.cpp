#include "pidControl.h"
#include <Servo.h>

Servo servo_9;  // Create a servo object to control a servo
int servoPos = 0;
PidController controller(30, 20, 5); // Initialize the PID controller
String incomingByte;
double target;
double state;

void setup() {
    Serial.begin(9600); // Start the serial communication
    servo_9.attach(9, 500, 2500); // Attach the servo to pin 9
    servo_9.write(servoPos); // Initialize the servo position
    state = 30;
}

void loop() {
//    servoPos = keyboard(servo_9, servoPos);
    state = state + (10 * (30 - state) - 100 * servoPos * 3.14 / 180) * 0.02 * 0.02; // dummy state
    state = min(30, max(20, state));

    if (Serial.available() > 0) {
        incomingByte = Serial.readStringUntil('\n');
        target = incomingByte.toDouble();
        servoPos = controller.computeControl(state, target, 0.02);
        servo_9.write(servoPos);
        Serial.print(servoPos);
        Serial.print(" ");
        Serial.println(state);
    }
    delay(20);
}
