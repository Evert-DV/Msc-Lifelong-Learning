#include "pidControl.h"
#include <Servo.h>

Servo servo_9;  // Create a servo object to control a servo
int servoPos = 0;
PidController controller(2, 5, .01); // Initialize the PID controller
String incomingByte;
double target;
double state;
double kos;
double fx;

void setup() {
    Serial.begin(9600); // Start the serial communication
    servo_9.attach(9, 500, 2500); // Attach the servo to pin 9
    servo_9.write(servoPos); // Initialize the servo position
    state = 0;
    kos = 30;
}

void loop() {
    fx = (30 - kos) * 0.75;
    state = degrees(
            radians(state) + 0.5 * (-2 * fx * 52 + 2 * 37 * 15 * radians(servoPos)) * 0.02 * 0.02); // dummy state
    state = min(8.5, max(0, state));
    kos = sqrt(46 * 46 - pow(35 + 52 * radians(state), 2));

    if (Serial.available() > 0) {
        incomingByte = Serial.readStringUntil('\n');
        target = incomingByte.toDouble();
        servoPos = controller.computeControl(state, target, 0.02);
//        servo_9.write(servoPos);
        Serial.print(servoPos);
        Serial.print(" ");
        Serial.println(state);
    }
    delay(20);
}
