#include "pidControl.h"
#include <Servo.h>

Servo servo_9;  // Create a servo object to control a servo
int servoPos = 0; // Start position of the servo at 90 degrees
PidController controller(0.5, 0.1, 0.1);
String servoPosStr;

void setup() {
    Serial.begin(9600); // Start the serial communication
    servo_9.attach(9, 500, 2500); // Attach the servo to pin 9 with min and max pulse widths
    servo_9.write(servoPos); // Initialize the servo position// Initialize the PID controller
}

void loop() {
//    servoPos = keyboard(servo_9, servoPos);
    if (Serial.available() > 0) {
//        double target = Serial.read();
//        servoPos = controller.computeControl(servoPos, target, 0.01);
        servoPosStr = Serial.readStringUntil('\n');
        servoPos = servoPosStr.toInt();
//        servo_9.write(servoPos);
        Serial.println(servoPos);
        delay(20);
    }
}
