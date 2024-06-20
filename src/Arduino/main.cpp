#include "pidControl.h"
#include <Servo.h>
#include <Wire.h>
#include <as5600.h>


AS5600 as5600;
Servo servo_9;
PidController controller(-.4, -.5, 0); // Initialize the PID controller

int servoPos = 500;
String incomingByte;
double target;
double beta;
double beta0;
double omega;

void setup() {
    Serial.begin(115200); // Start the serial communication
    servo_9.attach(9); // Attach the servo to pin 9
    servo_9.writeMicroseconds(servoPos); // Initialize the servo position

    Wire.begin();

    // Set up the AS5600
    as5600.begin(4);  //  set direction pin.
    as5600.setDirection(AS5600_COUNTERCLOCK_WISE);
//    int b = as5600.isConnected();
//    Serial.print("Connect: ");
//    Serial.println(b);

    delay(1000);
    beta0 = as5600.rawAngle() * AS5600_RAW_TO_DEGREES;
    beta = 0;
    omega = 0;
    target = 0;
}

void loop() {
    // measure state
    double old_beta = as5600.rawAngle() * AS5600_RAW_TO_DEGREES - beta0;
    double old_omega = as5600.getAngularSpeed();

    if (Serial.available() > 0) {
        while (Serial.available() > 0) {
            incomingByte = Serial.readStringUntil('\n');
        }
        target = incomingByte.toDouble();
    }
    // compute and execute control
//    servoPos = min(1700, max(550, target)); // direct control for testing
    double controlForce = controller.computeControl(beta, target, 0.02);
    double servoAngle = degrees((controlForce / 0.25 - 32 * radians(beta)) / 15);
    servoPos = map(servoAngle, 0, 75, 550, 1700);
//    servoPos = min(1700, max(550, servoPos));
    servo_9.writeMicroseconds(min(1700, max(550, servoPos)));

    delay(20);

    // measure resulting state
    beta = as5600.rawAngle() * AS5600_RAW_TO_DEGREES - beta0;
    omega = as5600.getAngularSpeed();

    // send to serial connection
    String message = String(old_beta) + " " + String(old_omega) + " " + String(servoPos) + " " + String(beta) + " " +
                     String(omega);
    Serial.println(message);
}
