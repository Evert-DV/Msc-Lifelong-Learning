#include "pidControl.h"
#include <Servo.h>
#include <Wire.h>
#include <as5600.h>


AS5600 as5600;
Servo servo_9;
PidController controller(-.5, 0, 0); // Initialize the PID controller

int servoPos = 500;
String incomingByte;
double target;
double beta;
double beta0;

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
    target = 0;
}

void loop() {
    // measure state
    double old_beta = as5600.rawAngle() * AS5600_RAW_TO_DEGREES - beta0;

    if (Serial.available() > 0) {
        // read target
        incomingByte = Serial.readStringUntil('\n');
        if (incomingByte.toDouble() != target) {
            target = incomingByte.toDouble();
        }
    }
    // compute and execute control
//    servoPos = min(1700, max(550, target)); // direct control for testing
    double controlForce = controller.computeControl(beta, target, 0.02);
    double servoAngle = degrees((controlForce / 0.25 - 32 * radians(beta)) / 15);
    servoPos = map(servoAngle, 0, 75, 550, 1700);
    servoPos = min(1550, max(550, servoPos));
    servo_9.writeMicroseconds(servoPos);

    delay(20);

    // measure resulting state
    beta = as5600.rawAngle() * AS5600_RAW_TO_DEGREES - beta0;

    // send to serial connection
    String message = String(old_beta) + " " + String(servoPos) + " " + String(beta);
    Serial.println(message);
}
