#include "pidControl.h"
#include <Servo.h>
#include <Wire.h>
#include <as5600.h>

AS5600 as5600;
Servo servo_9;

int servoPos = 0;
PidController controller(2.5, 0.75, 0.005); // Initialize the PID controller
String incomingByte;
double target;
double beta;
double kos;
double fx;
double beta0;

void setup() {
    Serial.begin(115200); // Start the serial communication
    servo_9.attach(9, 500, 2500); // Attach the servo to pin 9
    servo_9.write(servoPos); // Initialize the servo position

    Wire.begin();

    as5600.begin(4);  //  set direction pin.
    as5600.setDirection(AS5600_CLOCK_WISE);  //  default, just be explicit.
    int b = as5600.isConnected();
    Serial.print("Connect: ");
    Serial.println(b);

    delay(1000);
    beta0 = as5600.rawAngle() * AS5600_RAW_TO_DEGREES;
}

void loop() {
    if (Serial.available() > 0) {
        // measure state
        double old_beta = as5600.rawAngle() * AS5600_RAW_TO_DEGREES - beta0;

        // read target
        incomingByte = Serial.readStringUntil('\n');
        target = incomingByte.toDouble();

//        servoPos = min(75, max(0, target)); // direct control for testing

        // compute and execute control
        double controlForce = controller.computeControl(beta, target, 0.02);
//        servoPos = degrees((controlForce / 0.25 + 37 * radians(beta)) / 15);
        servoPos = min(75, max(0, controlForce));
        servo_9.write(servoPos);
//
//        fx = (30 - kos) * 0.2;
//        beta = degrees(
//                radians(beta) + 0.5 * (-2 * fx * 52 + .25 * 37 * (15 * radians(servoPos) - 37 * radians(beta))) * 0.02 *
//                                0.02); // dummy state
//        beta = min(28, max(-9, beta));
//
        // measure state
//        kos = sqrt(43 * 43 - pow(31 + 52 * radians(beta), 2)); //dummy state
        beta = as5600.rawAngle() * AS5600_RAW_TO_DEGREES - beta0;
        delay(20);

        String message = String(old_beta) + " " + String(servoPos) + " " + String(beta);
        Serial.println(message);
    }
}
