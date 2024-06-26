#include "pidControl.h"
#include <Servo.h>
#include <Wire.h>
#include <as5600.h>


AS5600 as5600;
Servo servo_9;
PidController controller(-.25, -.5, .002); // Initialize the PID controller

int servoPos = 550;
String incomingByte;
double target;
double beta;
double beta0;
double omega;
//unsigned long lastTime;
//unsigned long currentTime;
double dt = 0.017;
bool first = true;
unsigned long int counter = 0;

void setup() {
    Serial.begin(115200); // Start the serial communication
    servo_9.attach(9); // Attach the servo to pin 9
    servo_9.writeMicroseconds(servoPos); // Initialize the servo position

    Wire.begin();

    // Set up the AS5600
    int b = as5600.isConnected();
    Serial.print("Connect: ");
    Serial.println(b);

    delay(500);
    beta0 = as5600.rawAngle() * AS5600_RAW_TO_DEGREES;
    beta = 0;
    omega = 0;
    target = 0;
    dt = 0.02;
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

//    currentTime = millis();
//    if (not first) {
//        dt = (currentTime - lastTime) / 1000.0;
//    }
//    lastTime = currentTime;

    // compute control
    double controlForce = controller.computeControl(beta, target, dt);
    double servoAngle = degrees((controlForce / .23 - 35 * radians(old_beta)) / 15);
    servoPos = map(servoAngle, 0, 110, 550, 1700);
    servoPos = min(1700, max(550, servoPos));
    servo_9.writeMicroseconds(servoPos);

    delay(17);

    // measure resulting state
    beta = as5600.rawAngle() * AS5600_RAW_TO_DEGREES - beta0;
    omega = as5600.getAngularSpeed();

    // send to serial connection
    String message = String(counter) + " " + String(old_beta) + " " + String(old_omega) + " " + String(servoPos) + " " + String(beta) + " " +
                     String(omega) + " " + String(target);
    Serial.println(message);

    if (first) {
        first = false;
    }
    counter++;
}
