#include "pidControl.h"
#include <Servo.h>
#include <Wire.h>
#include <as5600.h>


AS5600 as5600;
Servo servo_9;
PidController controller(-.35, -.3, .0); // Initialize the PID controller (default)
//PidController controller(-.45, -.4, .0); // Initialize the PID controller (barely stable)
//PidController controller(-.6, -.45, .00); // Initialize the PID controller (hard)
//PidController controller(-.15, -.15, .0); // Initialize the PID controller (soft)
// PID values kp, ki, kd:
// 75-75: -.7, -.5, .001
// 150 - 50: -.35, -.3, .001

int servoPos = 540;
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
    while (b == 0) { // If the sensor is not connected, halt the program
        // Infinite loop to prevent proceeding to loop()
        delay(1000); // Delay to reduce CPU usage, not necessary but nice to have
        Serial.println("AS5600 not connected. Please check the sensor.");
        b = as5600.isConnected();
    }

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
    servoPos = map(servoAngle, 0, 125, 540, 1800);
    servoPos = min(1800, max(540, servoPos));
    servo_9.writeMicroseconds(servoPos);
//    servo_9.write(target);
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
