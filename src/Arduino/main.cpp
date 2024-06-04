#include "pidControl.h"
#include <Servo.h>

Servo servo_9;  // Create a servo object to control a servo
int servoPos = 0;
PidController controller(-16, 0, 0); // Initialize the PID controller
String incomingByte;
double target;
double beta;
double kos;
double fx;

void setup() {
    Serial.begin(9600); // Start the serial communication
    servo_9.attach(9, 500, 2500); // Attach the servo to pin 9
    servo_9.write(servoPos); // Initialize the servo position
    beta = 0;
    kos = 30;
}

void loop() {
    if (Serial.available() > 0) {
        // measure state
        double old_kos = kos;

        // read target
        incomingByte = Serial.readStringUntil('\n');
        target = incomingByte.toDouble();

        // compute and execute control
        servoPos = controller.computeControl(kos, target, 0.02);
        //        servo_9.write(servoPos);

        fx = (30 - kos) * 0.4;
        beta = degrees(
                radians(beta) + 0.5 * (-2 * fx * 52 + .25 * 37 * 15 * radians(servoPos)) * 0.02 * 0.02); // dummy state
        beta = min(28, max(-9, beta));

        // measure state
        delay(20);
        kos = sqrt(43 * 43 - pow(31 + 52 * radians(beta), 2)); //dummy state

        Serial.print(old_kos);
        Serial.print(" ");
        Serial.print(servoPos);
        Serial.print(" ");
        Serial.println(kos);
    }
}
