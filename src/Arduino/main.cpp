#include "pidControl.h"
#include <Servo.h>
#include <Wire.h>
#include <as5600.h>
#include <MPU9250_WE.h>

#define MPU6500_ADDR 0x68

AS5600 as5600;
MPU6500_WE mpu6500 = MPU6500_WE(MPU6500_ADDR);
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
//    Serial.print("Connect: ");
//    Serial.println(b);

    if(!mpu6500.init()){
//        Serial.println("MPU6500 does not respond");
    }
    else{
//        Serial.println("MPU6500 is connected");
    }

    mpu6500.enableGyrDLPF();
    mpu6500.setGyrDLPF(MPU6500_DLPF_6);
    mpu6500.setSampleRateDivider(5);
    mpu6500.setGyrRange(MPU6500_GYRO_RANGE_250);
    mpu6500.setAccRange(MPU6500_ACC_RANGE_2G);
    mpu6500.enableAccDLPF(true);
    mpu6500.setAccDLPF(MPU6500_DLPF_6);

    delay(1000);
//    beta0 = as5600.rawAngle() * AS5600_RAW_TO_DEGREES;
    xyzFloat gValue = mpu6500.getGValues();
    double beta0 = degrees(atan(gValue.y / (gValue.x * gValue.x + gValue.z * gValue.z)));;
    beta = 0;

}

void loop() {
    if (Serial.available() > 0) {
        // measure state
        double old_beta = beta;
//        double old_beta = as5600.rawAngle() * AS5600_RAW_TO_DEGREES - beta0;

        // read target
        incomingByte = Serial.readStringUntil('\n');
        target = incomingByte.toDouble();

        servoPos = min(75, max(0, target)); // direct control for testing

        // compute and execute control
//        double controlForce = controller.computeControl(beta, target, 0.02);
//        servoPos = degrees((controlForce / 0.25 + 37 * radians(beta)) / 15);
//        servoPos = min(75, max(0, controlForce));
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
//        beta = as5600.rawAngle() * AS5600_RAW_TO_DEGREES - beta0;
        delay(20);

        xyzFloat gValue = mpu6500.getGValues();
//        double pitch = degrees(atan(gValue.x / (gValue.y * gValue.y + gValue.z * gValue.z)));
//        double roll = degrees(atan(gValue.y / (gValue.x * gValue.x + gValue.z * gValue.z)));
        double beta = degrees(atan(gValue.y / (gValue.x * gValue.x + gValue.z * gValue.z))); - beta0;

        String message = String(old_beta) + " " + String(servoPos) + " " + String(beta);
        Serial.println(message);
    }
}
