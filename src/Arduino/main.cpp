#include "pidControl.h"
#include <Servo.h>
#include <Wire.h>
#include <as5600.h>
#include <MPU9250_WE.h>

#define MPU6500_ADDR 0x68

AS5600 as5600;
MPU6500_WE mpu6500 = MPU6500_WE(MPU6500_ADDR);
Servo servo_9;
PidController controller(2.5, 0.75, 0.005); // Initialize the PID controller

int servoPos = 0;
String incomingByte;
double target;
xyzFloat gValue;
double beta;
double theta;
double beta0;
double theta0;

void setup() {
    Serial.begin(115200); // Start the serial communication
    servo_9.attach(9, 500, 2500); // Attach the servo to pin 9
    servo_9.write(servoPos); // Initialize the servo position

    Wire.begin();

    // Set up the AS5600
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

    // Set up the MPU6500
    mpu6500.enableGyrDLPF();
    mpu6500.setGyrDLPF(MPU6500_DLPF_6);
    mpu6500.setSampleRateDivider(5);
    mpu6500.setGyrRange(MPU6500_GYRO_RANGE_250);
    mpu6500.setAccRange(MPU6500_ACC_RANGE_2G);
    mpu6500.enableAccDLPF(true);
    mpu6500.setAccDLPF(MPU6500_DLPF_6);

    delay(1000);

    gValue = mpu6500.getGValues();
    theta0 = degrees(atan(-gValue.x / (gValue.y * gValue.y + gValue.z * gValue.z)));
    beta0 = as5600.rawAngle() * AS5600_RAW_TO_DEGREES;
    beta = 0;
    theta = 0;
}

void loop() {
    if (Serial.available() > 0) {
        // measure state
        gValue = mpu6500.getGValues();
        double old_theta = degrees(atan(-gValue.x / (gValue.y * gValue.y + gValue.z * gValue.z))) - theta0;
        beta = as5600.rawAngle() * AS5600_RAW_TO_DEGREES - beta0;

        // read target
        incomingByte = Serial.readStringUntil('\n');
        target = incomingByte.toDouble();

        // compute and execute control
        servoPos = min(75, max(0, target)); // direct control for testing
//        double controlForce = controller.computeControl(theta, target, 0.02);
//        servoPos = degrees((controlForce / 0.25 + 37 * radians(beta)) / 15);
//        servoPos = min(75, max(0, controlForce));
        servo_9.write(servoPos);

        delay(20);

        // measure state
        gValue = mpu6500.getGValues();
        theta = degrees(atan(-gValue.x / (gValue.y * gValue.y + gValue.z * gValue.z))) - theta0;
//        double roll = degrees(atan(gValue.y / (gValue.x * gValue.x + gValue.z * gValue.z)));

        // send to serial connection
        String message = String(old_theta) + " " + String(servoPos) + " " + String(theta);
        Serial.println(message);
    }
}
