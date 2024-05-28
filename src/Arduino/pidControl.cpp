#include "pidControl.h"

int PidController::computeControl(double state, double target, double dt) {
    double controlForce;
    int controlAngle;

    double error = target - state;
    integral_error_ += error * dt;
    double derivative_error = (error - previous_error_) / dt;
    previous_error_ = error;

    controlForce =  kp_ * error + ki_ * integral_error_ + kd_ * derivative_error;
    controlAngle = map(controlForce, -255, 255, 5, 75);
    Serial.println(controlForce);

    return controlAngle;
}