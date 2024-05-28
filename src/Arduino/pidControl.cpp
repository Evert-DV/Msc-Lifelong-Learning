#include "pidControl.h"

int PidController::computeControl(double state, double target, double dt) {
    double controlForce;
    int controlAngle;

    double error = state - target;
    integral_error_ += error * dt;
    double derivative_error = (error - previous_error_) / dt;
    previous_error_ = error;

    controlForce =  kp_ * error + ki_ * integral_error_ + kd_ * derivative_error;
    controlForce = max(0, controlForce);
    controlAngle = map(controlForce, 0, 10 * ki_, 0, 75);

    return controlAngle;
}