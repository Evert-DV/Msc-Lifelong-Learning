#include "pidControl.h"

PidController::PidController(double kp, double ki, double kd)
        : kp_(kp), kd_(kd), ki_(ki), integral_error_(0.0), previous_error_(0.0) {}

PidController::~PidController() {}

int PidController::computeControl(double state, double target, double dt) {
    double controlForce;
    int controlAngle;

    double error = state - target;
    integral_error_ += error * dt;
    double derivative_error = (error - previous_error_) / dt;
    previous_error_ = error;

    controlForce = kp_ * error + ki_ * integral_error_ + kd_ * derivative_error;
    controlForce = max(0, min(controlForce, 10 * kp_ + 10 * kd_ + 10 * ki_));
    controlAngle = map(controlForce, 0, 10 * kp_ + 10 * kd_ + 10 * ki_, 0, 75);
    controlAngle = min(75, max(0, controlAngle));

    return controlAngle;
}