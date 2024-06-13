#include "pidControl.h"

PidController::PidController(double kp, double ki, double kd)
        : kp_(kp), ki_(ki), kd_(kd), integral_error_(0.0), previous_error_(0.0) {}

PidController::~PidController() {}

double PidController::computeControl(double state, double target, double dt) {
    double controlForce;

    double error = target - state;
    integral_error_ += error * dt;
    double derivative_error = (error - previous_error_) / dt;
    previous_error_ = error;

    controlForce = kp_ * error + ki_ * integral_error_ + kd_ * derivative_error;

    return controlForce;
}

void PidController::reset_integral() {
    integral_error_ = 0.0;
}