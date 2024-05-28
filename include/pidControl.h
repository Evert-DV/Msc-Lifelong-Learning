#ifndef MSC_LIFELONG_LEARNING_PIDCONTROL_H
#define MSC_LIFELONG_LEARNING_PIDCONTROL_H

#include <Arduino.h>

class PidController {
public:
    double kp_, kd_, ki_;
    double integral_error_;
    double previous_error_;

    PidController(double kp, double ki, double kd) : kp_(kp), ki_(ki), kd_(kd), integral_error_(0),
                                                     previous_error_(0) {}
    ~PidController() {}

    int computeControl(double state, double target, double dt);
};

#endif //MSC_LIFELONG_LEARNING_PIDCONTROL_H
