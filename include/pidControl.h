#ifndef MSC_LIFELONG_LEARNING_PIDCONTROL_H
#define MSC_LIFELONG_LEARNING_PIDCONTROL_H

#include <Arduino.h>

class PidController {
public:
    double kp_, ki_, kd_;
    double integral_error_;
    double previous_error_;

    PidController(double kp, double ki, double kd);
    ~PidController();

    double computeControl(double state, double target, double dt);
    void reset_integral();
};

#endif //MSC_LIFELONG_LEARNING_PIDCONTROL_H
