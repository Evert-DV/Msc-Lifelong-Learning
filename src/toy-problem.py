import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# Create a mass-spring-damper toy problem

class System:
    def __init__(self, m, k, c, l0=0, g=9.81):
        self.m = m
        self.k = k
        self.c = c
        self.l0 = l0
        self.g = g
        self.u = m * g + k * l0

        self.A = np.array([[0., 1.], [-k / m, -c / m]])
        self.B = np.array([[0., 1 / m]]).T
        self.C = np.array([[1., 0.]])
        self.D = np.array([[0.]])
        self.sys = sp.signal.StateSpace(self.A, self.B, self.C, self.D)

    def response(self, s, a=0, dt=0.01, do_update=False):
        _, _, s = sp.signal.lsim(self.sys, 2 * [self.u + a], [0, dt], s)

        if do_update:
            self.update()

        return s[-1]

    def update(self):
        self.k *= 0.995
        self.c *= 0.995
        self.l0 += 1 / self.l0
        self.u = self.m * self.g + self.k * self.l0

        self.A = np.array([[0., 1.], [-self.k / self.m, -self.c / self.m]])
        self.B = np.array([[0., 1 / self.m]]).T
        self.C = np.array([[1., 0.]])
        self.D = np.array([[0.]])
        self.sys = sp.signal.StateSpace(self.A, self.B, self.C, self.D)


class PIDController:
    def __init__(self, kp, kd, ki):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.integral_error = 0.

    def compute_control(self, current_state, target_pos, dt):
        position_error = target_pos - current_state[0]
        velocity_error = -current_state[1]
        self.integral_error += position_error * dt

        control_action = self.kp * position_error + self.kd * velocity_error + self.ki * self.integral_error

        return control_action


def main():
    np.random.seed(15)

    system = System(5, 10, 3, 5)
    controller = PIDController(350, 107.5, 1257)

    dt = 1 / 60
    x0 = [9.9, 0]  # 9.9 was found to be the steady state
    signal = []
    targets = []
    controls = []
    t = np.arange(0, 45, dt)
    target = None
    buffer = []

    for ti in t:
        if ti % 15 == 0:
            target = np.random.rand(1) * 6 + 7
        targets.append(target)
        a = controller.compute_control(x0, target, dt)
        controls.append(a)
        x = system.response(x0, a, do_update=True)
        signal.append(x)
        buffer.append([*x0, *a, *x, *target])
        x0 = x

    signal = np.asarray(signal)

    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(t, signal[:, 0])
    ax[0].plot(t, targets)
    ax[0].invert_yaxis()

    ax[1].plot(t, controls)
    ax[1].invert_yaxis()

    fig.tight_layout()
    plt.savefig("./tmp/plot.png", dpi=300)

    # # System identification
    # z = c / (2 * np.sqrt(m * k))
    # wn = np.sqrt(k / m) * np.sqrt(1 - z ** 2)


if __name__ is "__main__":
    main()
