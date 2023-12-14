import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# Create a mass-spring-damper toy problem

class System:
    def __init__(self, m, k, c, l0=0, g=9.81):
        self.A = np.array([[0., 1.], [-k / m, -c / m]])
        self.B = np.array([[0., 1 / m]]).T
        self.C = np.array([[1., 0.]])
        self.D = np.array([[0.]])
        self.sys = sp.signal.StateSpace(self.A, self.B, self.C, self.D)
        self.u = m * g + k * l0

    def response(self, s, a=0, dt=0.01):
        _, _, s = sp.signal.lsim(self.sys, 2 * [self.u + a], [0, dt], s)

        return s[-1]


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


np.random.seed(15)

system = System(5, 10, 3, 5)
controller = PIDController(0.35*600, 1.5*600*0.57/8, 2*0.6*600/0.57)  # w/ kp_ult = 600 and T_ult = 0.57

dt = 0.01
x0 = [9.9, 0]  # 9.9 was found to be the steady state
signal = []
targets = []
t = np.arange(0, 50, dt)

for ti in t:
    if ti % 15 == 0:
        target = np.random.rand(1) * 6 + 7
    targets.append(target)
    a = controller.compute_control(x0, target, dt)
    x = system.response(x0, a)
    signal.append(x)
    x0 = x

signal = np.asarray(signal)

fig, ax = plt.subplots(1)
ax.plot(t, signal[:, 0])
ax.plot(t, targets)
ax.invert_yaxis()
plt.savefig("./tmp/plot.png", dpi=300)

# # System identification
# z = c / (2 * np.sqrt(m * k))
# wn = np.sqrt(k / m) * np.sqrt(1 - z ** 2)
#
# # Problem setup
#
# target_pos = np.random.rand()
# print(f"Target: {target_pos}")
