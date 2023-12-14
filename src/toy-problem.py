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
        _, _, s = sp.signal.lsim(self.sys, [self.u + a, self.u], [0, dt], s)

        return s[-1]


dt = 0.01
system = System(5, 10, 3, 5)
x0 = [8, 0]
signal = []
t = np.arange(0, 50, dt)

for _ in t:
    x = system.response(x0)
    signal.append(x)
    x0 = x

steady_state = signal[-1]
print(f"Steady state: {steady_state[0]:.2f}")

fig, ax = plt.subplots(1)
ax.plot(t, signal)
ax.invert_yaxis()
plt.savefig("./tmp/plot.png")


# # System identification
# z = c / (2 * np.sqrt(m * k))
# wn = np.sqrt(k / m) * np.sqrt(1 - z ** 2)
#
# # Problem setup
#
# target_pos = np.random.rand()
# print(f"Target: {target_pos}")
