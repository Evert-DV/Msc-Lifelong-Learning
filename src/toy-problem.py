import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Create a mass-spring-damper toy problem

# System model parameters
m = 5
l0 = 5
k = 10
c = 2
g = 9.81
dt = 0.01
t = np.arange(0, 50, dt)
x0 = [8, 0]

A = np.array([[0., 1.], [-k / m, -c / m]])
B = np.array([[0., 1 / m]]).T
C = np.array([[1., 0.]])
D = np.array([[0.]])
sys = sp.signal.StateSpace(A, B, C, D)

_, signal, _ = sp.signal.lsim(sys, np.ones(len(t)) * (m * g + k * l0), t, x0)

fig, ax = plt.subplots(1)
ax.plot(t, signal)

# System identification
frequencies = np.linspace(0.1, 10, 500)  # Frequency range in Hz
angular_frequencies = 2 * np.pi * frequencies  # Convert to rad/s
w, mag, phase = sp.signal.bode(sys)
phase = np.radians(phase[np.argmax(mag)])

# Periodic excitation
z = c / (2 * np.sqrt(m * k))
wn = np.sqrt(k / m) * np.sqrt(1 - z ** 2)
Tn = 2 * np.pi / wn
t_wait = phase / wn

X = 2
F0 = X * np.sqrt((k - m * wn ** 2) ** 2 + (c * wn) ** 2)
F = F0 * np.cos(wn * t + phase)

impulse = np.sum(np.abs(F[:int(Tn/dt)] * dt))

F0_inst = impulse / dt
F_inst = np.zeros(len(t))
F_inst[int((Tn - t_wait)/dt)::int(Tn / dt)] = F0_inst

_, signal2, _ = sp.signal.lsim((A, B, C, D), np.ones(len(t)) * (m * g + k * l0) + F, t, x0)
_, signal3, _ = sp.signal.lsim((A, B, C, D), np.ones(len(t)) * (m * g + k * l0) + F_inst, t, x0)

ax.plot(t, signal2)
ax.plot(t, signal3)

ax.invert_yaxis()
plt.savefig("./tmp/plot.png", dpi=200)
