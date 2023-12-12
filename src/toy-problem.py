import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Create a mass-spring-damper toy problem

# System model parameters
m = 10

l0 = 5
k = 5

c = 2

g = 9.81
t = np.arange(0, 50, 0.01)
x0 = [5, 0]

A = np.array([[0., 1.], [-k / m, -c / m]])
B = np.array([[0., 1 / m]]).T
C = np.array([[1., 0.], [0., 1.]])
D = np.array([[0., 0]]).T

_, signal, _ = sp.signal.lsim((A, B, C, D), np.ones(len(t)) * (m * g + k * l0), t, x0)

plt.plot(t, signal[:, 0])
plt.show()

# Agent
