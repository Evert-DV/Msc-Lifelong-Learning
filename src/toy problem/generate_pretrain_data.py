import numpy as np
import matplotlib.pyplot as plt
from toy_tools import PIDController, System


np.random.seed(42)

x0 = [9.9, 0]  # 9.9 was found to be the steady state
target = np.array([11., 0.])
signal = []
targets = []
controls = []
data = []

controller = PIDController(350, 107.5, 1257)
system = System(5, 10, 3, 5)

dt = 1 / 60
t = np.arange(0, 600, dt)

for ti in t:
    if ti % 10 == 0:
        target = [np.random.rand() * 6 + 7, 0.]

    targets.append(target)
    control_action = controller.compute_control(x0, target, dt)
    x = system.response(x0, control_action, do_update=False)
    signal.append(x)
    data.append([*x0, control_action, *x, *target])

    x0 = x

data = np.asarray(data)
np.save("tmp/pretrain_data.npy", data)

signal = np.asarray(signal)
targets = np.asarray(targets)

plt.plot(t, signal[:, 0], label="signal")
plt.plot(t, targets[:, 0], '--', label="target")
plt.show()