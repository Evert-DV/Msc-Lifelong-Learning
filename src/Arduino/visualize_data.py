import numpy as np
from matplotlib import pyplot as plt


data = np.load("Dynamics data/1-1 perforation/dynamics_1_1_2nd5m.npy")

signal = data[..., 0]
control_action = data[..., 1]
targets = data[..., -1]

fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

ax[0].plot(signal, color='tab:blue', alpha=0.33, label="Reference controller")
ax[0].plot(targets, '--', color='tab:gray', label="Target position")
ax[0].plot(signal, color='tab:blue', label="Adaptive controller")
ax[0].legend(fontsize=8, loc='upper left')

ax[1].plot(control_action, '--', color='tab:gray', label="Target position")
ax[1].legend(fontsize=8, loc='upper left')

fig.tight_layout()
plt.show()