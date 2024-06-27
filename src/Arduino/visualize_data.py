import numpy as np
from matplotlib import pyplot as plt


data = np.load("Dynamics data/75-75 perforation/user_save.npy")
count = data[..., [0]].ravel()
beta = data[..., [4]].ravel()
omega = data[..., [5]].ravel()
control_action = data[..., [3]].ravel()
targets = data[..., [-2]].ravel()
true_targets = data[..., [-1]].ravel()

fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

ax[0].plot(count, beta, marker='.', color='tab:blue', alpha=0.5, markersize=3, label="Angle")
ax[0].plot(count, targets, '--', color='tab:gray', alpha=0.33, label="Adapted target")
ax[0].plot(count, true_targets, '--', color='tab:gray', label="True target")
ax[0].legend(fontsize=8, loc='upper left')

ax[1].plot(count, omega, marker='.', color='tab:orange', alpha=0.5, markersize=3, label="Angular velocity")
ax[1].legend(fontsize=8, loc='upper left')

ax[2].plot(count, control_action, marker='.', color='tab:green', alpha=0.5, markersize=3, label="Control action")
ax[2].legend(fontsize=8, loc='upper left')

fig.tight_layout()
plt.show()