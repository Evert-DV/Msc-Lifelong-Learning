import numpy as np
from matplotlib import pyplot as plt


data = np.load("Dynamics data/75-75 perforation/auto_save_2.npy")
signal = data[..., [0, 3]].ravel()
control_action = data[..., [1, 1]].ravel()
targets = data[..., [-2, -2]].ravel()
true_targets = data[..., [-1, -1]].ravel()

fig, ax = plt.subplots(1, 1, figsize=(8, 4), sharex=True)

ax.plot(signal, color='tab:blue', alpha=0.5)
ax.plot(true_targets, '--', color='tab:gray', alpha=0.33, label="True target")
ax.plot(targets, '--', color='tab:gray', label="Adapted target")
ax.legend(fontsize=8, loc='upper left')

fig.tight_layout()
plt.show()