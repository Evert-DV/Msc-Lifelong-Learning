import numpy as np
from matplotlib import pyplot as plt


data = np.load("Dynamics data/75-75 perforation/b_auto_save_3.npy")
signal = data[..., [0, 3]].ravel()
control_action = data[..., [1, 1]].ravel()
targets = data[..., [-1, -1]].ravel()

fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

ax[0].plot(signal, color='tab:blue', alpha=0.33)
ax[0].plot(targets, '--', color='tab:gray', label="Target position")
ax[0].legend(fontsize=8, loc='upper left')

ax[1].plot(control_action, '--', color='tab:gray', label="Target position")
ax[1].legend(fontsize=8, loc='upper left')

fig.tight_layout()
plt.show()