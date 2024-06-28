import pickle
import numpy as np
from matplotlib import pyplot as plt

data = np.load("Dynamics data/75-75 perforation/deployed_adapter_25.npy")
count = data[..., [0]].ravel()
beta = data[..., [4]].ravel()
omega = data[..., [5]].ravel()
control_action = data[..., [3]].ravel()
targets = data[..., [-2]].ravel()
true_targets = data[..., [-1]].ravel()

try:
    with open('./tmp/plot_counters.pkl', 'rb') as f:
        plot_counters = pickle.load(f)
    js_div_vals, js_div_counts, selection_counts, trespass_counts, update_counts = plot_counters
except FileNotFoundError:
    js_div_vals, js_div_counts, selection_counts, trespass_counts, update_counts = [[], []], [], [], [], []

fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

ax[0].plot(count, beta, marker='.', color='tab:blue', alpha=0.5, markersize=3, label="Angle")
ax[0].plot(count, targets, '--', color='tab:gray', alpha=0.33, label="Adapted target")
ax[0].plot(count, true_targets, '--', color='tab:gray', label="True target")
ax[0].legend(fontsize=8, loc='upper left')

ax[1].plot(count, omega, marker='.', color='tab:orange', alpha=0.5, markersize=3, label="Angular velocity")
ax[1].legend(fontsize=8, loc='upper left')

ax[2].plot(count, control_action, marker='.', color='tab:green', alpha=0.5, markersize=3, label="Control action")
ax[2].legend(fontsize=8, loc='upper left')

ax[3].plot(js_div_counts, js_div_vals, marker='.', markersize=3., lw=1., alpha=0.7,
           label=['Running distribution divergence vs. updated reference', 'Updated reference vs. Backed-up reference'])
ax[3].vlines(update_counts, 0., np.log(2), color='lightgrey', linestyle='-', label='Updates')
ax[3].vlines(selection_counts, 0., np.log(2), color='tab:green', linestyle='--', label='KB selections')
ax[3].vlines(trespass_counts, 0., np.log(2), color='tab:red', linestyle=':', label='Trespasses')
ax[3].axhline(np.log(2) / 2, color='black', linestyle='--', lw=1.)
ax[3].legend(fontsize=8, loc='upper left')

fig.tight_layout()
plt.show()
