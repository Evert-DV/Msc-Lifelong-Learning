import pickle
import numpy as np
from matplotlib import pyplot as plt

adapter_window = 15
file = f"deployed_adapter_{adapter_window}"
# file = "auto_save_0"
data = np.load(f"Dynamics data/75-75 perforation/{file}.npy")
count = data[..., [0]].ravel()
beta = data[..., [1]].ravel()
omega = data[..., [2]].ravel()
control_action = data[..., [3]].ravel()
targets = data[..., [-2]].ravel()
true_targets = data[..., [-1]].ravel()

try:
    with open(f'./tmp/plot_counters_adapter_{adapter_window}.pkl', 'rb') as f:
        plot_counters = pickle.load(f)
    js_div_vals, js_div_counts, selection_counts, trespass_counts, update_counts = plot_counters
    kb_plot = True
except FileNotFoundError:
    js_div_vals, js_div_counts, selection_counts, trespass_counts, update_counts = [[], []], [], [], [], []
    kb_plot = False

dt = np.diff(count / 50)  # approximated dt
target_windows = np.argwhere(np.diff(true_targets) != 0).ravel() + 1
windowed_data = [data[i:j] for i, j in zip(target_windows[:-1], target_windows[1:])]
windowed_data.insert(0, data[:target_windows[0]])
windowed_data.append(data[target_windows[-1]:])

# Rise and settle times
rise_times = []
settle_times = []
overshoots = []
for window in windowed_data:
    # Rise time
    rise_threshold = window[0, -1] - 0.1 * (window[0, -1] - window[0, 1])
    rise_crossings = np.argwhere(np.diff(np.sign(window[:, 1] - rise_threshold)) != 0.).ravel()
    rise_count = rise_crossings[0] if rise_crossings.size > 0 else len(window) - 1
    rise_times.append((window[rise_count, 0] - window[0, 0]) / 50)

    # Settling time
    settle_threshold = 0.02 * np.abs(window[0, -1] - window[0, 1])
    settle_switch = np.diff((np.abs(window[:, -1] - window[:, 1]) <= settle_threshold).astype(int))
    settling_indices = np.argwhere(settle_switch == 1).ravel()
    last_settle_idx = settling_indices[-1] if settling_indices.size > 0 else len(window) - 1
    if np.sum(settle_switch[last_settle_idx:]) != 1.:
        last_settle_idx = len(window) - 1
    settle_times.append((window[last_settle_idx, 0] - window[0, 0]) / 50)

    # Overshoot
    overshoots.append(np.max((window[:, -1] - window[:, 1]) * np.sign(window[0, 1] - window[0, -1])))

# Mean rise and settle times, and overshoot
mean_rise_time = np.mean(rise_times)
mean_settle_time = np.mean(settle_times)
mean_overshoot = np.mean(overshoots)

# Root mean squared error
se = (true_targets - beta) ** 2
rmse = np.sqrt(np.trapz(se, count) / (count[-1] - count[0]))

# Integral of the squared error
ise = np.cumsum(se[:-1] * dt)

# Mean absolute control effort
mace = np.trapz(np.abs(control_action), count) / (count[-1] - count[0])

# Normalized total variation of control action
ntv = np.sum(np.abs(np.diff(control_action))) / (count[-1] - count[0])

# Define headers
headers = ["Metric", "Value"]
metrics = ["Mean Rise Time [s]", "Mean Settle Time [s]", "Mean Overshoot [deg]", "RMSE [deg]", "ISE [deg]", "MACE [-]",
           "NTV [-]"]
values = [mean_rise_time, mean_settle_time, mean_overshoot, rmse, ise[-1], mace, ntv]

# Print header
header_row = "| {:<20} | {:<20} |".format(headers[0], headers[1])
print(header_row)
print("-" * len(header_row))

# Print each metric
for metric, value in zip(metrics, values):
    if isinstance(value, float):
        print("| {:<20} | {:<20.4f} |".format(metric, value))
    else:
        print("| {:<20} | {:<20} |".format(metric, "Array"))


signal_fig, signal_ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

signal_ax[0].plot(count, beta, marker='.', color='tab:blue', alpha=0.5, markersize=3, label="Angle")
signal_ax[0].plot(count, targets, '--', color='tab:gray', alpha=0.33, label="Adapted target")
signal_ax[0].plot(count, true_targets, '--', color='tab:gray', label="True target")
signal_ax[0].legend(fontsize=8, loc='upper left')

signal_ax[1].plot(count, omega, marker='.', color='tab:orange', alpha=0.5, markersize=3, label="Angular velocity")
signal_ax[1].legend(fontsize=8, loc='upper left')

signal_ax[2].plot(count, control_action, marker='.', color='tab:green', alpha=0.5, markersize=3, label="Control action")
signal_ax[2].legend(fontsize=8, loc='upper left')

signal_fig.tight_layout()
plt.savefig(f'tmp/signal_{file}.png')


ise_fig, ise_ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

ise_ax[0].plot(count, beta, marker='.', color='tab:blue', alpha=0.5, markersize=3, label="Angle")
ise_ax[0].plot(count, true_targets, '--', color='tab:gray', label="True target")
ise_ax[0].legend(fontsize=8, loc='upper left')

ise_ax[1].plot(count, se, color='tab:red', alpha=0.5, label="Squared error")
ise_ax[1].legend(fontsize=8, loc='upper left')

ise_ax[2].plot(count[1:], ise, color='tab:red', alpha=0.75, label="Integral of the squared error")
ise_ax[2].legend(fontsize=8, loc='upper left')

ise_fig.tight_layout()
plt.savefig(f'tmp/ise_{file}.png')

if kb_plot:
    kb_fig, kb_ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    kb_ax[0].plot(count, beta, marker='.', color='tab:blue', alpha=0.5, markersize=3, label="Angle")
    kb_ax[0].plot(count, true_targets, '--', color='tab:gray', label="True target")
    kb_ax[0].legend(fontsize=8, loc='upper left')

    js_lines = kb_ax[1].plot(js_div_counts, js_div_vals, marker='.', markersize=3., lw=1., alpha=0.7)
    for t_update, t_selection, t_trespass in zip(update_counts, selection_counts, trespass_counts):
        update_lines = kb_ax[1].axvline(t_update, color='lightgrey', linestyle='-')
        selection_lines = kb_ax[1].axvline(t_selection, color='tab:green', linestyle='--')
        trespass_lines = kb_ax[1].axvline(t_trespass, color='tab:red', linestyle=':')
    kb_ax[1].axhline(np.log(2) / 2, color='tab:blue', alpha=.5, linestyle='--', lw=1.)
    kb_ax[1].axhline(np.log(2) / 4, color='tab:orange', alpha=.5, linestyle='--', lw=1.)
    kb_ax[1].legend(fontsize=8, loc='upper left',
                    handles=(*js_lines, update_lines, selection_lines, trespass_lines),
                    labels=['Running distribution vs. updated reference', 'Updated reference vs. Backed-up reference',
                            'Updates', 'KB selections', 'Trespasses'])

    kb_fig.tight_layout()
    plt.savefig(f'tmp/kb_{file}.png')


plt.show()


