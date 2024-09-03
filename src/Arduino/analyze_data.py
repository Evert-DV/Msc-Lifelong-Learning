import os
import pickle
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

plt.style.use('tableau-colorblind10')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# mpl.use("pgf")
# mpl.rcParams.update({
#     "pgf.texsystem": "xelatex",
#     'font.size': 8,
#     'text.usetex': True,
#     'pgf.rcfonts': False,
#     "pgf.preamble": r"\usepackage{amsmath}"
#                     r"\usepackage{lmodern}"
# })
tex_line_width = 3.48
tex_text_width = 7.17

root = os.getcwd() + '\\..\\..\\'
data_folders = [
    # "Dynamics data\\250-30 perforation\\PID 150-50\\EOL\\wo adapter",
    # "Dynamics data\\250-30 perforation\\PID 150-50\\EOL\\w adapter",
    # "Dynamics data\\200-50 perforation\\PID 150-50\\EOL\\wo adapter",  # auto_save_10 is filtered here
    # "Dynamics data\\200-50 perforation\\PID 150-50\\EOL\\w adapter",
    # "Dynamics data\\150-50 perforation\\PID 150-50\\EOL\\wo adapter",             # <--
    # "Dynamics data\\150-50 perforation\\PID 150-50\\EOL\\w adapter",              # <--
    # "Dynamics data\\150-50 perforation\\PID 150-50\\EOL extra\\wo adapter",       # <--
    # "Dynamics data\\150-50 perforation\\PID 150-50\\EOL extra\\w adapter",        # <--
    # "Dynamics data\\150-50 perforation\\PID 150-50\\EOL extra 2\\wo adapter",       # <--
    # "Dynamics data\\150-50 perforation\\PID 150-50\\EOL extra 2\\w adapter",        # <--
    # "Dynamics data\\150-50 perforation\\PID 150-50\\EOL extra 3\\wo adapter",
    # "Dynamics data\\150-50 perforation\\PID 150-50\\EOL extra 3\\w adapter",
    # "Dynamics data\\150-50 perforation\\PID 150-50\\EOL short window\\wo adapter",
    # "Dynamics data\\150-50 perforation\\PID 150-50\\EOL short window\\w adapter",
    # "Dynamics data\\150-50 perforation\\PID 150-50\\no adapter to adapter",       # <--
    # "Dynamics data\\150-50 perforation\\PID 150-50\\no adapter to adapter 2",
    "Dynamics data\\150-50 perforation\\PID 150-50\\3 step model\\wo adapter",      # <--
    # "Dynamics data\\150-50 perforation\\PID 150-50\\3 step model\\w adapter",
    "Dynamics data\\150-50 perforation\\PID 150-50\\3 step model\\w adapter 2",     # <--
    # "Dynamics data\\150-50 perforation\\PID 150-50\\KB\\wo adapter",  # <--
    # "Dynamics data\\150-50 perforation\\PID 150-50\\KB\\wo kb",       # <--
    # "Dynamics data\\150-50 perforation\\PID 150-50\\KB\\w kb",
    # "Dynamics data\\150-50 perforation\\PID 150-50\\crawling gait\\wo adapter",               # <--
    # "Dynamics data\\150-50 perforation\\PID 150-50\\crawling gait\\w adapter pretrained",     # <--
    # "Dynamics data\\150-50 perforation\\PID 150-50\\crawling gait\\w adapter online",         # <--
    # "Dynamics data\\150-50 perforation\\PID 75-75\\EOL wo adapter",
    # "Dynamics data\\150-50 perforation\\PID 75-75\\EOL w adapter",
    # "Dynamics data\\150-50 perforation\\PID soft\\wo adapter",    # <--
    # "Dynamics data\\150-50 perforation\\PID soft\\w adapter",     # <-- (auto_save_4 accidentally cut short)
    # "Dynamics data\\150-50 perforation\\PID hard\\EOL\\wo adapter",
    # "Dynamics data\\150-50 perforation\\PID hard\\EOL\\w adapter",
    # "Dynamics data\\150-50 perforation\\PID barely stable\\wo adapter",
    # "Dynamics data\\150-50 perforation\\PID barely stable\\w adapter",
    # "Dynamics data\\25-25 perforation\\PID 150-50\\wo adapter",       # <--
    # "Dynamics data\\25-25 perforation\\PID 150-50\\w adapter",        # <--
    # "Dynamics data\\50-50 perforation\\PID 150-50\\wo adapter",       # <--
    # "Dynamics data\\50-50 perforation\\PID 150-50\\w adapter",        # <--
]
file_name = "auto_save_0"
file = f"{data_folders[0]}/{file_name}.npy"

mean_rise_time, mean_settle_time, mean_overshoot, ae, mae, iae, cae, mav, ntv, mca = 10 * [None]
count, beta, omega, control_action, targets, true_targets = 6 * [None]

try:
    with open(f'{root}\\tmp\\plot_counters_adapter.pkl', 'rb') as f:
        plot_counters = pickle.load(f)
    js_div_vals, js_div_counts, selection_counts, trespass_counts, update_counts = plot_counters
    kb_plot = True
except FileNotFoundError:
    js_div_vals, js_div_counts, selection_counts, trespass_counts, update_counts = [[], []], [], [], [], []
    kb_plot = False


def get_file_data(file):
    data = np.load(file)
    count = data[..., [0]].ravel()
    beta = data[..., [1]].ravel()
    omega = data[..., [2]].ravel()
    control_action = data[..., [3]].ravel()
    targets = data[..., [-2]].ravel()
    true_targets = data[..., [-1]].ravel()

    return data, count, beta, omega, control_action, targets, true_targets


def metrics(file, do_print=True, do_plot=True):
    global mean_rise_time, mean_settle_time, mean_overshoot, ae, mae, iae, cae, mav, ntv
    global count, beta, omega, control_action, targets, true_targets

    data, count, beta, omega, control_action, targets, true_targets = get_file_data(file)

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
        settle_threshold = 0.05 * np.abs(window[0, -1] - window[0, 1])
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
    ae = np.abs(true_targets - beta)
    mae = np.trapz(ae, count) / (count[-1] - count[0])
    cae = np.convolve(ae, np.ones(15 * 50) / (15 * 50), mode='same')

    # Integral of the squared error
    iae = np.cumsum(ae[:-1] * dt)

    # Mean absolute velocity
    mav = np.trapz(np.abs(omega), count) / (count[-1] - count[0])

    # Normalized total variation of control action
    ntv = np.sum(np.abs(np.diff(control_action))) / (count[-1] - count[0])

    # mean magnitude of the control action
    mca = np.mean(np.abs(control_action))

    if do_print:
        # Define headers
        headers = ["Metric", "Value"]
        metrics = ["Mean Rise Time [s]", "Mean Settle Time [s]", "Mean Overshoot [deg]", "MAE [deg]", "IAE [deg]",
                   "MAV [deg/s]", "NTV [-]", "MCA [-]"]
        values = [mean_rise_time, mean_settle_time, mean_overshoot, mae, iae[-1], mav, ntv, mca]

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

    if do_plot:
        plot_file()

    return mean_rise_time, mean_settle_time, mean_overshoot, mae, iae[-1], mav, ntv, mca


def plot_file():
    global mean_rise_time, mean_settle_time, mean_overshoot, ae, mae, iae, cae, mav, ntv, mca

    signal_fig, signal_ax = plt.subplots(3, 1, figsize=(
        tex_line_width, 0.75 * tex_line_width) if mpl.get_backend() == 'pgf' else (12, 8), sharex=True)

    signal_ax[0].plot(count, targets, '--', color=colors[-1], label="Adapted target")
    signal_ax[0].plot(count, beta, color=colors[0], marker='.', markersize=3, label="System response")
    signal_ax[0].plot(count, true_targets, '--', color=colors[6], label="True target")
    signal_ax[0].legend(fontsize=8, loc='upper left')
    signal_ax[0].set_ylabel("Angle [deg]")

    signal_ax[1].plot(count, omega, color=colors[-2], marker='.', markersize=3)
    signal_ax[1].set_ylabel("Angular velocity [deg/s]")

    signal_ax[2].plot(count, control_action, color=colors[5], marker='.', markersize=3)
    signal_ax[2].set_xlabel("Count")
    signal_ax[2].set_ylabel("Control action [-]")

    signal_fig.tight_layout()
    plt.savefig(f'{root}\\tmp\\signal.{"pgf" if mpl.get_backend() == "pgf" else "png"}')

    ise_fig, ise_ax = plt.subplots(3, 1, figsize=(
        tex_line_width, 0.75 * tex_line_width) if mpl.get_backend() == 'pgf' else (12, 8), sharex=True)

    ise_ax[0].plot(count, beta, color=colors[0], marker='.', markersize=3, label="System response")
    ise_ax[0].plot(count, true_targets, '--', color=colors[6], label="True target")
    ise_ax[0].legend(fontsize=8, loc='upper left')
    ise_ax[0].set_ylabel("Angle [deg]")

    ise_ax[1].plot(count, ae, color=colors[5], label='Error')
    ise_ax[1].plot(count, cae, color=colors[1], label='Moving average')
    ise_ax[1].legend(fontsize=8, loc='upper left')
    ise_ax[1].set_ylabel("Absolute error [deg]")

    ise_ax[2].plot(count[1:], iae, color=colors[5])
    ise_ax[2].set_xlabel("Count")
    ise_ax[2].set_ylabel("Integral of absolute error [deg]")

    ise_fig.tight_layout()
    plt.savefig(f'{root}\\tmp\\ise.{"pgf" if mpl.get_backend() == "pgf" else "png"}')

    if kb_plot:
        kb_fig, kb_ax = plt.subplots(2, 1, figsize=(
            tex_line_width, 0.5 * tex_line_width) if mpl.get_backend() == 'pgf' else (12, 6), sharex=True)

        kb_ax[0].plot(count, beta, marker='.', markersize=3, label="System response")
        kb_ax[0].plot(count, true_targets, '--', color=colors[6], label="True target")
        kb_ax[0].legend(fontsize=8, loc='upper left')
        kb_ax[0].set_ylabel("Angle [deg]")

        js_lines = kb_ax[1].plot(js_div_counts, js_div_vals, marker='.', markersize=3., lw=1.)
        for t_update in update_counts:
            update_lines = kb_ax[1].axvline(t_update, linestyle='-', color=colors[2])
        for t_selection in selection_counts:
            selection_lines = kb_ax[1].axvline(t_selection, linestyle='--', color=colors[-1])
        for t_trespass in trespass_counts:
            trespass_lines = kb_ax[1].axvline(t_trespass, linestyle=':', color=colors[-4])
        kb_ax[1].axhline(np.log(2) / 2, alpha=.5, linestyle='--', lw=1., color=colors[-3])
        kb_ax[1].axhline(np.log(2) / 4, alpha=.5, linestyle='--', lw=1., color=colors[-2])
        kb_ax[1].legend(fontsize=8, loc='upper left',
                        handles=(*js_lines, update_lines, selection_lines, trespass_lines),
                        labels=['Running distribution vs. updated reference',
                                'Updated reference vs. Backed-up reference',
                                'Updates', 'KB selections', 'Trespasses'])
        kb_ax[1].set_xlabel("Count")
        kb_ax[1].set_ylabel("JS divergence [-]")

        kb_fig.tight_layout()
        plt.savefig(f'{root}\\tmp\\kb_{file_name}.{"pgf" if mpl.get_backend() == "pgf" else "png"}')

    # plt.show()


def make_xls(data_folder, save_folder=None, do_plot=False, file_type='auto_save'):
    files = [f for f in os.listdir(data_folder) if file_type in f and f.endswith(".npy")]
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    metrics_headers = ["Mean Rise Time [s]", "Mean Settle Time [s]", "Mean Overshoot [deg]", "MAE [deg]",
                       "IAE [deg]",
                       "MAV [deg/s]", "NTV [-]", "MCA [-]"]
    csv_data = {"Metric": metrics_headers}

    for file in files:
        file_path = os.path.join(data_folder, file)
        file_metrics = metrics(file_path, do_print=False, do_plot=False)
        csv_data[file] = file_metrics

    df = pd.DataFrame(csv_data)
    if save_folder is None:
        save_folder = data_folder

    excel_file = os.path.join(save_folder, f"metrics_summary.xlsx")
    df.to_excel(excel_file, index=False)

    if do_plot:
        plot_metrics_evolution(excel_file)

    return excel_file


def plot_metrics_evolution(excel_files, save_folder=f'{root}\\tmp'):
    dfs = [pd.read_excel(file) for file in excel_files]
    metrics_headers = dfs[0]["Metric"].tolist()
    file_names = [df.columns[1:] for df in dfs]

    x_ticks_list = []
    for file_name in file_names:
        start_t = 5 + 5 * int(file_name[0].split('_')[-1].strip('.npy'))
        x_ticks = range(start_t, start_t + 5 * len(file_name), 5)
        x_ticks_list.append(x_ticks)

    fig, axes = plt.subplots(len(metrics_headers) // 2, 2,
                             figsize=(tex_text_width, len(metrics_headers) * 0.33 * tex_line_width / 2),
                             sharex=True)
    axes = axes.flatten()

    for ax, header in zip(axes, metrics_headers):
        for df, x_ticks, color in zip(dfs, x_ticks_list, colors):
            ax.plot(x_ticks, df.loc[df["Metric"] == header].values[0][1:], lw=1., marker='.', markersize=3, color=color)
        # ax.set_title(header)
        ax.set_ylabel(header, fontsize=7)
        ax.set_xticks(x_ticks[::2])
        ax.tick_params(axis='both', labelsize=6)
    ax.set_xlabel('Time [min]', fontsize=7)

    # for ax in axes[len(metrics_headers):]:
    #     fig.delaxes(ax)

    # Find the longest common prefix
    common_prefix = os.path.commonpath(excel_files)
    short_labels = [
        file.replace(common_prefix, '').replace('\\', ' ').replace('metrics_summary.xlsx', '').replace('_', ' ').strip(
            '.npy') for file in excel_files]

    fig.legend(short_labels, loc='lower left', fontsize=6, frameon=False, bbox_to_anchor=(0., -0.01))

    fig.tight_layout()
    plot_file = os.path.join(save_folder, f'metrics_evolution.{"pgf" if mpl.get_backend() == "pgf" else "png"}')

    plt.savefig(plot_file)
    # plt.show()


def compare_metrics(data_folders, save_folder, file_type='auto_save'):
    excel_files = [make_xls(folder, file_type=file_type) for folder in data_folders]
    plot_metrics_evolution(excel_files, save_folder)


def metrics_similarity(data_folders, file_type='ref_minute'):
    labels = []
    all_metrics = []

    for folder in data_folders:
        files = [f for f in os.listdir(folder) if file_type in f and f.endswith(".npy")]
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        labels += [os.path.join(folder, file) for file in files]

        folder_metrics = np.array([[*metrics(os.path.join(folder, f), do_print=False, do_plot=False)] for f in files])
        all_metrics.append(folder_metrics)

    all_metrics = np.concatenate(all_metrics, axis=0)
    # all_metrics /= np.max(all_metrics, axis=0)
    all_metrics /= [5, 5, 25, 25, len(all_metrics) * 25, 8 * 60, 1800 / 1.9, 1800]

    # Find the longest common prefix
    common_prefix = os.path.commonpath(labels)
    short_labels = [label.replace(common_prefix, '').replace('\\', ' ').replace('_', ' ').strip('.npy') for label in
                    labels]

    distances = np.empty((len(all_metrics), len(all_metrics)))
    for i, metric_vec in enumerate(all_metrics):
        for j, metric_vec2 in enumerate(all_metrics):
            distances[i, j] = np.linalg.norm(metric_vec - metric_vec2, axis=-1)

    fig = plt.figure(figsize=(tex_line_width, 0.8 * tex_line_width) if mpl.get_backend() == 'pgf' else (10, 8))
    cax = plt.imshow(distances, vmin=0, cmap='viridis', aspect='auto')
    cbar = fig.colorbar(cax, label='metric distance')
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.yaxis.label.set_size(6)

    num_rows, num_cols = distances.shape
    cell_width = fig.get_size_inches()[0] / num_cols
    cell_height = fig.get_size_inches()[1] / num_rows
    font_size = min(min(cell_width, cell_height) * 15, 9)  # Adjust the multiplier as needed

    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            plt.text(j, i, f'{distances[i, j]:.2f}', ha='center', va='center', color='white', fontsize=font_size)

    plt.xticks(np.arange(distances.shape[1]), labels=short_labels, fontsize=6, rotation=45, ha='right',
               rotation_mode='anchor')
    plt.yticks(np.arange(distances.shape[0]), labels=short_labels, fontsize=6, ha='right')

    plt.tight_layout()

    plt.savefig(f'{root}\\tmp\\metrics_similarity.{"pgf" if mpl.get_backend() == "pgf" else "png"}')
    # plt.show()


def main():
    metrics(file, do_print=True, do_plot=True)
    # compare_metrics(data_folders, f'{root}\\tmp', file_type='auto_save')
    # metrics_similarity(data_folders, file_type='ref_minute')


if __name__ == '__main__':
    main()
