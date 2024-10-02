import os
import pickle
import pandas as pd
import numpy as np
import matplotlib as mpl
import scienceplots
from botocore.compat import file_type
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox, TransformedBbox, BboxTransformTo
from adjustText import adjust_text, expand_axes_to_fit
from rich.cells import cell_len

plt.style.use('vibrant')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "xelatex",
    'font.size': 8,
    'text.usetex': True,
    'pgf.rcfonts': False,
    "pgf.preamble": r"\usepackage{amsmath}"
                    r"\usepackage{lmodern}"
                    r"\usepackage{siunitx}"
})
tex_line_width = 3.48
tex_text_width = 7.17

root = os.getcwd() + '\\..\\..\\'
data_folders = [
    # "Dynamics data\\250-30 perforation\\PID 150-50\\EOL\\wo adapter",
    # "Dynamics data\\250-30 perforation\\PID 150-50\\EOL\\w adapter",
    "Dynamics data\\200-50 perforation\\PID 150-50\\EOL\\wo adapter",  # <-- varying perf, auto_save_10 is filtered here
    "Dynamics data\\200-50 perforation\\PID 150-50\\EOL\\w adapter",  # <-- varying perf, + qualitative late stage
    # "Dynamics data\\150-50 perforation\\PID 150-50\\EOL\\wo adapter",
    # "Dynamics data\\150-50 perforation\\PID 150-50\\EOL\\w adapter",
    # "Dynamics data\\150-50 perforation\\PID 150-50\\EOL extra\\wo adapter",
    # "Dynamics data\\150-50 perforation\\PID 150-50\\EOL extra\\w adapter",
    # "Dynamics data\\150-50 perforation\\PID 150-50\\EOL extra 2\\wo adapter",
    # "Dynamics data\\150-50 perforation\\PID 150-50\\EOL extra 2\\w adapter",
    "Dynamics data\\150-50 perforation\\PID 150-50\\EOL extra 3\\wo adapter",       # <-- limitations
    "Dynamics data\\150-50 perforation\\PID 150-50\\EOL extra 3\\w adapter",        # <-- limitations
    # "Dynamics data\\150-50 perforation\\PID 150-50\\EOL short window\\wo adapter",
    # "Dynamics data\\150-50 perforation\\PID 150-50\\EOL short window\\w adapter",
    # "Dynamics data\\150-50 perforation\\PID 150-50\\no adapter to adapter",       # <-- appendix?
    # "Dynamics data\\150-50 perforation\\PID 150-50\\no adapter to adapter 2",
    # "Dynamics data\\150-50 perforation\\PID 150-50\\3 step model\\wo adapter",
    # "Dynamics data\\150-50 perforation\\PID 150-50\\3 step model\\w adapter",
    # "Dynamics data\\150-50 perforation\\PID 150-50\\3 step model\\w adapter 2",
    # "Dynamics data\\150-50 perforation\\PID 150-50\\KB\\wo adapter",  # <-- as 12 is copy of as 13
    # "Dynamics data\\150-50 perforation\\PID 150-50\\KB\\w adapter",       # <--
    # "Dynamics data\\150-50 perforation\\PID 150-50\\KB\\w kb",
    # "Dynamics data\\150-50 perforation\\PID 150-50\\crawling gait\\wo adapter",  # <-- crawling gait
    # "Dynamics data\\150-50 perforation\\PID 150-50\\crawling gait\\w adapter pretrained", # <-- crawling gait, + qualitative
    # "Dynamics data\\150-50 perforation\\PID 150-50\\crawling gait\\w adapter online",  # <-- crawling gait
    # "Dynamics data\\150-50 perforation\\PID 75-75\\EOL wo adapter",
    # "Dynamics data\\150-50 perforation\\PID 75-75\\EOL w adapter",
    # "Dynamics data\\150-50 perforation\\PID soft\\wo adapter",  # <-- varying PID
    # "Dynamics data\\150-50 perforation\\PID soft\\w adapter",   # <-- varying PID, + qualitative, (auto_save_4 copy of as 5)
    # "Dynamics data\\150-50 perforation\\PID slow overshoot\\wo adapter",  # <-- varying PID
    # "Dynamics data\\150-50 perforation\\PID slow overshoot\\w adapter",  # <-- varying PID, qualitative, self-correcting
    # "Dynamics data\\150-50 perforation\\PID slow overshoot 2\\wo adapter",    # <-- varying PID
    # "Dynamics data\\150-50 perforation\\PID slow overshoot 2\\w adapter",  # <-- varying PID
    # "Dynamics data\\150-50 perforation\\PID hard\\EOL\\wo adapter",       # <-- limitations ?
    # "Dynamics data\\150-50 perforation\\PID hard\\EOL\\w adapter",        # <-- limitations ?
    # "Dynamics data\\150-50 perforation\\PID barely stable\\wo adapter",
    # "Dynamics data\\150-50 perforation\\PID barely stable\\w adapter",
    # "Dynamics data\\25-25 perforation\\PID 150-50\\wo adapter",  # <-- varying perf
    # "Dynamics data\\25-25 perforation\\PID 150-50\\w adapter",  # <-- varying perf
    # "Dynamics data\\50-50 perforation\\PID 150-50\\wo adapter",  # <-- varying perf
    # "Dynamics data\\50-50 perforation\\PID 150-50\\w adapter",  # <-- varying perf, + qualitative
    # "Dynamics data\\150-50 perforation\\",
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
        plot_file([file])

    return mean_rise_time, mean_settle_time, mean_overshoot, mae, mav, ntv  # iae[-1], , mca


def plot_file(files):
    global mean_rise_time, mean_settle_time, mean_overshoot, ae, mae, iae, cae, mav, ntv, mca

    signal_fig, signal_ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(
        tex_line_width, 0.57 * tex_line_width) if mpl.get_backend() == 'pgf' else (12, 8), sharex=True)

    target_styles = [(0, (5, 2)), '-']
    signal_styles = [(0, (2, 2)), '-']
    control_styles = [(0, (3, 3)), '-']

    target_colors = [colors[0], colors[-3]]
    target_alpha = [.8, 1.]
    signal_alpha = [.8, 1.]
    control_colors = [colors[-2], colors[2]]

    t_lines = []
    x_lines = []
    u_lines = []

    offsets = [.000, .000]

    for i, file in enumerate(files):
        data, count, beta, omega, control_action, targets, true_targets = get_file_data(file)
        count -= count[0]
        count /= count[-1]
        count -= offsets[i]

        target_line, = signal_ax[0].plot(count, targets, linestyle=target_styles[i], color=target_colors[i],
                                         alpha=target_alpha[i], linewidth=.7,
                                         )
        # truetarget_lines, = signal_ax[0].plot(count, true_targets, linestyle=(0, (6, 1)), color=colors[3], alpha=0.5,
        #                                       linewidth=1, label="True target")
        signal_line, = signal_ax[0].plot(count, beta, linestyle=signal_styles[i], color=colors[1],
                                         alpha=signal_alpha[i], linewidth=.7,  # marker='.', markersize=1,
                                         )
        control_line, = signal_ax[1].plot(count, control_action, linestyle=control_styles[i], color=control_colors[i],
                                          linewidth=.7)

        t_lines.append(target_line)
        x_lines.append(signal_line)
        u_lines.append(control_line)

    signal_ax[0].set_ylabel("$\\theta$ [deg]")
    # signal_ax[1].set_xticks(plot_count[::1000])
    # signal_ax[1].plot(count, omega, color=colors[-2], marker='.', markersize=2, lw=.5)
    # signal_ax[1].set_ylabel("Angular velocity [deg/s]")

    signal_ax[1].set_xlabel("Normalized count")
    signal_ax[1].set_ylabel("$u$ [\si{\micro\second}]")
    for ax in signal_ax:
        ax.tick_params(axis='both', labelsize=6)

    # signal_ax[0].set_xlim([0.286, 0.386])  # PID soft
    # signal_ax[0].set_xlim([.82, .92])  # 200-50
    # signal_ax[0].set_ylim([-25, -7.5])  # 200-50
    # signal_ax[0].set_xlim([.145, .245])  # slow overshoot
    # signal_ax[0].set_ylim([-27, -1])  # slow overshoot
    # signal_ax[0].set_xlim([.53, .63])   # crawling gait
    # signal_ax[0].set_ylim([-21, -5.5])  # crawling gait
    # signal_ax[0].set_xlim([.183, .215])   # 50--50
    # signal_ax[0].set_ylim([-8.5, -3.7])   # 50--50
    # signal_ax[0].set_xlim([.096, .128])  # slow over (2)
    # signal_ax[0].set_ylim([-27, -7.5])  # slow over (2)
    # signal_ax[0].set_xlim([0.64, 0.685])  # limitation
    # signal_ax[0].set_ylim([-22, -13.5])  # limitation
    signal_ax[0].set_xlim([0, 0.198])  # pre vs online
    signal_ax[0].set_ylim([-20, -4])  # pre vs online
    signal_fig.legend(handles=[*t_lines, *x_lines, *u_lines],
                      labels=["PID target", "Adapter target", "PID signal", "Adapter signal", "PID control",
                              "Adapter control"],
                      frameon=False, loc='lower left', ncol=3, handlelength=1., bbox_to_anchor=(0, -0.15))

    signal_fig.tight_layout()
    if mpl.get_backend() == 'pgf':
        plt.savefig(f'{root}\\reports\\Thesis\\figures\\discussion\\pre-vs-online-pre.pgf', bbox_inches='tight',
                    dpi=300)

    # ise_fig, ise_ax = plt.subplots(3, 1, figsize=(
    #     tex_line_width, 0.75 * tex_line_width) if mpl.get_backend() == 'pgf' else (12, 8), sharex=True)
    #
    # ise_ax[0].plot(count, beta, color=colors[0], marker='.', markersize=3, label="System response")
    # ise_ax[0].plot(count, true_targets, '--', color=colors[6], label="True target")
    # ise_ax[0].legend(fontsize=8, loc='upper left')
    # ise_ax[0].set_ylabel("Angle [deg]")
    #
    # ise_ax[1].plot(count, ae, color=colors[5], label='Error')
    # ise_ax[1].plot(count, cae, color=colors[1], label='Moving average')
    # ise_ax[1].legend(fontsize=8, loc='upper left')
    # ise_ax[1].set_ylabel("Absolute error [deg]")
    #
    # ise_ax[2].plot(count[1:], iae, color=colors[5])
    # ise_ax[2].set_xlabel("Count")
    # ise_ax[2].set_ylabel("Integral of absolute error [deg]")
    #
    # ise_fig.tight_layout()
    # plt.savefig(f'{root}\\tmp\\ise.{"pgf" if mpl.get_backend() == "pgf" else "png"}')

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


def generate_latex_table(data, columns, file_path):
    import math

    # Modify the column headers
    columns = [str((i - 1) * 5 + 5) if col.startswith('auto_save_') else col for i, col in enumerate(columns)]

    max_cols_per_table = 12  # Maximum number of data columns per table (excluding first column)
    num_data_cols = len(columns) - 1  # Exclude the first column (labels)
    number_of_tables = math.ceil(num_data_cols / max_cols_per_table)

    with open(file_path, 'w') as f:
        for table_idx in range(number_of_tables):
            start_idx = 1 + table_idx * max_cols_per_table
            end_idx = min(start_idx + max_cols_per_table, len(columns))

            table_columns = [columns[0]] + columns[start_idx:end_idx]

            # Write the LaTeX table code
            f.write(
                "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l|" + "r" * (len(table_columns) - 1) + "}\n")
            f.write("    \\toprule\n")
            f.write("    " + " & ".join([f"\\textbf{{{col}}}" for col in table_columns]) + " \\\\\n")
            f.write("    \\midrule\n")
            for row in data:
                row_data = [row[0]] + row[start_idx:end_idx]
                formatted_row = ["{:.2f}".format(value) if isinstance(value, (int, float)) else str(value) for value in
                                 row_data]
                formatted_row[0] = f"\\textbf{{{formatted_row[0]}}}"  # Make the first column bold
                f.write("    " + " & ".join(formatted_row) + " \\\\\n")
            f.write("    \\bottomrule\n")
            f.write("\\end{tabular*}\n\n")  # Add a newline between tables


def make_xls(data_folder, save_folder=None, do_plot=False, file_type='auto_save', do_latex=False):
    files = [f for f in os.listdir(data_folder) if file_type in f and f.endswith(".npy")]
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    metrics_headers = ["MRT [s]", "MST [s]", "MO  [deg]", "MAE [deg]",
                       # "IAE [deg]",
                       "MAV [deg/s]", "NTV [\si{\micro\second}]",
                       #                        "MCA [\si{\micro\second}]"
                       ]
    csv_data = {"Metric": metrics_headers}

    avg_metrics = []
    for file in files:
        file_path = os.path.join(data_folder, file)
        file_metrics = metrics(file_path, do_print=False, do_plot=False)
        csv_data[file] = file_metrics
        avg_metrics.append(file_metrics)

    avg_metrics = np.mean(avg_metrics, axis=0)

    df = pd.DataFrame(csv_data)
    if save_folder is None:
        save_folder = data_folder

    excel_file = os.path.join(save_folder, f"metrics_summary.xlsx")
    df.to_excel(excel_file, index=False)

    if do_latex:
        columns = df.columns.tolist()
        data = df.values.tolist()
        latex_file_path = os.path.join(f"{root}\\reports\\Thesis\\appendix tables\\",
                                       f'ref crawling gait {os.path.basename(data_folder)}.tex')
        generate_latex_table(data, columns, latex_file_path)

    if do_plot:
        plot_metrics_evolution(excel_file)

    # Print header
    print(f"\nMetrics for {os.path.basename(data_folder)}")
    headers = ["Metric", "Value"]
    header_row = "| {:<20} | {:<20} |".format(headers[0], headers[1])
    print(header_row)
    print("-" * len(header_row))

    # Print each metric
    for metric, value in zip(metrics_headers, avg_metrics):
        if isinstance(value, float):
            print("| {:<20} | {:<20.4f} |".format(metric, value))
        else:
            print("| {:<20} | {:<20} |".format(metric, "Array"))

    return excel_file


def plot_metrics_evolution(excel_files, save_folder=f'{root}\\tmp', file_type='auto_save'):
    dfs = [pd.read_excel(file) for file in excel_files]
    metrics_headers = [dfs[0]["Metric"][i] for i in [0, 1, 2, 3, 4, 5]]  # [MRT, MST, MO, MAE, MAV, NTV]
    file_names = [df.columns[1:] for df in dfs]

    # Find the longest common prefix
    common_prefix = os.path.commonpath(excel_files)
    short_labels = [
        file.replace(common_prefix, '').replace('\\', ' ').replace('metrics_summary.xlsx', '').replace('_', ' ').strip(
            '.npy') for file in excel_files]

    x_ticks_list = []
    combined_ticks = set()
    for file_name in file_names:
        if file_type == 'auto_save':
            x_ticks = [5 + 5 * int(file.split('_')[-1].split('.')[0]) for file in file_name]
        else:
            x_ticks = [int(file.split('_')[-1].strip('.npy')) for file in file_name]
        x_ticks_list.append(x_ticks)
        combined_ticks.update(x_ticks)

    n_cols = 2
    fig, axes = plt.subplots(len(metrics_headers) // n_cols, n_cols,
                             figsize=(tex_text_width, len(metrics_headers) * 0.28 * tex_line_width / n_cols),
                             sharex=True)

    axes = axes.flatten()
    for ax, header in zip(axes, metrics_headers):
        texts = []
        for df, x_ticks, label in zip(dfs, x_ticks_list, short_labels):
            y_values = df.loc[df["Metric"] == header].values[0][1:]
            line, = ax.plot(x_ticks, y_values, marker='.', markersize=1, lw=.5, alpha=0.5,
                            color=colors[1] if 'w adapter' in label else colors[-3],
                            linestyle='-' if 'w adapter' in label else '--')
            # Choose a point to annotate
            x_last = x_ticks[-1]
            y_last = y_values[-1]

            # Create the text annotation
            txt = ax.text(
                x_last,
                y_last,
                '-'.join([str(float(f) / 100) for f in label.strip().split(' ')[0].split('-')]),
                # label.strip().split(' ')[-1],
                fontsize=6,
                color=line.get_color(),
                alpha=1.,
                ha='left',
                va='bottom'
            )
            texts.append(txt)  # Add to the list of texts

        # Adjust the text positions to prevent overlapping
        adjust_text(texts, ax=ax, expand_axes=True, avoid_self=False)
        # texts[0].set_visible(False)  # Hide the first text annotation

        # Determine the maximum length of the data
        max_length_wo_adapter = max([df.shape[1] for df, label in zip(dfs, short_labels) if 'wo adapter' in label])
        max_length_w_adapter = max([df.shape[1] for df, label in zip(dfs, short_labels) if 'w adapter' in label])

        # Pad the shorter data arrays with NaN values
        padded_data_wo_adapter = [
            np.pad(df.loc[df["Metric"] == header].values[0][1:], (0, max_length_wo_adapter - df.shape[1]),
                   constant_values=np.nan)
            for df, label in zip(dfs, short_labels) if 'wo adapter' in label
        ]
        padded_data_w_adapter = [
            np.pad(df.loc[df["Metric"] == header].values[0][1:], (0, max_length_w_adapter - df.shape[1]),
                   constant_values=np.nan)
            for df, label in zip(dfs, short_labels) if 'w adapter' in label
        ]

        # Calculate the averages, ignoring NaN values
        avg_values_wo_adapter = np.nanmean(padded_data_wo_adapter, axis=0)
        avg_values_w_adapter = np.nanmean(padded_data_w_adapter, axis=0)

        # Plot the averages
        wo_adapter_lines, = ax.plot(sorted(list(combined_ticks))[:max_length_wo_adapter - 1], avg_values_wo_adapter,
                                    linestyle='--', lw=.7, marker='.', markersize=2, color=colors[-3])
        w_adapter_lines, = ax.plot(sorted(list(combined_ticks))[:max_length_w_adapter - 1], avg_values_w_adapter, lw=.7,
                                   marker='.', markersize=2, color=colors[1])

        skip_tick = max(1, len(combined_ticks) // 12)
        ax.set_ylabel(header)
        ax.set_xticks(sorted(list(combined_ticks))[::-skip_tick])
        ax.tick_params(axis='both', labelsize=6)

    for ax in axes[len(metrics_headers):]:
        fig.delaxes(ax)

    # Set x-label for the last subplot in each column
    if n_cols == 2:
        axes[len(metrics_headers) - 2].set_xlabel('Time [min]')
    axes[len(metrics_headers) - 1].set_xlabel('Time [min]')

    fig.legend(handles=[wo_adapter_lines, w_adapter_lines], labels=["w/o adapter", "w/ adapter"], loc='lower left',
               frameon=False, bbox_to_anchor=(0., -0.05), handlelength=1.25)

    fig.tight_layout()
    plot_file = os.path.join(save_folder, f'ref-25--25-50--50-metrics.{"pgf" if mpl.get_backend() == "pgf" else "png"}')

    if mpl.get_backend() == 'pgf':
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    # plt.show()


def compare_metrics(data_folders, save_folder, file_type='auto_save'):
    excel_files = [make_xls(folder, file_type=file_type) for folder in data_folders]
    plot_metrics_evolution(excel_files, save_folder, file_type)


def metrics_similarity(data_folders, file_type='ref_minute'):
    labels = []
    all_metrics = []

    for folder in data_folders:
        files = [f for f in os.listdir(folder) if file_type in f and f.endswith("_0.npy")]
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        labels += [os.path.join(folder, file) for file in files]

        folder_metrics = np.array([[*metrics(os.path.join(folder, f), do_print=False, do_plot=False)] for f in files])
        all_metrics.append(folder_metrics)

    all_metrics = np.concatenate(all_metrics, axis=0)
    # all_metrics /= np.max(all_metrics, axis=0)
    all_metrics /= [5, 5, 25, 25, 8 * 60, 1800 / 1.9]
    vmax = np.linalg.norm(np.ones(all_metrics.shape[-1]))

    # Find the longest common prefix
    common_prefix = os.path.commonpath(labels)
    short_labels = [label.replace(common_prefix, '').split(os.sep) for label in
                    labels]
    # short_labels = [l[1] for l in short_labels]
    short_labels = [' '.join(['-'.join([str(float(f) / 100) for f in l[1].strip().split(' ')[0].split('-')]), l[-2]]) for l in short_labels]

    distances = np.empty((len(all_metrics), len(all_metrics)))
    for i, metric_vec in enumerate(all_metrics):
        for j, metric_vec2 in enumerate(all_metrics):
            distances[i, j] = np.linalg.norm(metric_vec - metric_vec2, axis=-1)

    n_cells = len(short_labels)
    fig = plt.figure(
    # figsize=(tex_line_width, 0.8 * tex_line_width)
    figsize=(min(tex_line_width, 2.2/2.56 + (1.2 + n_cells) * 1.2 / 2.56), 1.6/2.56 + n_cells * 1.2 / 2.56)
    )

    cax = plt.imshow(distances, vmin=0, vmax=vmax, cmap='viridis', aspect='equal')
    cbar = fig.colorbar(cax, label='metric distance')
    cbar.ax.tick_params(labelsize=6)
    # cbar.ax.yaxis.label.set_size(6)

    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            plt.text(j, i, f'{distances[i, j]:.2f}', ha='center', va='center', color='white')

    plt.xticks(np.arange(distances.shape[1]), labels=short_labels, fontsize=6, rotation=45, ha='right',
               rotation_mode='anchor')
    plt.yticks(np.arange(distances.shape[0]), labels=short_labels, fontsize=6, ha='right')

    plt.tight_layout()

    if mpl.get_backend() == 'pgf':
        plt.savefig(f'{root}\\reports\\Thesis\\figures\\appendix\\ref_dist_150-200.pgf', bbox_inches='tight', dpi=300)
    # plt.show()


def main():
    # plot_file([f"{folder}/{file_name}.npy" for folder in data_folders])
    # metrics(file, do_print=True, do_plot=True)
    # compare_metrics(data_folders, f'{root}\\reports\\Thesis\\figures\\appendix', file_type='ref_minute')
    # compare_metrics(data_folders, f'{root}\\tmp', file_type='ref_minute')
    metrics_similarity(data_folders, file_type='ref_minute')


if __name__ == '__main__':
    main()
