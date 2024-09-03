import os

os.environ["KERAS_BACKEND"] = "torch"
from src.toy_problem.kb_tools import *
from src.toy_problem.toy_tools import *
import pickle
import keras
import matplotlib as mpl
from keras import optimizers, losses
from torch.utils.data import TensorDataset, DataLoader


def main():
    # Load 'vanilla' adapter model (or included in KB)
    prediction_window = 10
    adapter = TargetAdapter(state_size=2, target_size=2)
    # adapter.load_weights('../../tmp/target_adapter.weights.h5')
    optimizer = keras.optimizers.Adam(learning_rate=1.e-3)
    loss_fn = keras.losses.MeanAbsoluteError()
    adapter.compile(optimizer=optimizer, loss=loss_fn)

    # Load VAE model
    autoencoder = VariationalAutoEncoder(5, 2)
    autoencoder.load_weights('../../tmp/vae_skip_s.weights.h5')
    optimizer = optimizers.Adam(learning_rate=5.e-3)
    loss_fn = losses.MeanSquaredError()
    autoencoder.compile(optimizer=optimizer, loss=loss_fn)

    # Define system
    # system = System(5, 20, 87, 5)
    system = System(5, 20, 10, 5)
    # Define controller
    controller = PIDController(300, 10, 50)
    # controller = PIDController(300, 40, 5)

    # Define reference controller
    reference_controller = PIDController(300, 10, 50)
    # reference_controller = PIDController(300, 40, 5)
    # Setup KB
    use_kb = False
    load_kb = False
    if not load_kb:
        reference_data = np.load(f"../../tmp/sim data/m5k20c0.5_seed329.npy")
        x_reference = ops.array(reference_data)[..., [0, 1, 2, 3, 4]].reshape(-1, 5)
        z_mean, z_log_var = autoencoder.dynamics(x_reference)
        cov = sample(z_mean, z_log_var)[1]
        reference = MultivariateNormal(z_mean, cov)
        kb = [[reference], [adapter.get_weights()]]
    else:
        with open('tmp/kb.pkl', 'rb') as f:
            kb = pickle.load(f)

    initial_kb_len = len(kb[0])
    print(f"{initial_kb_len} entries in the KB\n")
    thres = np.log(2) / 4

    # Setup simulation loop
    dt = 1 / 50
    t_end = 60
    t = np.arange(0, t_end, dt)
    target = np.array([11., 0.])
    buffer = []
    true_target_list = []
    x0 = [9.9, 0]  # 9.9 was found to be the steady state
    x0_reference = [9.9, 0]
    train_interval = 15
    update_interval = 60
    kb_step = 5
    t_change = np.random.randint(train_interval, t_end - train_interval)

    # Setup plotting lists
    signal = []
    reference_signal = []
    targets = []
    adapted_targets = []
    reference_controls = []
    adapted_controls = []
    kl_loss = []
    js_selection = []
    t_train = []
    t_updates = []
    t_trespassing = []
    t_kb_selection = []
    t_reference_selection = []

    # Simulation loop
    for ti in t:
        print(f"\rt =  {ti:.0f}", end="")
        if ti > 30:
            buffer.pop(0)
            true_target_list.pop(0)

        # Target selection
        if ti % 5 == 0:
            target = [np.random.uniform(5, 15), 0.]
        targets.append(target)
        true_target_list.append(target)

        # # Hard change for testing
        # if ti == t_change:
        #     print("\n\nHARD CHANGE\n"
        #           f"k: {system.k:.1f}\t -> {system.k + 10:.1f}\n"
        #           f"c: {system.c:.1f}\t -> {system.c + 10:.1f}\n"
        #           f"l0: {system.l0:.1f}\t -> {system.l0 / 3:.1f}")
        #     system.k += 10
        #     system.c += 3
        #     system.l0 /= 3

        # Control loop
        noise = np.random.normal([0, 0], [0.0, 0], 2)
        noise2 = np.random.normal([0, 0], [0.0, 0], 2)
        x0 += noise
        delta_target = adapter.predict(ops.array([*x0, *target])[None], verbose=0)[0]
        predicted_target = target + delta_target
        adapted_targets.append(predicted_target)
        control_action = controller.compute_control(x0, predicted_target, dt)
        adapted_controls.append(control_action)
        x = system.response(x0, control_action, do_update=True) + noise2
        signal.append(x)
        x0 = x

        # Reference control loop
        x0_reference += noise
        reference_control = reference_controller.compute_control(x0_reference, target, dt)
        reference_controls.append(reference_control)
        x_reference = system.response(x0_reference, reference_control, do_update=True) + noise2
        reference_signal.append(x_reference)
        x0_reference = x_reference

        # Buffer data
        buffer.append([*x0, control_action, *x, *predicted_target])

        # Adapter update step
        if ti % train_interval == 0 and ti != 0:
            t_train.append(ti)
            adapter.optimizer.lr = 1.e-3
            buffer_array = ops.array(buffer)
            features, labels = prep_data(buffer_array, prediction_window, state_size=2, target_size=2, true_target_list=true_target_list)

            print("\nFitting model...")
            train_dataset, val_dataset = random_split(TensorDataset(features, labels),
                                                      [int(0.8 * len(features)),
                                                       len(features) - int(0.8 * len(features))])
            train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)
            callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       mode='min',
                                                       min_delta=1e-4,
                                                       patience=5,
                                                       restore_best_weights=True,
                                                       verbose=1),
                         keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                           factor=0.1,
                                                           patience=7,
                                                           min_lr=5e-5,
                                                           min_delta=1e-3,
                                                           verbose=0),
                         EpochLogger()
                         ]
            adapter.fit(train_dataloader,
                        epochs=1000,
                        callbacks=callbacks,
                        validation_data=val_dataloader,
                        verbose=0,
                        )
            # buffer = buffer.tolist()

        # KB step
        if not use_kb:
            continue

        if ti % kb_step == 0:
            # Get embeddings
            z_mean, z_log_var = autoencoder.dynamics(ops.array(buffer)[..., [0, 1, 2, 3, 4]])
            embeddings, cov = sample(z_mean, z_log_var, samples_per_centroid=1)

            # Pre-60 seconds
            if ti < update_interval:
                t_reference_selection.append(ti)
                running_distribution = MultivariateNormal(z_mean, cov)
                current_kb_idx, js_divs = search_kb(running_distribution, kb[0])
                js_selection.append(js_divs)
                continue

            if ti == update_interval:
                # Select KB entry
                reference = kb[0][current_kb_idx]
                updated_reference = reference.copy()
                backup_updated_reference = reference.copy()
                adapter.set_weights(kb[1][current_kb_idx])
                print(f"\nSelected {current_kb_idx} as reference")

            # Post-60 seconds
            # Update running distribution
            running_distribution.update(z_mean, cov, weight=kb_step/(update_interval-kb_step))

            # KL losses
            kl_updated_dist = js_divergence(updated_reference, running_distribution)
            kl_reference_updated = js_divergence(reference, updated_reference)
            kl_loss.append([kl_updated_dist.item(), kl_reference_updated.item()])

            # Check for shift
            if kl_updated_dist > thres or kl_reference_updated > .75 * thres:
                print("\nSHIFT DETECTED\nRestore KB entry reference")
                t_trespassing.append(ti)
                # retain last or leave it as it was?
                # kb[0][kb_idx] = backup_updated_reference.copy()
                # kb[1][kb_idx] = adapter.get_weights()
                print("Check KB for better match")
                best_idx, js_divs = search_kb(running_distribution, kb[0])
                if best_idx != current_kb_idx and js_divs[best_idx] < thres:
                    print(f"Use KB entry {best_idx} as reference")
                    t_kb_selection.append(ti)
                    reference = kb[0][best_idx]
                    updated_reference = reference.copy()
                    backup_updated_reference = reference.copy()
                    current_kb_idx = best_idx
                    adapter.set_weights(kb[1][best_idx])
                else:
                    print("No match found\nInitiate new KB entry")
                    reference = running_distribution.copy()
                    updated_reference = reference.copy()
                    kb[0].append(reference)
                    kb[1].append(adapter.get_weights())
                    current_kb_idx = len(kb[0]) - 1

        # KB update step
        if ti % update_interval == 0 and ti != 0:
            print('\nUPDATE STEP')
            torch.cuda.empty_cache()
            t_updates.append(ti)

            print("Check KB for better match")
            best_idx, js_divs = search_kb(running_distribution, kb[0])
            if best_idx != current_kb_idx and js_divs[best_idx] < thres:
                print(f"Use KB entry {best_idx} as reference")
                t_kb_selection.append(ti)
                kb[0][current_kb_idx] = backup_updated_reference
                kb[1][current_kb_idx] = adapter.get_weights()
                reference = kb[0][best_idx]
                updated_reference = reference.copy()
                backup_updated_reference = reference.copy()
                current_kb_idx = best_idx
                adapter.set_weights(kb[1][best_idx])
                continue

            # Update reference
            print("No match found")
            backup_updated_reference = updated_reference.copy()
            updated_reference.update(z_mean, cov, weight=0.1)
            print(f"Updated reference")

            # Empty buffer
            # buffer = []

    # Save the last updated reference
    if use_kb:
        kb[0][current_kb_idx] = backup_updated_reference
        kb[1][current_kb_idx] = adapter.get_weights()
        print(f"\n\n{len(kb[0])} entries in the KB")
        try:
            with open('tmp/kb.pkl', 'wb') as f:
                pickle.dump(kb, f)
        except FileNotFoundError:
            os.makedirs('tmp')
            with open('tmp/kb.pkl', 'wb') as f:
                pickle.dump(kb, f)
        print("KB saved")

    # Plotting
    signal = np.asarray(signal)
    reference_signal = np.asarray(reference_signal)
    targets = np.asarray(targets)
    adapted_targets = np.asarray(adapted_targets)
    reference_controls = np.asarray(reference_controls)
    adapted_controls = np.asarray(adapted_controls)

    plt.style.use('tableau-colorblind10')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.texsystem": "xelatex",
        'font.size': 8,
        'text.usetex': True,
        'pgf.rcfonts': False,
        "pgf.preamble": r"\usepackage{amsmath}"
                        r"\usepackage{lmodern}"
    })
    tex_line_width = 3.48
    tex_text_width = 7.17

    fig, ax = plt.subplots(3 if use_kb else 2, 1, gridspec_kw={'height_ratios': [2, 1]},
                           figsize=(tex_line_width, 0.67*tex_line_width) if mpl.get_backend() == 'pgf' else (16, 8), sharex=True)

    if use_kb:
        for ax_i in ax:
            for t_trn in t_train:
                ax_i.axvline(t_trn, color='tab:gray', linestyle=':', alpha=0.33)
            for t_update in t_updates:
                ax_i.axvline(t_update, color='lightgrey', linestyle='-')
            for t_kb in t_kb_selection:
                ax_i.axvline(t_kb, color='tab:green', linestyle='--')
            for t_tres in t_trespassing:
                ax_i.axvline(t_tres, color='tab:red', linestyle=':')
            ax_i.axvline(t_change, color='r', linestyle='-')

    ref_sig, = ax[0].plot(t, reference_signal[:, 0], color=colors[0], alpha=0.5, linewidth=.75)
    ref_target, = ax[0].plot(t, targets[:, 0], linestyle=(0, (6, 1)), color=colors[3], alpha=0.5, linewidth=1)
    adapted_target, =ax[0].plot(t, adapted_targets[:, 0], linestyle=(0, (3, 1)),  color=colors[3], linewidth=.5)
    sig, = ax[0].plot(t, signal[:, 0], color=colors[0], linewidth=.75)
    # ax[0].legend(fontsize=8, loc='upper left')
    ax[0].set_ylabel("Position [m]", fontsize=7)
    ax[0].tick_params(axis='both', labelsize=6)
    ax[0].set_xlim(49, 56)
    ax[0].set_ylim(5)

    ref_u, = ax[1].plot(t, reference_controls, linestyle=(0, (2, 2)), color=colors[1], linewidth=1)
    u, = ax[1].plot(t, adapted_controls, color=colors[5], linewidth=.5)
#     ax[1].legend(fontsize=8, loc='upper left')
    ax[1].set_ylabel("Force [N]", fontsize=7)
    ax[1].set_xlabel("Time [s]", fontsize=7)
    ax[1].tick_params(axis='both', labelsize=6)

    if use_kb:
        selection_lines = ax[2].plot(t_reference_selection, js_selection, linestyle='-.', marker='x', lw=1., alpha=.5)
        js_loss_lines = ax[2].plot(t[60 * update_interval:][::60 * kb_step], kl_loss, marker='.', markersize=3., lw=1.,
                                   alpha=0.7)
        ax[2].axhline(thres, color='k', linestyle='--', lw=.8)
        ax[2].axhline(0., color='k', linestyle='--', lw=.8)
        ax[2].ticklabel_format(style='plain')
        ax[2].set_ylim(-0.1, np.log(2) + 0.1)
        legend1 = ax[2].legend(fontsize=8, loc='upper left',
                               handles=selection_lines, labels=[f"KB entry {i}" for i in range(initial_kb_len)])
        ax[2].add_artist(legend1)
        ax[2].legend(fontsize=8, loc='upper right',
                     handles=js_loss_lines, labels=["updated-kb-dist vs. running dist", "kb-dist vs. updated-kb-dist"])

    fig.legend(handles=[ref_target, adapted_target, ref_sig, sig, ref_u, u],
               labels=["Reference target", "Adapted target", "Reference signal", "Signal", "Reference control", "Control"],
               loc='lower left', fontsize=6, frameon=False, ncol=3, bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    fig.savefig(f"../../reports/Thesis/figures/method/toy_plot_closeup.{'pgf' if mpl.get_backend() == 'pgf' else 'png'}",
                dpi=300, bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    print("Using backend " + keras.backend.backend())
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    model_location = 'tmp/varautoencoder.keras'
    seed = np.random.randint(0, 1000)
    seed = 869
    # print(f"Seed: {seed}")
    keras.utils.set_random_seed(seed)

    if torch.cuda.is_available():
        print("Using CUDA")
        with torch.cuda.device(0):
            main()
    else:
        main()
