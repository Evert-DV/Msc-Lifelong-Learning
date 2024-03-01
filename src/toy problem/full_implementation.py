from kb_tools import *
from toy_tools import *
import pickle
import keras
from keras import optimizers, losses
from torch.utils.data import TensorDataset, DataLoader


def main():
    # Load 'vanilla' adapter model (or included in KB)
    adapter = keras.models.load_model("tmp/target_adapter.keras")
    prediction_window = 10
    optimizer = keras.optimizers.Adam(learning_rate=5.e-3)
    loss_fn = keras.losses.MeanSquaredError()
    adapter.compile(optimizer=optimizer, loss=loss_fn)

    # Load VAE model
    autoencoder = keras.models.load_model("tmp/varautoencoder.keras")
    optimizer = optimizers.Adam(learning_rate=1.e-4)
    loss_fn = losses.MeanSquaredError()
    autoencoder.compile(optimizer=optimizer, loss=loss_fn)

    # Define system
    system = System(5, 10, 3, 5)

    # Define controller
    controller = PIDController(350, 107.5, 1257)

    # Define reference controller
    reference_controller = PIDController(350, 107.5, 1257)

    # Setup KB
    use_kb = True
    if not use_kb:
        reference_data = np.load(f"tmp/train data/m5k10c3_seed951.npy")
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
    dt = 1 / 60
    t_end = 600
    t = np.arange(0, t_end, dt)
    target = np.array([11., 0.])
    buffer = []
    x0 = [9.9, 0]  # 9.9 was found to be the steady state
    x0_reference = [9.9, 0]
    train_interval = 30
    update_interval = 60
    kb_step = 5
    t_change = np.random.randint(update_interval, t_end - update_interval)

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
        # Target selection
        if ti % 15 == 0:
            target = [np.random.rand() * 6 + 7, 0.]
        targets.append(target)

        # Hard change for testing
        # if ti == t_change:
        #     print("\n\nHARD CHANGE\n"
        #           f"k: {system.k:.1f}\t -> {system.k + 10:.1f}\n"
        #           f"c: {system.c:.1f}\t -> {system.c + 10:.1f}\n"
        #           f"l0: {system.l0:.1f}\t -> {system.l0 / 3:.1f}")
        #     system.k += 10
        #     system.c += 3
        #     system.l0 /= 3

        # Control loop
        predicted_target = adapter.predict(ops.array([*x0, *target])[None], verbose=0)[0]
        adapted_targets.append(predicted_target)
        control_action = controller.compute_control(x0, predicted_target, dt)
        adapted_controls.append(control_action)
        x = system.response(x0, control_action, do_update=True)
        signal.append(x)
        x0 = x

        # Reference control loop
        reference_control = reference_controller.compute_control(x0_reference, target, dt)
        reference_controls.append(reference_control)
        x_reference = system.response(x0_reference, reference_control, do_update=False)
        reference_signal.append(x_reference)
        x0_reference = x_reference

        # Buffer data
        buffer.append([*x0, control_action, *x, *predicted_target])

        # Adapter update step
        if ti % train_interval == 0 and ti != 0:
            t_train.append(ti)
            adapter.optimizer.lr = 1.e-3
            buffer = ops.array(buffer)
            features, labels = prep_data(buffer, prediction_window, interval=15)

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
                         EpochLogger()
                         ]
            adapter.fit(train_dataloader,
                        epochs=100,
                        callbacks=callbacks,
                        validation_data=val_dataloader,
                        verbose=0,
                        )
            buffer = buffer.tolist()

        # KB step
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
            buffer = []

    # Save the last updated reference
    kb[0][current_kb_idx] = backup_updated_reference
    kb[1][current_kb_idx] = adapter.get_weights()
    print(f"\n\n{len(kb[0])} entries in the KB")
    with open('tmp/kb.pkl', 'wb') as f:
        pickle.dump(kb, f)
    print("KB saved")

    # Plotting
    signal = np.asarray(signal)
    reference_signal = np.asarray(reference_signal)
    targets = np.asarray(targets)
    adapted_targets = np.asarray(adapted_targets)

    fig, ax = plt.subplots(3, 1, figsize=(16, 8), sharex=True)

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

    ax[0].plot(t, reference_signal[:, 0], color='tab:blue', alpha=0.33, label="Reference controller")
    ax[0].plot(t, targets[:, 0], '--', color='tab:gray', label="Target position")
    ax[0].plot(t, signal[:, 0], color='tab:blue', label="Adaptive controller")
    ax[0].legend(fontsize=8, loc='upper left')

    ax[1].plot(t, targets[:, 0], '--', color='tab:gray', label="Target position")
    ax[1].plot(t, adapted_targets[:, 0], color='tab:blue', label="Adapted targets")
    ax[1].legend(fontsize=8, loc='upper left')

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

    fig.tight_layout()
    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    # fig.savefig("tmp/plot.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    print("Using backend " + keras.backend.backend())
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    model_location = 'tmp/varautoencoder.keras'
    seed = np.random.randint(0, 1000)
    # seed = 42
    print(f"Seed: {seed}")
    keras.utils.set_random_seed(seed)

    if torch.cuda.is_available():
        print("Using CUDA")
        with torch.cuda.device(0):
            main()
    else:
        main()
