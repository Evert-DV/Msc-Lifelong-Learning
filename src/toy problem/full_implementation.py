from kb_tools import *
from toy_tools import *
import pickle
import keras
from keras import optimizers, losses
from torch.utils.data import TensorDataset, DataLoader


def main():
    # Load adapter model (or included in KB)
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
    system = System(5, 20, 6, 5)

    # Define controller
    controller = PIDController(350, 107.5, 1257)

    # Define reference controller
    reference_controller = PIDController(350, 107.5, 1257)

    # Setup KB
    use_kb = True
    if not use_kb:
        prior_data = np.load(f"tmp/train data/m5k10c3_seed951.npy")
        x_prior = ops.array(prior_data)[..., [0, 1, 2, 3, 4]].reshape(-1, 5)
        z_mean, z_log_var = autoencoder.dynamics(x_prior)
        cov = sample(z_mean, z_log_var)[1]
        prior = MultivariateNormal(z_mean, cov)
        kb = [[prior], [adapter.get_weights()]]
    else:
        with open('tmp/kb.pkl', 'rb') as f:
            kb = pickle.load(f)

    initial_kb_len = len(kb[0])
    print(f"{initial_kb_len} entries in the KB\n")
    thres = np.log(2) * 0.25

    # Setup simulation loop
    dt = 1 / 60
    t = np.arange(0, 600, dt)
    target = np.array([11., 0.])
    buffer = []
    x0 = [9.9, 0]  # 9.9 was found to be the steady state
    x0_reference = [9.9, 0]

    # Setup plotting lists
    signal = []
    reference_signal = []
    targets = []
    adapted_targets = []
    reference_controls = []
    adapted_controls = []
    kl_loss = []
    js_selection = []
    t_updates = []
    t_trespassing = []
    t_kb_selection = []
    t_prior_selection = []

    # Simulation loop
    for ti in t:
        print(f"\rt =  {ti:.0f}", end="")
        # Target selection
        if ti % 15 == 0:
            target = [np.random.rand() * 6 + 7, 0.]
        targets.append(target)

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
        if ti % 20 == 0 and ti != 0:
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
        if ti % 5 == 0:
            # Get embeddings
            z_mean, z_log_var = autoencoder.dynamics(ops.array(buffer)[..., [0, 1, 2, 3, 4]])
            embeddings, cov = sample(z_mean, z_log_var, samples_per_centroid=100)

            # Pre-60 seconds
            if ti < 60:
                t_prior_selection.append(ti)
                running_distribution = MultivariateNormal(z_mean, cov)
                kb_idx, js_divs = search_kb(running_distribution, kb[0])
                js_selection.append(js_divs)
                continue

            if ti == 60:
                # Select KB entry
                prior = kb[0][kb_idx]
                updated_prior = prior.copy()
                backup_updated_prior = prior.copy()
                adapter.set_weights(kb[1][kb_idx])
                print(f"\nSelected {kb_idx} as reference")

            # Post-60 seconds
            # KL losses
            kl_updated_dist = js_divergence(updated_prior, running_distribution)
            kl_prior_updated = js_divergence(prior, updated_prior)
            kl_loss.append([kl_updated_dist.item(), kl_prior_updated.item()])

            # Check for shift
            if kl_updated_dist > thres or kl_prior_updated > .9 * thres:
                print("\nSHIFT DETECTED\nRestore KB entry reference")
                t_trespassing.append(ti)
                # retain last or leave it as it was?
                # kb[0][kb_idx] = backup_updated_prior.copy()
                kb[1][kb_idx] = adapter.get_weights()
                print("Check KB for better match")
                best_idx, js_divs = search_kb(running_distribution, kb[0])
                if best_idx != kb_idx and js_divs[best_idx] < thres:
                    print(f"Use KB entry {best_idx} as reference")
                    t_kb_selection.append(ti)
                    prior = kb[0][best_idx]
                    updated_prior = prior.copy()
                    backup_updated_prior = prior.copy()
                    kb_idx = best_idx
                    adapter.set_weights(kb[1][best_idx])
                else:
                    print("No match found\nInitiate new KB entry")
                    prior = running_distribution.copy()
                    updated_prior = prior.copy()
                    kb[0].append(prior)
                    kb[1].append(adapter.get_weights())
                    kb_idx = len(kb[0]) - 1

        # KB update step
        if ti % 60 == 0 and ti != 0:
            print('\nUPDATE STEP')
            torch.cuda.empty_cache()
            t_updates.append(ti)

            # Update running distribution
            running_distribution.update(z_mean, cov, weight=0.5)
            print("Check KB for better match")
            best_idx, js_divs = search_kb(running_distribution, kb[0])
            if best_idx != kb_idx and js_divs[best_idx] < thres:
                print(f"Use KB entry {best_idx} as reference")
                t_kb_selection.append(ti)
                kb[1][kb_idx] = adapter.get_weights()
                prior = kb[0][best_idx]
                updated_prior = prior.copy()
                backup_updated_prior = prior.copy()
                kb_idx = best_idx
                adapter.set_weights(kb[1][best_idx])
                continue

            # Update prior
            print("No match found")
            backup_updated_prior = updated_prior.copy()
            updated_prior.update(z_mean, cov, weight=0.1)
            print(f"Updated prior")

            # Empty buffer
            buffer = []

    # Save the last updated prior
    kb[0][-1] = backup_updated_prior
    kb[1][-1] = adapter.get_weights()
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

    ax[0].plot(t, reference_signal[:, 0], color='tab:blue', alpha=0.33, label="Reference controller")
    ax[0].plot(t, targets[:, 0], '--', color='tab:gray', label="Target position")
    ax[0].plot(t, signal[:, 0], color='tab:blue', label="Adaptive controller")
    for t_update in t_updates:
        ax[0].axvline(t_update, color='lightgrey', linestyle='-')
    for t_kb in t_kb_selection:
        ax[0].axvline(t_kb, color='tab:green', linestyle='--')
    for t_tres in t_trespassing:
        ax[0].axvline(t_tres, color='tab:red', linestyle=':')
    ax[0].legend(fontsize=8, loc='upper left')

    ax[1].plot(t, targets[:, 0], '--', color='tab:gray', label="Target position")
    ax[1].plot(t, adapted_targets[:, 0], color='tab:blue', label="Adapted targets")
    ax[1].legend(fontsize=8, loc='upper left')

    ax[2].plot(t_prior_selection, js_selection, linestyle='-.', marker='x', lw=1., alpha=.5,
               label=[f"KB entry {i}" for i in range(initial_kb_len)])
    ax[2].ticklabel_format(style='plain')
    ax[2].legend()

    ax[2].vlines(t_updates, 0, 1, colors='lightgrey', linestyles='-')
    ax[2].vlines(t_kb_selection, 0, 1, colors='tab:green', linestyles='--')
    ax[2].vlines(t_trespassing, 0, 1, colors='tab:red', linestyles=':')
    ax[2].plot(t[60 * 60:][::300], kl_loss, lw=1., alpha=0.7,
               label=["updated-kb-dist vs. running dist", "kb-dist vs. updated-kb-dist"])
    ax[2].hlines([-thres, 0, thres], 0., t[-1], color='k', linestyle='--', lw=.8)
    ax[2].set_ylim(-0.1, np.log(2) + 0.1)
    ax[2].legend()

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
