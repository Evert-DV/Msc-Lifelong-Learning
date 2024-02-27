from kb_tools import *
from toy_tools import *
import pickle
import keras
from keras import optimizers, losses


def main():
    # Load adapter model (or included in KB)
    adapter = keras.models.load_model("tmp/target_adapter.keras")
    prediction_window = 10
    optimizer = keras.optimizers.Adam(learning_rate=5.e-3)
    loss_fn = keras.losses.MeanSquaredError()
    adapter.compile(optimizer=optimizer, loss=loss_fn)

    # Load VAE model
    autoencoder = autoencoder = keras.models.load_model("tmp/varautoencoder.keras")
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
    use_kb = False
    if not use_kb:
        prior_data = np.load(f"tmp/train data/m5k10c3_seed951.npy")
        x_prior = ops.array(prior_data)[..., [0, 1, 2, 3, 4]].reshape(-1, 5)
        z_mean, z_log_var = autoencoder.dynamics(x_prior)
        cov = sample(z_mean, z_log_var)[1]
        prior = MultivariateNormal(z_mean, cov)

        kb = [[prior], [1.], []]
    else:
        with open('tmp/kb.pkl', 'rb') as f:
            kb = pickle.load(f)
    initial_kb_len = len(kb[0])
    print(f"{initial_kb_len} entries in the KB\n")
    thres = np.log(2) * 0.25

    # Setup simulation loop
    dt = 1 / 60
    t = np.arange(0, 300, dt)
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
    predicted_targets = []
    kl_loss = []
    posteriors = []
    updates = []
    trespassing = []
    kb_selection = []
    posterior_selection_idx = []

    # Simulation loop
    for ti in t:
        print(f"\rt =  {ti:.0f}", end="")
        # Target selection
        if ti % 15 == 0:
            target = [np.random.rand() * 6 + 7, 0.]
            buffer = []
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

        buffer.append([*x0, control_action, *x, *predicted_target])

        # KB step
        if ti % 5 == 0:
            # Get embeddings
            z_mean, z_log_var = autoencoder.dynamics(ops.array(buffer)[..., [0, 1, 2, 3, 4]])
            embeddings, cov = sample(z_mean, z_log_var, samples_per_centroid=100)

            # Pre-60 seconds
            if ti < 60:
                # Likelihoods and relative posteriors
                post_probs = get_posteriors(embeddings, kb[0], kb[1])
                posterior_ewma = []
                for idx, post_prob in enumerate(post_probs):
                    if ti == 0:
                        ewma_post_prob = post_prob.mean(axis=-1)
                    else:
                        ewma_post_prob = ewma(post_prob, ewma_post_prob, rho=0.1, axis=-1)
                    posterior_ewma.append(ewma_post_prob.item())
                    kb[1][idx] = ewma_post_prob.item()  # update P(dist) in kb
                posteriors.append(posterior_ewma)
                continue

            # Select KB entry
            if ti == 60:
                kb_idx = np.argmax(posterior_ewma)  # rho can really influence the result
                prior = kb[0][kb_idx]
                updated_prior = prior.copy()
                backup_updated_prior = prior.copy()
                print(f"\nSelected {kb_idx} as reference")
                running_distribution = MultivariateNormal(z_mean, cov)  # TODO: fix this
                continue

            # Post-60 seconds
            # KL losses
            kl_updated_dist = js_divergence(updated_prior, running_distribution)
            kl_prior_updated = js_divergence(prior, updated_prior)
            kl_loss.append([kl_updated_dist.item(), kl_prior_updated.item()])

            # Check for shift
            if (kl_updated_dist > thres or kl_prior_updated > .9 * thres):
                print("\nSHIFT DETECTED\nRestore KB entry reference")
                # trespassing.append(i)
                # kb[kb_idx] = backup_updated_prior.copy()  # or leave it as it was?
                print("Check KB for better match")
                best_idx = search_dists(embeddings, kb[0], kb[1], running_distribution, kb_idx, thres)
                if best_idx is not None:
                    print(f"Use KB entry {best_idx} as reference")
                    prior = kb[0][best_idx]
                    updated_prior = prior.copy()
                    backup_updated_prior = prior.copy()
                    kb_idx = best_idx
                    continue

                print("No match found\nInitiate new KB entry")
                prior = running_distribution.copy()
                updated_prior = prior.copy()
                kb[0].append(prior)
                kb[1] = expand_prior_probs(kb[1])
                kb_idx = len(kb[0]) - 1

        # Update step
        if ti % 60 == 0:
            print('\nUPDATE STEP')
            torch.cuda.empty_cache()
            # updates.append(i)
            # Update running distribution
            running_distribution.update(z_mean, cov, weight=0.5)
            print("Check KB for better match")
            best_idx = search_dists(embeddings, kb[0], kb[1], running_distribution, kb_idx, thres)
            if best_idx is not None:
                print(f"Use KB entry {best_idx} as reference")
                prior = kb[0][best_idx]
                updated_prior = prior.copy()
                backup_updated_prior = prior.copy()
                kb_idx = best_idx
                continue

            # Update prior
            print("No match found")
            backup_updated_prior = updated_prior.copy()
            updated_prior.update(z_mean, cov, weight=0.5)
            print(f"Updated prior")

    #       Update running distribution

    # Save the last updated prior
    kb[0][-1] = backup_updated_prior
    kb[1] = len(kb[1]) * [1 / len(kb[1])]
    print(f"\n\n{len(kb[0])} entries in the KB")
    with open('tmp/kb.pkl', 'wb') as f:
        pickle.dump(kb, f)
    print("KB saved")


if __name__ == '__main__':
    print("Using backend " + keras.backend.backend())
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    model_location = 'tmp/varautoencoder.keras'
    seed = np.random.randint(0, 1000)
    # seed = 267
    print(f"Seed: {seed}")
    keras.utils.set_random_seed(seed)

    if torch.cuda.is_available():
        print("Using CUDA")
        with torch.cuda.device(0):
            main()
    else:
        main()
