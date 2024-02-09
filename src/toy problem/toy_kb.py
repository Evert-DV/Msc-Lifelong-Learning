import os

os.environ["KERAS_BACKEND"] = "torch"
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.distributions.kl import kl_divergence
import keras
from keras import layers, optimizers, losses
from kb_tools import *


def train():
    pretrain = False
    data = np.load(f"tmp/train data/m5k10c3_seed388.npy")
    features = ops.array(data)[..., [0, 1, 2, 3, 4]]
    labels = ops.array(data)[..., [0, 1, 2, 3, 4]]

    if pretrain:
        encoder = keras.Sequential([
            layers.Input(shape=(5,)),
            # layers.Normalization(),
            # layers.Dense(32, activation='relu'),
            # layers.Dropout(0.1),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(3),
        ])

        decoder = keras.Sequential([
            layers.Input(shape=(3,)),
            # layers.Dense(32, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(5),
            # layers.Normalization(invert=True),
        ])

        # normalization
        # encoder.layers[0].adapt(features)
        # decoder.layers[-1].adapt(features)

        autoencoder = keras.Sequential([
            encoder,
            decoder
        ])
    else:
        autoencoder = keras.models.load_model(model_location)

    optimizer = optimizers.Adam(learning_rate=1.e-4)
    loss_fn = losses.MeanSquaredError()
    autoencoder.compile(optimizer=optimizer, loss=loss_fn)

    dataset = TensorDataset(features, labels)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=256, shuffle=False)

    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                               mode='min',
                                               min_delta=1e-4,
                                               patience=7,
                                               restore_best_weights=True,
                                               verbose=1),
                 ]
    autoencoder.fit(train_dataloader,
                    epochs=1000,
                    callbacks=callbacks,
                    validation_data=val_dataloader,
                    )

    autoencoder.save(model_location)


def compare():
    autoencoder = keras.models.load_model(model_location)
    autoencoder.eval()
    encoder = autoencoder.layers[0]

    latent_features = []
    distributions = []
    filenames = []
    for file in os.listdir("tmp/train data"):
        if file.endswith(".npy"):
            print(file.title())
            data = np.load(f"tmp/train data/{file}")
            features = ops.array(data)[..., [0, 1, 2, 3, 4]]  # .reshape(-1, 5)
            embeddings = encoder(features)
            dist = GaussianDensityEstimation(embeddings, bandwidth=15, n_points=100)

            # visualize_distribution(dist, embeddings[::10])

            latent_features.append(embeddings)
            distributions.append(dist)
            filenames.append(file.split('_')[0] + "_w/update" if "update" in file else file.split('_')[0])

    scores = ops.empty((len(latent_features), len(latent_features)))
    for i, dist in enumerate(distributions):
        for j, dist2 in enumerate(distributions):
            scores[i, j] = kl_divergence(dist, dist2).item()

    scores_np = ops.convert_to_numpy(scores)
    plt.figure(figsize=(10, 8))
    cax = plt.imshow(scores_np, cmap='viridis', aspect='auto')
    plt.colorbar(cax, label='KL-Divergence')

    # Optional: Annotate the heatmap with exact log-likelihood values
    for i in range(scores_np.shape[0]):
        for j in range(scores_np.shape[1]):
            plt.text(j, i, f'{scores_np[i, j]:.2f}', ha='center', va='center', color='white')

    plt.title('KL-Divergence of Datasets Under Different Distributions')
    plt.xlabel('Distributions')
    plt.ylabel('Embeddings')
    plt.xticks(np.arange(scores_np.shape[1]), labels=filenames, fontsize=8, rotation=45)
    plt.yticks(np.arange(scores_np.shape[0]), labels=filenames, fontsize=8)
    plt.tight_layout()
    plt.show()


def implement():
    global kb_idx, ewma_post_prob, updated_prior, prior, backup_updated_prior, running_distribution
    autoencoder = keras.models.load_model(model_location)
    autoencoder.eval()
    encoder = autoencoder.layers[0]
    bw = 15.
    use_kb = True

    if not use_kb:
        prior_data = np.load(f"tmp/train data/m5k10c3_seed951.npy")
        x_prior = ops.array(prior_data)[..., [0, 1, 2, 3, 4]].reshape(-1, 5)

        prior = GaussianDensityEstimation(encoder(x_prior), encoder, bandwidth=bw)
        kb = [[prior], [1.]]
    else:
        with open('tmp/kb.pkl', 'rb') as f:
            kb = pickle.load(f)
    initial_kb_len = len(kb[0])
    print(f"{initial_kb_len} entries in the KB\n")

    # Simulate a realtime implementation
    data = np.load(f"tmp/train data/m5k20c6_w-update_seed843.npy")
    data = ops.array(data)
    t = np.arange(0, len(data) / 60, 1 / 60)
    kl_loss = []
    posteriors = []
    updates = []
    trespassing = []
    posterior_selection_idx = []
    thres = 2.5
    step = 300
    for i in np.arange(0, len(data), step):
        print(f"\rt =  {t[i]:.0f}", end="")
        x = data[0:i + step, [0, 1, 2, 3, 4]]

        if i < 60 * 60:
            embeddings = encoder(x)
            posterior_selection_idx.append(i)
            # Likelihoods and relative posteriors
            post_probs = get_posteriors(embeddings, kb[0], kb[1])
            posterior_ewma = []
            for idx, post_prob in enumerate(post_probs):
                if i == 0:
                    ewma_post_prob = post_prob.mean(axis=-1)
                else:
                    ewma_post_prob = ewma(post_prob, ewma_post_prob, rho=0.1, axis=-1)
                posterior_ewma.append(ewma_post_prob.item())
                kb[1][idx] = ewma_post_prob.item()  # update P(dist) in kb
            posteriors.append(posterior_ewma)
            kb_idx = np.argmax(posterior_ewma)  # rho can really influence the result
            prior = kb[0][kb_idx]
            updated_prior = prior.copy()
            backup_updated_prior = prior.copy()
            continue

        # Get embeddings
        embeddings = encoder(x)

        if i < 120 * 60:
            if i == 60 * 60:
                print(f"\nSelected {kb_idx} as reference")
            if i == 120 * 60 - step:
                running_distribution = GaussianDensityEstimation(embeddings, bandwidth=bw, n_points=100)
            continue

        # KL losses
        kl_updated_dist = kl_divergence(updated_prior, running_distribution)
        kl_prior_updated = kl_divergence(prior, updated_prior)
        kl_prior_dist = kl_divergence(prior, running_distribution)  # sanity check
        kl_loss.append([kl_updated_dist.item(), kl_prior_updated.item(), kl_prior_dist.item()])  # ,

        if (kl_updated_dist > thres or kl_prior_updated > .9 * thres) and i >= 2 * 60 * 60:
            print("\nSHIFT DETECTED\nRestore KB entry reference")
            trespassing.append(i)
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
            kb_idx = len(kb) - 1

        if i % (60 * 60) == 0:
            print('\nUPDATE STEP')
            torch.cuda.empty_cache()
            updates.append(i)
            # Update running distribution
            running_distribution.update(embeddings[i - 3600:i], weight=0.3)
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
            updated_prior.update(embeddings[i - 3600:i], weight=0.1)
            # If storing all recorded data is not a problem (i.e.: data[0:i, ...]), consider updating by:
            # updated_prior = GaussianDensityEstimation(embeddings, bandwidth=bw, n_points=100)
            print(f"Updated prior with {3600} samples")

    # Save the last updated prior
    kb[0][-1] = backup_updated_prior
    kb[1] = len(kb[1]) * [1 / len(kb[1])]
    print(f"\n\n{len(kb[0])} entries in the KB")
    with open('tmp/kb.pkl', 'wb') as f:
        pickle.dump(kb, f)
    print("KB saved")

    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(t[posterior_selection_idx], posteriors, linestyle='-.', marker='x', lw=1., alpha=.5,
               label=[f"P(dist_{i}|emb)" for i in range(initial_kb_len)])
    ax[0].legend()

    ax[1].vlines(t[updates], 0, 15, colors='tab:gray', linestyles=':')
    ax[1].vlines(t[trespassing], 0, 15, colors='tab:red', linestyles=':')
    ax[1].plot(t[120 * 60:][::step], kl_loss, lw=1., alpha=0.7,
               label=["updated-prior vs. dist", "prior vs. updated-prior", "prior vs. dist (sanity check)"])
    ax[1].hlines([-thres, 0, thres], 0., t[-1], color='k', linestyle='--', lw=.8)
    ax[1].set_ylim(-.5, 10.)
    ax[1].legend()

    fig.tight_layout()
    plt.show()


def main():
    # train()
    # compare()
    implement()


if __name__ == '__main__':
    print("Using backend " + keras.backend.backend())
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    model_location = 'tmp/autoencoder_relu.keras'
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
