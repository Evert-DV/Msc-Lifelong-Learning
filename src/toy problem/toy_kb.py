import os

os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.distributions.kl import kl_divergence
import keras
from keras import layers, optimizers, losses
from kb_tools import *


def train():
    model_location = 'tmp/autoencoder.keras'

    pretrain = True
    if pretrain:
        encoder = keras.Sequential([
            layers.Input(shape=(5,)),
            layers.Normalization(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.25),
            layers.Dense(3),
        ])

        decoder = keras.Sequential([
            layers.Input(shape=(3,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(5),
        ])

        autoencoder = keras.Sequential([
            encoder,
            decoder
        ])
    else:
        autoencoder = keras.models.load_model(model_location)

    optimizer = optimizers.Adam(learning_rate=1.e-4)
    loss_fn = losses.MeanSquaredError()
    autoencoder.compile(optimizer=optimizer, loss=loss_fn)

    data = np.load(f"tmp/train data/m5k10c3_seed131.npy")
    interval = 10
    # windowed_data = np.array(
    #     [data[i:i + interval * 60] for i in range(0, len(data) - interval * 60 + 1)[::interval * 60]])

    features = ops.array(data)[..., [0, 1, 2, 3, 4]]  # .reshape(-1, 5)
    labels = ops.array(data)[..., [0, 1, 2, 3, 4]]  # .reshape(-1, 5)

    dataset = TensorDataset(features, labels)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=256, shuffle=False)

    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                               mode='min',
                                               min_delta=1e-4,
                                               patience=5,
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
    model_location = 'tmp/autoencoder.keras'
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
            embeddings, dist = get_distribution(features, encoder, bandwidth=1.)

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
    autoencoder = keras.models.load_model('tmp/autoencoder.keras')
    encoder = autoencoder.layers[0]

    prior_data = np.load(f"tmp/train data/m4k9c2_seed24.npy")
    x_prior = ops.array(prior_data)[..., [0, 1, 2, 3, 4]].reshape(-1, 5)
    bw = 1.
    _, prior = get_distribution(x_prior, encoder, bandwidth=bw)
    _, updated_prior = get_distribution(x_prior, encoder, bandwidth=bw)

    data = np.load(f"tmp/train data/m6k11c4_seed219.npy")
    data = ops.array(data)
    t = np.arange(0, len(data) / 60, 1 / 60)
    kl_loss = []
    likelihoods = [[], [], [], []]
    step = 300
    for i in np.arange(0, len(data), step):
        print(f"\rt =  {t[i]:.0f}", end="")
        x = data[i:i + step, [0, 1, 2, 3, 4]]
        embeddings = encoder(x)
        embeddings_prior = encoder(x_prior[i:i + step])

        # Get distributions
        _, running_prior = get_distribution(x_prior[0:i + step], encoder, bandwidth=bw)
        _, dist = get_distribution(data[0:i + step, [0, 1, 2, 3, 4]], encoder, bandwidth=bw)

        # Likelihoods
        likelihoods[0].append(prior.log_prob(embeddings)[::30])
        likelihoods[1].append(updated_prior.log_prob(embeddings)[::30])
        likelihoods[2].append(prior.log_prob(embeddings_prior)[::30])
        likelihoods[3].append(updated_prior.log_prob(embeddings_prior)[::30])

        # KL losses
        kl_prior_dist = kl_divergence(prior, dist)
        kl_updated_dist = kl_divergence(updated_prior, dist)
        kl_prior_rprior = kl_divergence(prior, running_prior)
        kl_updated_rprior = kl_divergence(updated_prior, running_prior)
        kl_loss.append([kl_prior_dist.item(), kl_updated_dist.item(), kl_prior_rprior.item(), kl_updated_rprior.item()])

        # Update prior
        if i % (300 * 12) == 0 and i != 0:
            recorded_data = data[i - 3600:i, [0, 1, 2, 3, 4]]
            embeddings = encoder(recorded_data)
            updated_prior.update(embeddings, weight=0.25)
            print(f"\nUpdated prior with {len(recorded_data)} samples")

    print(f'\nBandwidth: {bw}\t')
    for i, p in enumerate(likelihoods):
        likelihoods[i] = ops.convert_to_numpy(ops.concatenate(p, axis=-1))

    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(t[::30], np.array(likelihoods).T, linestyle='-.', marker='x',
               label=["data|prior", "data|updated-prior", "running-prior|prior", "running-prior|updated-prior"])
    ax[0].legend()

    ax[1].vlines(t[::3600], 0, 5, colors='tab:gray', linestyles=':')
    ax[1].plot(t[::step], kl_loss,
               label=["prior|data", "updated-prior|data", "prior|running-prior", "updated-prior|running-prior"])
    ax[1].axhline(0., color='k', linestyle='--')
    ax[1].legend()

    fig.tight_layout()
    plt.show()


def main():
    # seed = np.random.randint(0, 1000)
    seed = 42
    print(f"Seed: {seed}")
    keras.utils.set_random_seed(seed)

    # train()
    # compare()
    implement()


if __name__ == '__main__':
    print("Using backend " + keras.backend.backend())
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    if torch.cuda.is_available():
        print("Using CUDA")
        with torch.cuda.device(0):
            main()
    else:
        main()
