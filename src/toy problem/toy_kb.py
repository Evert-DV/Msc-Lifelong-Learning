import os

os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.distributions import MultivariateNormal
import keras
from keras import layers, optimizers, losses, ops


def get_distribution(x, encoder):
    embeddings = encoder(x)
    mean = ops.mean(embeddings, axis=0)
    cov = torch.cov(embeddings.T)
    distribution = MultivariateNormal(mean, cov)

    return embeddings, distribution


def visualize_distribution(embeddings, distribution, dim='3d'):
    embeddings = embeddings[::30]
    data = ops.convert_to_numpy(embeddings)
    num_dimensions = data.shape[1]

    if dim == '3d':
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot for the data points
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], zdir='z', s=20, depthshade=True)

        samples = distribution.sample((1000,))
        probs = ops.convert_to_numpy(ops.exp(distribution.log_prob(samples)))
        samples = ops.convert_to_numpy(samples)

        # Normalize probabilities for color mapping
        min_prob, max_prob = probs.min(), probs.max()
        normalized_probs = (probs - min_prob) / (max_prob - min_prob)
        alpha_values = normalized_probs

        # Scatter plot with color gradient
        scatter = ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=normalized_probs, cmap='viridis',
                             alpha=alpha_values)

        # Colorbar to show the mapping from color to probability
        cbar = fig.colorbar(scatter, ax=ax)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

    elif dim == '2d':
        # Predefine combinations: For 3D data, this results in (0,1), (0,2), (1,2)
        combinations = [(i, j) for i in range(num_dimensions) for j in range(i + 1, num_dimensions)]

        # Set up the figure with subplots in a row
        fig, axs = plt.subplots(1, len(combinations), figsize=(5 * len(combinations), 5))

        for plot_idx, (i, j) in enumerate(combinations):
            other_dim = 3 - i - j  # Get the remaining dimension
            ax = axs[plot_idx]  # Get the current axis

            # Scatter plot for dimensions i vs j
            ax.scatter(data[:, i], data[:, j], alpha=0.5)

            # Overlay contour plot for the distribution
            x, y = np.meshgrid(np.linspace(data[:, i].min(), data[:, i].max(), 100),
                               np.linspace(data[:, j].min(), data[:, j].max(), 100))

            # Fix the other dimension at its mean value
            fixed_value = distribution.mean[other_dim].item()
            z = np.full_like(x, fixed_value)

            # Prepare position tensors for log_prob calculation
            pos = np.empty(x.shape + (3,))
            pos[:, :, i] = x
            pos[:, :, j] = y
            pos[:, :, other_dim] = z
            pos = ops.array(pos.reshape(-1, 3))
            prob = distribution.log_prob(pos).reshape(100, 100)
            z = ops.convert_to_numpy(ops.exp(prob))
            ax.contour(x, y, z, levels=5, colors='r')

            ax.set_xlabel(f'Dim {i}')
            ax.set_ylabel(f'Dim {j}')

    plt.tight_layout()
    plt.show()


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
    encoder = autoencoder.layers[0]

    latent_features = []
    distributions = []
    filenames = []
    for file in os.listdir("tmp/train data"):
        if file.endswith(".npy"):
            print(file.title())
            data = np.load(f"tmp/train data/{file}")
            # interval = 10
            # windowed_data = np.array(
            #     [data[i:i + interval * 60] for i in range(0, len(data) - interval * 60 + 1)[::interval * 60]])

            features = ops.array(data)[..., [0, 1, 2, 3, 4]]  # .reshape(-1, 5)

            embeddings, dist = get_distribution(features, encoder)
            latent_features.append(embeddings)
            distributions.append(dist)
            filenames.append(file.split('_')[0] + "_w/update" if "update" in file else file.split('_')[0])

    scores = ops.empty((len(latent_features), len(latent_features)))
    for i, dist in enumerate(distributions):
        for j, data in enumerate(latent_features):
            total_log_likelihood = ops.sum(dist.log_prob(data))
            scores[i, j] = total_log_likelihood

    scores_np = ops.convert_to_numpy(scores)
    plt.figure(figsize=(10, 8))
    cax = plt.imshow(scores_np, cmap='viridis', aspect='auto')
    plt.colorbar(cax, label='Log-Likelihood')

    # Optional: Annotate the heatmap with exact log-likelihood values
    for i in range(scores_np.shape[0]):
        for j in range(scores_np.shape[1]):
            plt.text(j, i, f'{scores_np[i, j] * 1e-5:.1f}e5', ha='center', va='center', color='white')

    plt.title('Log-Likelihood of Datasets Under Different Distributions')
    plt.xlabel('Distributions')
    plt.ylabel('Embeddings')
    plt.xticks(np.arange(scores_np.shape[1]), labels=filenames, fontsize=8, rotation=45)
    plt.yticks(np.arange(scores_np.shape[0]), labels=filenames, fontsize=8)
    plt.tight_layout()
    plt.show()


def implement():
    autoencoder = keras.models.load_model('tmp/autoencoder.keras')
    encoder = autoencoder.layers[0]
    prior_data = np.load(f"tmp/train data/m5k10c3_w-update_seed314.npy")
    x = ops.array(prior_data)[..., [0, 1, 2, 3, 4]].reshape(-1, 5)
    emb, prior = get_distribution(x, encoder)
    baseline = prior.log_prob(encoder(x)).mean()
    visualize_distribution(emb, prior)

    data = np.load(f"tmp/train data/m5k20c3_seed879.npy")
    t = np.arange(0, len(data) / 60, 1 / 60)
    loss = []
    likelihoods = []
    reconstructed = []
    rho = 0.9
    step = 60
    for i in np.arange(1, len(data), step):
        x = ops.array(data[i:i + step, [0, 1, 2, 3, 4]])
        y = autoencoder(x)
        reconstructed.append(y)
        likelihood = prior.log_prob(encoder(ops.array(data[0:i + step, [0, 1, 2, 3, 4]]))).mean()

        # if i < step:
        #     ewma = likelihood.item()
        # else:
        #     ewma = rho * ewma + (1 - rho) * likelihood.item()
        #     # ewma = (likelihood.item() + (i - step / 2) * ewma) / i
        #
        # loss.append(ewma)
        likelihoods.append(likelihood.item())

    reconstructed = ops.convert_to_numpy(ops.concatenate(reconstructed, axis=0))

    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(t, data[:, 0], label="signal")
    ax[0].plot(t, data[:, -2], label="target")
    ax[0].plot(t[:-1], reconstructed[:, 0], label="reconstructed signal")

    # ax[1].plot(t[::step], loss, label="loss")
    ax[1].plot(t[::step], likelihoods, '.', markersize=1., label="likelihood")
    ax[1].axhline(baseline.item(), color='r', linestyle='--', label="baseline")
    ax[1].legend()

    fig.tight_layout()
    plt.show()


def main():
    seed = np.random.randint(0, 1000)
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
