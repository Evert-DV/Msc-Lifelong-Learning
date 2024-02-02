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
    windowed_data = np.array(
        [data[i:i + interval * 60] for i in range(0, len(data) - interval * 60 + 1)[::interval * 60]])

    features = ops.array(windowed_data)[..., [0, 1, 2, 3, 4]].reshape(-1, 5)
    labels = ops.array(windowed_data)[..., [0, 1, 2, 3, 4]].reshape(-1, 5)

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
            interval = 10
            windowed_data = np.array(
                [data[i:i + interval * 60] for i in range(0, len(data) - interval * 60 + 1)[::interval * 60]])

            features = ops.array(windowed_data)[..., [0, 1, 2, 3, 4]].reshape(-1, 5)

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
            plt.text(j, i, f'{scores_np[i, j]*1e-5:.1f}e5', ha='center', va='center', color='white')

    plt.title('Log-Likelihood of Datasets Under Different Distributions')
    plt.xlabel('Distributions')
    plt.ylabel('Embeddings')
    plt.xticks(np.arange(scores_np.shape[1]), labels=filenames, fontsize=8, rotation=45)
    plt.yticks(np.arange(scores_np.shape[0]), labels=filenames, fontsize=8)
    plt.tight_layout()
    plt.show()


def implement():
    pass


def main():
    seed = np.random.randint(0, 1000)
    seed = 42
    print(f"Seed: {seed}")
    keras.utils.set_random_seed(seed)

    # train()
    compare()


if __name__ == '__main__':
    print("Using backend " + keras.backend.backend())
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    if torch.cuda.is_available():
        print("Using CUDA")
        with torch.cuda.device(0):
            main()
    else:
        main()
