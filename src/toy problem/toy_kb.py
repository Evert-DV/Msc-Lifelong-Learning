import os

os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import keras
from keras import layers, optimizers, losses, ops


def representation(x, encoder):
    embeddings = encoder(x)
    mean = ops.mean(embeddings, axis=0)
    std = ops.std(embeddings, axis=0)
    return mean, std


def z_score(means, stds, sizes):
    z = (means[0] - means[1])/ops.sqrt(stds[0]**2/sizes[0] + stds[1]**2/sizes[1])

    return z


def train():
    model_location = 'tmp/autoencoder.keras'

    pretrain = False
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
        encoder = autoencoder.layers[0]
        decoder = autoencoder.layers[-1]

    optimizer = optimizers.Adam(learning_rate=1.e-4)
    loss_fn = losses.MeanSquaredError()
    autoencoder.compile(optimizer=optimizer, loss=loss_fn)

    data = np.load(f"tmp/train data/pretrain_data_w_update_m5k10c3_seed934.npy")
    interval = 10
    windowed_data = ops.array(
        [data[i:i + interval * 60] for i in range(0, len(data) - interval * 60 + 1)[::interval * 60]])

    features = windowed_data[..., [0, 1, 2, 3, 4]].reshape(-1, 5)
    labels = windowed_data[..., [0, 1, 2, 3, 4]].reshape(-1, 5)

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

    means = []
    stds = []
    for file in os.listdir("tmp/train data"):
        if file.endswith(".npy"):
            data = np.load(f"tmp/train data/{file}")
            interval = 10
            windowed_data = ops.array(
                [data[i:i + interval * 60] for i in range(0, len(data) - interval * 60 + 1)[::interval * 60]])

            features = windowed_data[..., [0, 1, 2, 3, 4]].reshape(-1, 5)

            mean, std = representation(features, encoder)
            print(f"File: {file}\n"
                  f"Mean: {mean.tolist()}")  #\tStd: {std.tolist()}\n")
            # TODO: check statistical tests for comparison of distributions
            means.append(mean)
            stds.append(std)

    print(f"Means: {means}")


def main():
    seed = np.random.randint(0, 1000)
    seed = 42
    print(f"Seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)

    # train()
    compare()


if __name__ == '__main__':
    print("Using backend " + keras.backend.backend())
    if torch.cuda.is_available():
        print("Using CUDA")
        with torch.cuda.device(0):
            main()
    else:
        main()
