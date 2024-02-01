import os

os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import keras
from keras import layers, optimizers, losses, ops


def main():
    seed = np.random.randint(0, 1000)
    seed = 42
    print(f"Seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)

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
            layers.Dropout(0.25),
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

    optimizer = optimizers.Adam(learning_rate=1.e-5)
    loss_fn = losses.MeanSquaredError()
    autoencoder.compile(optimizer=optimizer, loss=loss_fn)

    data = np.load("tmp/train data/pretrain_data_m5.npy")
    interval = 10
    windowed_data = ops.array(
        [data[i:i + interval * 60] for i in range(0, len(data) - interval * 60 + 1)[::interval * 60]])

    features = windowed_data[..., [0, 1, 2, 3, 4]].reshape(-1, 5)
    labels = windowed_data[..., [0, 1, 2, 3, 4]].reshape(-1, 5)

    dataset = TensorDataset(features, labels)
    train_size = int(0.8 * len(dataset))
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

    # reconstruction losses
    # for i in range(len(labels))[::15*60]:
    #     losses.append(loss_fn(labels[i:i+15*60], autoencoder.predict(features[i:i+15*60], verbose=0)).item())
    # TODO: check linear independence test embeddings


if __name__ == '__main__':
    print("Using backend " + keras.backend.backend())
    if torch.cuda.is_available():
        print("Using CUDA")
        with torch.cuda.device(0):
            main()
    else:
        main()
