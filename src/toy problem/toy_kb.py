import os

os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.distributions.kl import kl_divergence
import keras
from keras import layers, optimizers, losses
from kb_tools import *


def train():
    pretrain = False
    if pretrain:
        encoder = keras.Sequential([
            layers.Input(shape=(5,)),
            layers.Normalization(),
            layers.Dense(32, activation='leaky_relu'),
            layers.Dropout(0.25),
            layers.Dense(3),
        ])

        decoder = keras.Sequential([
            layers.Input(shape=(3,)),
            layers.Dense(32, activation='leaky_relu'),
            layers.Dense(5),
        ])

        autoencoder = keras.Sequential([
            encoder,
            decoder
        ])
    else:
        autoencoder = keras.models.load_model(model_location)

    optimizer = optimizers.Adam(learning_rate=1.e-5)
    loss_fn = losses.MeanSquaredError()
    autoencoder.compile(optimizer=optimizer, loss=loss_fn)

    data = np.load(f"tmp/train data/m5k10c3_seed131.npy")

    features = ops.array(data)[..., [0, 1, 2, 3, 4]]
    labels = ops.array(data)[..., [0, 1, 2, 3, 4]]

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
            embeddings, dist = get_distribution(features, encoder, bandwidth=0.1)

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
    autoencoder = keras.models.load_model(model_location)
    encoder = autoencoder.layers[0]

    prior_data = np.load(f"tmp/train data/m4k9c2_seed24.npy")
    x_prior = ops.array(prior_data)[..., [0, 1, 2, 3, 4]].reshape(-1, 5)
    bw = 0.1
    _, prior = get_distribution(x_prior, encoder, bandwidth=bw)
    # _, updated_prior = get_distribution(x_prior, encoder, bandwidth=bw)
    updated_prior = prior.copy()

    data = np.load(f"tmp/train data/m6k11c4_seed219.npy")
    data = ops.array(data)
    t = np.arange(0, len(data) / 60, 1 / 60)
    kl_loss = []
    posteriors = []
    updates = []
    trespassing = []
    thres = 1.5
    step = 300
    for i in np.arange(0, len(data), step):
        print(f"\rt =  {t[i]:.0f}", end="")
        x = data[i:i + step, [0, 1, 2, 3, 4]]
        embeddings = encoder(x)
        embeddings_prior = encoder(x_prior[i:i + step])

        # Get distributions
        _, running_prior = get_distribution(x_prior[0:i + step], encoder, bandwidth=bw)
        _, dist = get_distribution(data[0:i + step, [0, 1, 2, 3, 4]], encoder, bandwidth=bw)

        # # Likelihoods
        # p_y_dist = ops.exp(dist.log_prob(embeddings))
        # p_y_prior = ops.exp(prior.log_prob(embeddings))
        # p_y_updated_prior = ops.exp(updated_prior.log_prob(embeddings))
        # p_any = 1 / 3
        # p_y = p_any * (p_y_dist + p_y_prior + p_y_updated_prior)
        # p_dist_y = p_y_dist * p_any / p_y
        # p_prior_y = p_y_prior * p_any / p_y
        # p_updated_prior_y = p_y_updated_prior * p_any / p_y
        # if i < step:
        #     ewma_p_dist_y = p_dist_y.mean()
        #     ewma_p_prior_y = p_prior_y.mean()
        #     ewma_p_updated_prior_y = p_updated_prior_y.mean()
        # else:
        #     ewma_p_dist_y = ewma(p_dist_y, ewma_p_dist_y)
        #     ewma_p_prior_y = ewma(p_prior_y, ewma_p_prior_y)
        #     ewma_p_updated_prior_y = ewma(p_updated_prior_y, ewma_p_updated_prior_y)
        # posteriors.append([ewma_p_dist_y.item(), ewma_p_prior_y.item(), ewma_p_updated_prior_y.item()])

        # KL losses
        kl_updated_dist = kl_divergence(updated_prior, dist)
        kl_prior_updated = kl_divergence(prior, updated_prior)
        kl_prior_dist = kl_divergence(prior, dist)
        # kl_prior_rprior = kl_divergence(prior, running_prior)
        # kl_updated_rprior = kl_divergence(updated_prior, running_prior)
        kl_loss.append([kl_updated_dist.item(), kl_prior_updated.item(), kl_prior_dist.item()])  # ,
                        # kl_prior_rprior.item(), kl_updated_rprior.item()])

        if (kl_updated_dist > thres or kl_prior_updated > thres) and i > 2 * 60 * 60:
            print("\nUPDATE KB ENTRY RELATED TO PRIOR")
            print("<<placeholder: set entry's prior to updated_prior>>")
            print("INITIATE NEW KB ENTRY WITH CURRENT DISTRIBUTION")
            prior = dist.copy()
            updated_prior = dist.copy()
            trespassing.append(i)

        # Update prior
        update_dist = (i % (60 * 60) == 0 and i > 60 * 60)
        if update_dist:
            updates.append(i)
            recorded_data = data[i - 3600:i, [0, 1, 2, 3, 4]]
            embeddings = encoder(recorded_data)
            updated_prior.update(embeddings, weight=0.25)
            print(f"\nUpdated prior with {len(recorded_data)} samples")

    print(f'\nBandwidth: {bw}\t')

    fig, ax = plt.subplots(1, 1, sharex=True)

    # ax[0].plot(t[::step], posteriors, linestyle='-.', marker='x',
    #            label=["P(dist|data)", "P(prior|data)", "P(updated-prior|data)"])
    # ax[0].legend()

    ax.vlines(t[updates], 0, 15, colors='tab:gray', linestyles=':')
    ax.vlines(t[trespassing], 0, 15, colors='tab:red', linestyles=':')
    ax.plot(t[::step], kl_loss, lw=1., alpha=0.7,
               label=["updated-prior vs. dist", "prior vs. updated-prior", "prior vs. dist (sanity check)"])  # ,
                      # "prior vs. running-prior (sanity check)",
                      # "updated-prior vs. running-prior (sanity check)"])
    ax.hlines([-thres, 0, thres], 0., t[-1], color='k', linestyle='--', lw=.8)
    ax.set_ylim(-.5, 2)
    ax.set_xlim(100, 260)
    ax.legend()

    fig.tight_layout()
    plt.show()


def main():
    seed = np.random.randint(0, 1000)
    # seed = 42
    print(f"Seed: {seed}")
    keras.utils.set_random_seed(seed)

    # train()
    # compare()
    implement()


if __name__ == '__main__':
    print("Using backend " + keras.backend.backend())
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    model_location = 'tmp/autoencoder_leakyrelu.keras'

    if torch.cuda.is_available():
        print("Using CUDA")
        with torch.cuda.device(0):
            main()
    else:
        main()
