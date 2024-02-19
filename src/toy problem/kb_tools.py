import os

import numpy as np

os.environ["KERAS_BACKEND"] = "torch"
from matplotlib import pyplot as plt
import keras
from keras import ops, layers
import torch
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical
from torch.distributions.kl import register_kl, kl_divergence


def sample(z_mean, z_log_var):
    batch = ops.shape(z_mean)[0]
    dim = ops.shape(z_mean)[1]
    epsilon = keras.random.normal((batch, dim))

    # ensure positive variance
    z_var = ops.exp(z_log_var)

    # Cholesky decomposition
    cholesky = ops.zeros((batch, dim, dim))
    indices = torch.tril_indices(dim, dim)
    cholesky[:, indices[0], indices[1]] = z_var

    # Reparameterization trick
    sample = z_mean + ops.matmul(cholesky, epsilon[..., None])[..., 0]

    return sample, ops.matmul(cholesky, ops.transpose(cholesky, axes=(0, 2, 1)))


class VariationalAutoEncoder(keras.Model):
    def __init__(self, input_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # define encoder
        self.input_shape = input_shape
        inputs = layers.Input(shape=(input_shape,))
        x = layers.Dense(32, activation='softplus')(inputs)
        x = layers.Dense(16, activation='softsign')(x)
        encoded = layers.Dense(9)(x)
        z_mean = layers.Lambda(lambda z: z[..., :3])(encoded)
        z_log_var = layers.Lambda(lambda z: z[..., 3:])(encoded)
        self.encoder = keras.Model(inputs, [z_mean, z_log_var])

        # define decoder
        latent_inputs = layers.Input(shape=(3,))
        x = layers.Dense(16, activation='softsign')(latent_inputs)
        x = layers.Dense(32, activation='softplus')(x)
        outputs = layers.Dense(input_shape)(x)
        self.decoder = keras.Model(latent_inputs, outputs)

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z, covariance = sample(z_mean, z_log_var)
        reconstructed = self.decoder(z)

        # Add KL divergence regularization loss.
        kl_loss = 0.5 * (ops.trace(covariance, axis1=-2, axis2=-1) + ops.sum(z_mean ** 2, axis=-1) - 3 - ops.log(
            torch.linalg.det(covariance)))
        self.add_loss(ops.mean(kl_loss))

        return reconstructed

    def get_config(self):
        return {'input_shape': self.input_shape}


class PCA:
    def __init__(self, data, n_components=2):
        mean = ops.mean(data, axis=0)
        std = ops.std(data, axis=0)
        standardized_data = (data - mean) / std
        cov = torch.cov(standardized_data.T)
        eig_vals, eig_vecs = torch.linalg.eigh(cov)
        idx = torch.argsort(eig_vals, descending=True)
        eig_vecs = eig_vecs[:, idx]
        self.components = eig_vecs[:, :n_components]
        self.pca_data = torch.matmul(standardized_data, self.components)


class GaussianDensityEstimation(MixtureSameFamily):
    def __init__(self, x=None, mix=None, components=None, bandwidth=1., n_points=100, use_pca_weights=True):
        if components is not None and mix is not None:
            # Initialize using precomputed components and mix
            if components.loc.shape[0] > n_points:
                weights = mix.probs
                top_n_points = torch.argsort(weights, descending=True)[:n_points]
                loc = components.loc[top_n_points]
                covariance_matrix = components.covariance_matrix[top_n_points]
                components = MultivariateNormal(loc, covariance_matrix)
                probs = mix.probs[top_n_points]
                mix = Categorical(probs)
            super(GaussianDensityEstimation, self).__init__(mix, components)
        else:
            assert x is not None, "Either x, or components and mix must be provided"
            if x.shape[0] < n_points * 10:
                n_points = x.shape[0] // 10
            # spread the point selection
            used_points, labels = k_means_cluster(x, n_points, 10, use_clusters_from_x=True)
            covs = ops.stack([torch.cov(x[labels == idx].T) if x[labels == idx].shape[0] > 1 else ops.eye(3) * 0.1
                              for idx in range(used_points.shape[0])])
            covs += ops.full((covs.shape[0], 3), bandwidth).diag_embed()

            components = MultivariateNormal(used_points, covs)
            pca_x = PCA(used_points, n_components=2).pca_data
            weights = torch.linalg.norm(pca_x, dim=-1)
            if not use_pca_weights:
                weights = ops.ones(len(used_points))
            mix = Categorical(weights)
            super(GaussianDensityEstimation, self).__init__(mix, components)

        self.bw = bandwidth
        self.n_points = n_points

    def update(self, new_data, weight=0.1):
        assert 0. <= weight <= 1., "Weight must be between 0 and 1"
        new_mix = GaussianDensityEstimation(new_data, bandwidth=self.bw, n_points=self.n_points,
                                            use_pca_weights=True)
        mix_locs = new_mix.component_distribution.loc
        mix_covs = new_mix.component_distribution.covariance_matrix
        mix_weights = new_mix.mixture_distribution.probs
        loc = ops.concatenate([self.component_distribution.loc, mix_locs], axis=0)
        covariance_matrix = ops.concatenate(
            [self.component_distribution.covariance_matrix, mix_covs], axis=0)
        probs = ops.concatenate(
            [(1 - weight) * self.mixture_distribution.probs, weight * mix_weights], axis=0)

        self.__init__(mix=Categorical(probs), components=MultivariateNormal(loc, covariance_matrix), bandwidth=self.bw,
                      n_points=self.n_points)

    def copy(self):
        components = self.component_distribution
        mix = self.mixture_distribution
        new_instance = GaussianDensityEstimation(mix=mix, components=components, bandwidth=self.bw,
                                                 n_points=self.n_points)

        return new_instance


def k_means_cluster(x, k, iters=10, use_clusters_from_x=False):
    pca_x = PCA(x, n_components=2).pca_data
    weights = torch.linalg.norm(pca_x, dim=-1)
    weights = ops.sqrt(weights) + ops.mean(weights)
    # weights = ops.ones(k)
    indices = torch.multinomial(weights, k, replacement=False)
    centroids = x[indices]
    for _ in range(iters):
        distances = torch.cdist(x, centroids)
        cluster_labels = ops.argmin(distances, axis=-1)
        unique_clusters = torch.unique(cluster_labels)
        if len(unique_clusters) < k:
            centroids = centroids[unique_clusters]
        if use_clusters_from_x:
            for i, c in enumerate(unique_clusters):
                cluster_members_idx = (cluster_labels == c).nonzero(as_tuple=False)[0]
                member_distances = distances[cluster_members_idx, c]
                closest_member = ops.argmin(member_distances)
                centroids[i] = x[cluster_members_idx[closest_member]]
            continue
        centroids = torch.stack([x[cluster_labels == c].mean(dim=0) for c in unique_clusters])
    return centroids, cluster_labels


def kl_div(p, q, n_samples=1000, samples=None):
    if samples is None:
        samples = ops.concatenate((p.sample((n_samples,)), q.sample((n_samples,))), axis=0)

    log_probs_p = p.log_prob(samples)
    log_probs_q = q.log_prob(samples)

    probs_p = ops.exp(log_probs_p)

    kl_div = ops.sum(probs_p * (log_probs_p - log_probs_q)) / ops.sum(probs_p)

    return kl_div, samples


@register_kl(GaussianDensityEstimation, GaussianDensityEstimation)
def kl_symmetric(p, q, n_samples=1000):
    kl_p_q, samples = kl_div(p, q, n_samples)
    kl_q_p, _ = kl_div(q, p, samples=samples)

    return (kl_p_q + kl_q_p) / 2


def visualize_distribution(distribution, embeddings=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    if embeddings is not None:
        # embeddings = embeddings[::30]
        data = ops.convert_to_numpy(embeddings)
        # Scatter plot for the data points
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker='x', c='tab:red', alpha=0.5, depthshade=True)

    used_points = ops.convert_to_numpy(distribution.component_distribution.loc)
    ax.scatter(used_points[:, 0], used_points[:, 1], used_points[:, 2], marker='x', c='cyan', alpha=0.7, s=100)

    samples = distribution.sample((1000,))
    probs = ops.convert_to_numpy(ops.exp(distribution.log_prob(samples)))
    samples = ops.convert_to_numpy(samples)

    # Normalize probabilities for color mapping
    min_prob, max_prob = probs.min(), probs.max()
    normalized_probs = (probs - min_prob) / (max_prob - min_prob)
    # alpha_values = normalized_probs * 0.5 + 0.1

    # Scatter plot with color gradient
    scatter = ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=normalized_probs, cmap='viridis',
                         alpha=0.3)

    # Colorbar to show the mapping from color to probability
    fig.colorbar(scatter, ax=ax)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    fig.tight_layout()
    plt.show()


def ewma(data, prev_avg, rho=0.8, axis=0):
    return rho * prev_avg + (1 - rho) * ops.mean(data, axis=axis)


def get_posteriors(data, distributions, p_dist):
    p_dist = ops.array(p_dist)
    p_data_dist = ops.exp([dist.log_prob(data) for dist in distributions])
    p_dist_data = p_data_dist * p_dist[None].T
    p_dist_data /= ops.sum(p_dist_data, axis=0)

    return p_dist_data


def expand_prior_probs(prior_probs):
    prob_new_element = 1 / (len(prior_probs) + 1)
    remaining_prob = 1 - prob_new_element
    prior_probs = [prob * remaining_prob for prob in prior_probs]
    prior_probs.append(prob_new_element)

    return prior_probs


def search_dists(x, dists, prior_probs, current_dist, current_idx, thres):
    post_probs = get_posteriors(x, dists, prior_probs).mean(axis=-1)
    best_idx = ops.argmax(post_probs)
    if best_idx != current_idx:
        kl_prior_dist = kl_divergence(dists[best_idx], current_dist)
        if kl_prior_dist < .9 * thres:
            return best_idx
    return None


def get_autoencoder(input_shape, latent_dim=3, skip_connections=False):
    inputs = layers.Input(shape=(input_shape,))
    encoder = keras.Sequential([
        layers.Normalization(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(latent_dim),
    ])
    decoder = keras.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(input_shape),
    ])
    encoding = encoder(inputs)
    decoding = decoder(encoding)
    if skip_connections:
        in_slice = layers.Lambda(lambda x: x[..., :-2])(inputs)
        out_slice = layers.Lambda(lambda x: x[..., -2:])(decoding)
        decoding = layers.concatenate([in_slice, out_slice], axis=-1)

    autoencoder = keras.Model(inputs, decoding)

    return autoencoder
