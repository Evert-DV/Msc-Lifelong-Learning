from matplotlib import pyplot as plt
from keras import ops
import torch
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical
from torch.distributions.kl import register_kl


class GaussianDensityEstimation(MixtureSameFamily):
    def __init__(self, x=None, mix=None, components=None, bandwidth=1., n_points=100):
        if components is not None and mix is not None:
            # Initialize using precomputed components and mix
            super(GaussianDensityEstimation, self).__init__(mix, components)
        else:
            assert x is not None, "Either x, or components and mix must be provided"
            if n_points > x.shape[0]:
                n_points = x.shape[0]
            step = x.shape[0] // (n_points // 2)
            furthest = ops.max(torch.linalg.norm(x, axis=-1), axis=0)
            sort_idx = ops.argsort(torch.linalg.norm(x - furthest, axis=-1), axis=0)
            x = x[sort_idx]
            used_points = ops.stack([ops.mean(x[i:i + step], axis=0) for i in range(0, x.shape[0] - 1, step // 2)])
            covs = ops.stack([torch.cov(x[i:i + step].T) for i in range(0, x.shape[0] - 1, step // 2)])
            covs += ops.full((covs.shape[0], 3), bandwidth).diag_embed()
            components = MultivariateNormal(used_points, covs)
            mix = Categorical(ops.ones(used_points.shape[0]) / len(used_points))
            super(GaussianDensityEstimation, self).__init__(mix, components)

        self.bw = bandwidth
        self.n_points = n_points

    def update(self, new_data, weight=0.1):
        factor = int(1 / weight - 1)
        n_samples = factor * len(new_data)
        sampled_data = self.sample(torch.Size((n_samples,)))
        combined_data = ops.concatenate([new_data, sampled_data], axis=0)
        self.__init__(combined_data, bandwidth=self.bw, n_points=self.n_points)

    def copy(self):
        components = self.component_distribution
        mix = self.mixture_distribution
        new_instance = GaussianDensityEstimation(mix=mix, components=components, bandwidth=self.bw,
                                                 n_points=self.n_points)

        return new_instance


def get_distribution(x, encoder, bandwidth=1., n_points=100):
    embeddings = encoder(x)
    distribution = GaussianDensityEstimation(embeddings, bandwidth=bandwidth, n_points=n_points)

    return embeddings, distribution


@register_kl(GaussianDensityEstimation, GaussianDensityEstimation)
def kl_divergence(p, q, n_samples=1000):
    samples = p.sample((n_samples,))

    log_probs_p = p.log_prob(samples)
    log_probs_q = q.log_prob(samples)

    probs_p = ops.exp(log_probs_p)

    kl_div = ops.sum(probs_p * (log_probs_p - log_probs_q)) / ops.sum(probs_p)

    return kl_div


def visualize_distribution(distribution, embeddings=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    if embeddings is not None:
        embeddings = embeddings[::30]
        data = ops.convert_to_numpy(embeddings)
        # Scatter plot for the data points
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker='x', c='tab:red', alpha=0.2, depthshade=True)

    samples = distribution.sample((1000,))
    probs = ops.convert_to_numpy(ops.exp(distribution.log_prob(samples)))
    samples = ops.convert_to_numpy(samples)

    # Normalize probabilities for color mapping
    min_prob, max_prob = probs.min(), probs.max()
    normalized_probs = (probs - min_prob) / (max_prob - min_prob)
    alpha_values = normalized_probs * 0.5 + 0.1

    # Scatter plot with color gradient
    scatter = ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=normalized_probs, cmap='viridis',
                         alpha=alpha_values, depthshade=True)

    # Colorbar to show the mapping from color to probability
    fig.colorbar(scatter, ax=ax)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    fig.tight_layout()
    plt.show()


def ewma(data, prev_avg, rho=0.8):
    return rho * prev_avg + (1 - rho) * ops.mean(data)
