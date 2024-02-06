import numpy as np
from matplotlib import pyplot as plt
from keras import ops
import torch
from torch.distributions import MultivariateNormal, Distribution, MixtureSameFamily, Categorical


class GaussianDensityEstimation(MixtureSameFamily):
    def __init__(self, x, bandwidth=10, n_points=100):
        step = x.shape[0] // n_points
        furthest = ops.max(torch.linalg.norm(x, axis=-1), axis=0)
        sort_idx = ops.argsort(torch.linalg.norm(x - furthest, axis=-1), axis=0)
        x = x[sort_idx]
        used_points = ops.stack([ops.mean(x[i:i + step], axis=0) for i in range(0, x.shape[0], step)])
        covs = ops.stack([torch.cov(x[i:i + step].T) for i in range(0, x.shape[0], step)])
        covs += ops.full((covs.shape[0], 3), bandwidth).diag_embed()
        components = MultivariateNormal(used_points, covs)
        mix = Categorical(ops.ones(used_points.shape[0]) / len(used_points))
        super(GaussianDensityEstimation, self).__init__(mix, components)


def get_distribution(x, encoder):
    embeddings = encoder(x)
    # mean = ops.mean(embeddings, axis=0)
    # cov = torch.cov(embeddings.T)
    # distribution = MultivariateNormal(mean, cov)
    # distribution = TorchKDEWrapper(embeddings, bandwidth=0.2)
    distribution = GaussianDensityEstimation(embeddings)

    return embeddings, distribution


def visualize_distribution(embeddings, distribution, dim='3d'):
    embeddings = embeddings[::30]
    data = ops.convert_to_numpy(embeddings)
    num_dimensions = data.shape[1]

    if dim == '3d':
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot for the data points
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.2, depthshade=True)

        samples = distribution.sample((1000,))
        probs = ops.convert_to_numpy(ops.exp(distribution.log_prob(samples)))
        samples = ops.convert_to_numpy(samples)

        # Normalize probabilities for color mapping
        min_prob, max_prob = probs.min(), probs.max()
        normalized_probs = (probs - min_prob) / (max_prob - min_prob)
        alpha_values = normalized_probs * .5 + .25

        # Scatter plot with color gradient
        scatter = ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=normalized_probs, cmap='viridis',
                             alpha=alpha_values, depthshade=True)

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
