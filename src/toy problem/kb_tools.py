from matplotlib import pyplot as plt
from keras import ops
import torch
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical
from torch.distributions.kl import register_kl


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
            used_points, labels = k_means_cluster(x, n_points, 5, use_clusters_from_x=True)
            covs = ops.stack([torch.cov(x[labels == idx].T) if x[labels == idx].shape[0] > 1 else ops.eye(3) * 0.1
                              for idx in range(used_points.shape[0])])
            covs += ops.full((covs.shape[0], 3), bandwidth).diag_embed()

            # pca = PCA(x, n_components=2)
            # proj = ops.matmul(x, pca.components)
            # sort_idx = ops.argsort(proj[:, 0])
            # x = x[sort_idx]
            # used_points = ops.stack([ops.mean(x[i:i + step], axis=0) for i in range(0, x.shape[0] - 1, step // 2)])
            # covs = ops.stack([torch.cov(x[i:i + step].T) for i in range(0, x.shape[0] - 1, step // 2)])
            # covs += ops.full((covs.shape[0], 3), bandwidth).diag_embed()

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
