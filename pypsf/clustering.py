import numpy as np
from sklearn.cluster import KMeans


def run_clustering(cycles: list[np.array], n_clusters: int) -> KMeans:
    """
    Apply K-means clustering to the provided list of cycles.
    Args:
        cycles (list[np.array]):
            The cycles to cluster
        n_clusters (int):
            Number of clusters (k) to form.

    Returns:
        kmeans (KMeans):
            The fitted K-means clustering object
    """
    cycles = np.array(cycles)
    return KMeans(n_clusters=n_clusters, init='random', n_init="auto", random_state=3683475120).fit(cycles)
