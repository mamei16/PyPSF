import numpy as np
import sklearn
from sklearn.cluster import KMeans
from typing import List


def run_clustering(cycles: List[np.array] or np.array, n_clusters: int) -> KMeans:
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
    return KMeans(n_clusters=n_clusters, init='random',
                  n_init="auto" if sklearn.__version__ >= "1.2" else 10, random_state=3683475120).fit(np.array(cycles))
