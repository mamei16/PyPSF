import numpy as np
from sklearn.cluster import KMeans


def _cluster_labels(dataset, n_clusters):
    """
    Parameters:
                dataset : pandas.series
                    The data to perform k-means clustering on.

                n_clusters : int
                     Number of clusters (k) to form.

    Returns:
                cluster_labels : array
                    Index of the cluster each sample belongs to.
    """
    dataset = np.array(dataset)
    kmeans = KMeans(n_clusters=n_clusters, init='random', n_init="auto", random_state=3683475120).fit(dataset)
    return kmeans
