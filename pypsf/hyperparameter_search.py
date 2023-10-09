import numpy as np
from sklearn.metrics import silhouette_score, mean_absolute_error

from pypsf.clustering import run_clustering
from pypsf.predict import psf_predict


def optimum_k(data: np.array, k_values: tuple[int]) -> int:
    """
    Perform a hyperparameter search using the provided values of 'k' to
    determine the number of clusters 'k' that achieve the highest silhouette
    score.
    Args:
        data (np.array):
            The training data to perform clustering on.
        k_values (tuple[int]):
            The 'k' values to include in te search

    Returns:
        best_k (int):
            The values of 'k' that achieved the highest silhouette score.

    """
    data = np.array(data)

    # Find best number of clusters.
    best_s = -1
    best_k = 1
    for k in k_values:
        if 1 < k < len(data):
            # Using algorithm kMeans for clustering.
            k_means = run_clustering(data, k)
            cluster_labels = k_means.labels_
            # Evaluate clustering using silhouette index.
            s = silhouette_score(data, cluster_labels)
            # Store best k value so far.
            if s > best_s:
                best_s = s
                best_k = k
    return best_k


def optimum_w(data: np.array, k: int, cycle_length: int, w_values: tuple[int]) -> int:
    """
    Perform a hyperparameter search using the provided values of 'w' to
    determine the window size that results in the lowest mean absolute error.
    This is done by splitting 'data' into a training and a validation set, run
    the PSF algorithm on the training set for each 'w' and calculate the
    validation error.
    Args:
        data: (np.array):
             The training data to use for finding the best 'w'
        k (int):
            The number of clusters to form
        cycle_length (int):
            The cycle length or periodicity of the data.
        w_values (tuple[int]):
             The 'w' values to include in te search

    Returns:
        best_w (int):
            The 'w' that achieved the lowest validation error
    """
    # Step 1. Take validation set out from training.
    n_ahead = int(0.3 * len(data))
    test = data[-n_ahead:]
    training = data[:len(data) - n_ahead]
    n = len(training)
    best_w = 0

    # Step 2. Find the window size (W) that minimizes the error.
    min_err = np.Inf
    for w in w_values:
        if 0 < w < n:
            # 2.1 Perform prediction with the current 'w' value.
            pred = psf_predict(dataset=training, k=k, w=w, cycle_length=cycle_length, n_ahead=cycle_length * n_ahead,
                               supress_warnings=True)
            pred = np.array(pred)

            # 2.2 Evaluate error and update the minimum.
            err = mean_absolute_error(test, pred)
            if err < min_err:
                min_err = err
                best_w = w
    return best_w
