import numpy as np
from sklearn.metrics import silhouette_score, mean_absolute_error

from pypsf.__kmeans_cluster import _cluster_labels
from pypsf.__psf_predict import _psf_predict


def _optimum_k(dataset, k_values):
    """
    Function to calculate optimum number of clusters
    Parameters:
                dataset :
                    The array of cycles

                k_values : tuple
                    Values of k to test with.

    Returns:
                best_k: int
                    Optimum Number of clusters for PSF.
    """
    global best_k
    dataset = np.array(dataset)

    # Find best number of clusters.
    best_s = -1
    for k in k_values:
        if 1 < k < len(dataset):
            # Using algorithm kMeans for clustering.
            k_means = _cluster_labels(dataset, k)
            clusters = k_means.labels_
            # Evaluate clustering using silhouette index.
            s = silhouette_score(dataset, clusters)
            # Store best k value so far.
            if s > best_s:
                best_s = s
                best_k = k
    return best_k


def _optimum_w(data, k, cycle, w_values):
    """
    Function to calculate optimal window size
    Parameters:
                data :
                    The array of cycles

                cycle : int
                    Periodicity of original time series.

                k : int
                    Number of clusters.

                w_values : tuple
                    Size of window.

    Returns:
                best_w: int
                    Optimum size of window for PSF.
    """
    # Step 1. Take validation set out from training.
    n_ahead = int(0.3 * len(data))
    test = data.iloc[-n_ahead:]
    training = data.iloc[:len(data) - n_ahead] 
    n = len(training)
    best_w = 0

    # Step 2. Find the window size (W) that minimizes the error.
    min_err = np.Inf
    for w in w_values:
        if 0 < w < n:
            # 2.1 Perform prediction with the current 'w' value.
            pred = _psf_predict(dataset=training, k=k, w=w, cycle=cycle, n_ahead=cycle * n_ahead, surpress_warnings=True)
            pred = np.array(pred)

            # 2.2 Evaluate error and update the minimum.
            err = mean_absolute_error(test, pred)
            if err < min_err:
                min_err = err
                best_w = w
    return best_w
    

def _optimum_w(data, k, cycle, w_values):
    """
    Function to calculate optimal window size
    Parameters:
                data :
                    The array of cycles

                cycle : int
                    Periodicity of original time series.

                k : int
                    Number of clusters.

                w_values : tuple
                    Size of window.

    Returns:
                best_w: int
                    Optimum size of window for PSF.
    """
    # Step 1. Take validation set out from training.
    n_ahead = int(0.3 * len(data))
    test = data.iloc[-n_ahead:]
    training = data.iloc[:len(data) - n_ahead] 
    n = len(training)
    best_w = 0

    # Step 2. Find the window size (W) that minimizes the error.
    min_err = np.Inf
    for w in w_values:
        if 0 < w < n:
            # 2.1 Perform prediction with the current 'w' value.
            pred = _psf_predict(dataset=training, k=k, w=w, cycle=cycle, n_ahead=cycle * n_ahead, surpress_warnings=True)
            pred = np.array(pred)

            # 2.2 Evaluate error and update the minimum.
            err = mean_absolute_error(test, pred)
            if err < min_err:
                min_err = err
                best_w = w
    
    pred = _psf_predict(dataset=training, k=k, w=best_w, cycle=cycle, n_ahead=cycle * n_ahead, surpress_warnings=True)
    pred = np.array(pred)            
    
    return best_w
