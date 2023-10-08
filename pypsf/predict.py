import warnings
from collections import Counter

import numpy as np

from pypsf.clustering import run_clustering
from pypsf.neighbors import neighbor_indices


def psf_predict(dataset: np.array, n_ahead: int, cycle_length: int, k, w,
                supress_warnings=False, pca=None) -> np.array:
    """
    Run the PSF algorithm on the provided data to generate the desired number
    of predictions.

    Args:
        dataset (np.array):
            The training data on which to run the algorithm
        n_ahead (int):
            The desired number of predictions
        cycle_length (int):
            The cycle length or periodicity of the data
        k (int):
            The number of clusters the form when clustering the cycles
        w (int):
            The desired initial window size. If at some point during an
            iteration of the PSF algorithm no matching neighbors can be
            found in the cluster label sequence, this value will be decremented
            and another neighbor search will be performed, until the values
            reaches 0. At this point, the fallback mechanism goes into effect:
            Use the centroid of the largest cluster as the prediction. The
            window size will be reset in the next iteration of the algorithm.
        supress_warnings (bool):
            If True, do not generate a warning each time the fallback prediction
            mechanism is ued.
    Returns (np.array):
        The predicted cycles
    """

    clusters = None
    temp = list(np.array(dataset))
    n_ahead_cycles = int(n_ahead / cycle_length)  # Assuming n_ahead_cycle >= 1
    n = 1
    cw = w

    while n <= n_ahead_cycles:
        # Step 1. Dataset clustering (if window size was not reduced).
        if cw == w:
            k_means = run_clustering(temp, k, pca)
            clusters = k_means.labels_
        # Step 2. Find the pattern in training data (neighbors).
        neighbor_index = neighbor_indices(clusters, cw)
        # Step 3. Check for patterns found.
        if not neighbor_index:
            # If no patterns were found, decrease the window size.
            cw = cw - 1
            if cw == 0:
                # If no window size produces neighbors, use fallback
                label_counts = Counter(clusters)
                biggest_cluster, _ = max(label_counts.items(), key=lambda x: x[1])
                centroid = k_means.cluster_centers_[biggest_cluster]
                if pca is not None:
                    temp.append(pca.inverse_transform(centroid))
                else:
                    temp.append(centroid)   # Use centroid of biggest cluster as prediction
                # Set the current window to its initial value and make next prediction.
                cw = w
                n = n + 1

                warnings.formatwarning = format_warning
                warn_str = "No pattern was found in training for any window size.\n" \
                           "Using centroid of largest cluster as the prediction!"
                if not supress_warnings:
                    warnings.warn(warn_str)
        else:
            # If some patterns were found.
            # Step 4. Compute the average of the neighbors.
            pred = np.mean([temp[x] for x in neighbor_index], axis=0)

            # Step 5. Append prediction to produce the following ones.
            temp.append(pred)

            # Step 6. Set the current window to its initial value and take next horizon.
            cw = w
            n = n + 1
    return np.array(temp[-n_ahead_cycles:])


def format_warning(message, category, filename, lineno, line=''):
    return f"{filename}:{lineno}:{category.__name__}:{message}\n"