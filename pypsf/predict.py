from collections import Counter

import numpy as np

from pypsf.clustering import run_clustering
from pypsf.neighbors import neighbor_indices
from pypsf.utils import psf_warn


def psf_predict(dataset: np.array, n_ahead: int, cycle_length: int, k, w,
                supress_warnings=False) -> np.array:
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
    temp = list(np.array(dataset))
    n_ahead_cycles = int(n_ahead / cycle_length)  # Assuming n_ahead_cycle >= 1

    for _ in range(n_ahead_cycles):
        # Step 1. Apply clustering to cycles
        k_means = run_clustering(temp, k)
        cluster_labels = k_means.labels_
        # Step 2. Find patterns matching the last 'w' cluster labels in training data (neighbors).
        # If no matches were found, decrease the window size.
        neighbors = None
        for current_w in range(w, 0, -1):
            # Step 3. Check for matching patterns found.
            neighbors = neighbor_indices(cluster_labels, current_w)
            if neighbors:
                break
        if not neighbors:
            # If no window size produces neighbors, use fallback
            label_counts = Counter(cluster_labels)
            biggest_cluster, _ = max(label_counts.items(), key=lambda x: x[1])
            centroid = k_means.cluster_centers_[biggest_cluster]
            temp.append(centroid)  # Use centroid of biggest cluster as prediction
            # Set the current window to its initial value and make next prediction.
            if not supress_warnings:
                psf_warn("No pattern was found in training for any window size.\n"
                         "Using centroid of largest cluster as the prediction!")
        else:
            # If some patterns were found.
            # Step 4. Compute the average of the neighbors.
            pred = np.mean([temp[x] for x in neighbors], axis=0)
            # Step 5. Append prediction to produce the following ones.
            temp.append(pred)

    return np.array(temp[-n_ahead_cycles:])
