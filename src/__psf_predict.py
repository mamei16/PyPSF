import warnings
import numpy as np
from collections import Counter
from __kmeans_cluster import _cluster_labels
from __neighbor import neighbor_indices


def _psf_predict(dataset, n_ahead, cycle, k, w, surpress_warnings=False):
    """
    Parameters :
                dataset :
                    The data to compute predictions from.
                n_ahead : int
                    The number of values to predict.
                cycle : int
                    Periodicity of time series.
                k : int
                    Number of clusters.
                w : int
                    Size of window.
    Returns :
                temp[-n_ahead_cycles:] : list
                    List of predicted cycles.
    """
    clusters = None
    temp = list(np.array(dataset))
    n_ahead_cycles = int(n_ahead / cycle)  # Assuming n_ahead_cycle >= 1
    n = 1
    cw = w

    while n <= n_ahead_cycles:
        # Step 1. Dataset clustering (if window size was not reduced).
        if cw == w:
            k_means = _cluster_labels(temp, k)
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
                temp.append(centroid)   # Use centroid of biggest cluster as prediction
                # Set the current window to its initial value and make next prediction.
                cw = w
                n = n + 1

                warnings.formatwarning = format_warning
                warn_str = "No pattern was found in training for any window size.\n" \
                           "Using centroid of largest cluster as the prediction!"
                if not surpress_warnings:
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
    return temp[-n_ahead_cycles:]


def format_warning(message, category, filename, lineno, line=''):
    return f"{filename}:{lineno}:{category.__name__}:{message}\n"