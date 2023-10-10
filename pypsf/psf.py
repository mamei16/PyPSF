import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError

from pypsf.utils import reverse_diff, psf_warn
from pypsf.hyperparameter_search import optimum_k, optimum_w
from pypsf.predict import psf_predict


class Psf:
    """
    Based on https://pypi.org/project/PSF-Py/ and hence transitively based on
    https://cran.r-project.org/web/packages/PSF/index.html
    """
    def __init__(self, cycle_length: int, k: int or None = None, w: int or None = None,
                 suppress_warnings: bool = False, apply_diff: bool = False, diff_periods: int = 1,
                 detrend: bool = False):
        """
        A Pattern Sequence Based Forecasting model.

        Args:
            cycle_length (int):
                The cycle length c
            k (int or None): optional
                The user-defined number of desired clusters when running K-means
                on the cycles
            w (int or None): optional
                The user-defined window size
            suppress_warnings (bool):
                Suppress all warnings
            apply_diff (bool):
                Apply first order differencing to the time series before
                applying PSF
            diff_periods (bool):
                Periods to shift for calculating difference, to allow for either
                ordinary or seasonal differencing
            detrend (bool):
                Remove a linear trend from the series prior to applying PSF by
                fitting a simple linear regression model. The trend is
                subsequently re-added to the predictions.
        """
        self.apply_diff = apply_diff
        self.diff_periods = diff_periods
        self.detrend = detrend
        self.k = k
        self.w = w
        self.cycle_length = cycle_length
        self.suppress_warnings = suppress_warnings
        self.apply_diff = apply_diff
        self.min_max_scaler = MinMaxScaler()
        self.norm_data = None  # will be instantiated when calling 'fit'
        self.preds = None  # will be instantiated when calling 'predict'

    def preprocessing(self, data: ArrayLike) -> np.array:
        """
        Performs the following steps to prepare the data for the PSF algorithm:
            1. (Optional) Remove a linear trend from the data if self.detrend is
                True
            2. (Optional) Apply first order differencing to the data if
                self.apply_diff is True. Optionally, this can be seasonal
                differencing if self.diff_periods > 1
            3. Normalize the data
            4. Split the data into cycles
        Returns (np.array):
            The preprocessed data on which to run the PSF algorithm.
        """
        data = np.array(data)
        if np.isnan(data).any():
            raise ValueError('Time Series contains missing values.')
        if self.detrend:
            data_len = len(data)
            idx = np.linspace(0, data_len-1, data_len, dtype=int).reshape(-1, 1)
            self.idx = idx[-1]+1
            self.trend_mod = LinearRegression().fit(idx, data)
            data = data - self.trend_mod.predict(idx)
        if self.apply_diff:
            self.last_data_points = data[-self.diff_periods:]
            data = data[self.diff_periods:] - data[:-self.diff_periods]
        # Normalize data
        norm_data = self.min_max_scaler.fit_transform(data.reshape(-1, 1)).flatten()
        # Split data into cycles
        fit = len(data) % self.cycle_length
        if fit > 0 and not self.suppress_warnings:
            psf_warn(f"\nTime Series length {'after differencing ' if self.apply_diff else ''}"
                     f"is not a multiple of {self.cycle_length}. Cutting first {fit} values!")
        norm_data = norm_data[fit:]
        split_idxs = np.arange(self.cycle_length, len(norm_data), self.cycle_length, dtype=int)
        cycles = np.array_split(norm_data, split_idxs)
        return np.stack(cycles)

    def fit(self, data: ArrayLike, k_values=tuple(range(2, 12)), w_values=tuple(range(1, 20))) -> "Psf":
        """
        Performs a hyperparameter search for good values of 'k' (the number of
        clusters) and 'w' (the window size), unless they were already provided
        by the user in '__init__'.

        Args:
            data (ArrayLike):
                The training timeseries data that is used in the hyperparameter
                search
            k_values (tuple):
                The range of 'k' values to search
            w_values (tuple):
                The range of 'w' values to search
        Returns:
            self (Psf)
        """
        num_training_samples = len(data)
        if (num_training_samples < self.cycle_length or
                (self.apply_diff and num_training_samples - self.diff_periods < self.cycle_length)):
            raise ValueError(f"Length of training data {'after differencing ' if self.apply_diff else ''}"
                             "must at least be equal to cycle length")
        self.norm_data = self.preprocessing(data)
        if (num_cycles := len(self.norm_data)) <= 2 and not self.suppress_warnings:
            psf_warn(f"\nOnly {num_cycles} cycles remaining after preprocessing."
                          f" Only a single cluster will be formed.")
        # Find optimal number (K) of clusters (or use the value specified by the user).
        if self.k is None:
            self.k = optimum_k(self.norm_data, k_values)
        # Find optimal window size (W) (or use the value specified by the user).
        if self.w is None:
            self.w = optimum_w(self.norm_data, self.k, self.cycle_length, w_values)
        return self

    def predict(self, n_ahead: int) -> np.array:
        """
        Run the PSF algorithm to predict the next 'n_ahead' values.
        Args:
            n_ahead (int):
                The number of values to predict.

        Returns (np.array):
            A numpy array of generated predictions
        """
        if self.norm_data is None:
            raise NotFittedError("This Psf instance is not fitted yet. Call 'fit' with "
                                 "appropriate arguments before using this estimator.")
        orig_n_ahead = n_ahead
        n_ahead = int((n_ahead / self.cycle_length) + 1)
        fit = orig_n_ahead % self.cycle_length
        if fit > 0 and not self.suppress_warnings:
            psf_warn(f"\nPrediction horizon {orig_n_ahead} is not multiple of {self.cycle_length}."
                     f" Using {n_ahead * self.cycle_length} as intermediate prediction horizon!")

        # Predict the 'n_ahead' next values for the time series.
        preds = psf_predict(dataset=self.norm_data, n_ahead=self.cycle_length * n_ahead,
                            cycle_length=self.cycle_length, k=self.k, w=self.w,
                            supress_warnings=self.suppress_warnings)
        self.preds = self.postprocessing(preds, orig_n_ahead)
        return self.preds

    def postprocessing(self, preds: list[np.array], orig_n_ahead: int) -> np.array:
        """
        Performs the inverse of 'preprocessing', i.e.:
            1. (Optional) Re-add a linear trend from the data if self.detrend is
                True
            2. (Optional) Reverse first order differencing to the data if
                self.apply_diff is True. Optionally, this can be seasonal
                differencing if self.diff_periods > 1
            3. Revert the data normalization in the predictions
            4. Concatenate the predicted cycles into a flat array
        Args:
            preds (list[np.array]):
                A list of predicted cycles
            orig_n_ahead (int):
                The desired number of values to be predicted. Excess predicted
                 values will be cut off.
        Returns (np.array):
            The final predictions, ready to be used.
        """
        preds = np.concatenate(preds)[:orig_n_ahead]  # cut off surplus preds of intermediate prediction horizon
        # Denormalize predicted data.
        preds = self.min_max_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        if self.apply_diff:
            preds = reverse_diff(self.last_data_points, preds, self.diff_periods)
        if self.detrend:
            pred_len = len(preds)
            pred_idx = np.linspace(self.idx, self.idx + pred_len - 1, pred_len, dtype=int).reshape(-1, 1)
            preds += self.trend_mod.predict(pred_idx)
        return preds

    def __repr__(self):
        return f"PSF | k = {self.k}, w = {self.w}, cycle_length = {self.cycle_length}"
