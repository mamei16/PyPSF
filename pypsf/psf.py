import warnings

import numpy as np

from pypsf.hyperparameter_search import optimum_k, optimum_w
from pypsf.predict import psf_predict, format_warning
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


class Psf:
    """
    Based on https://pypi.org/project/PSF-Py/ and hence transitively based on
    https://cran.r-project.org/web/packages/PSF/index.html
    """
    def __init__(self, cycle, k=None, w=None, suppress_warnings=False, apply_diff=False, diff_periods=1,
                 detrend=False):
        """
        :param cycle: int
            The cycle length c
        :param k: optional
            The user-defined number of desired clusters when running K-means on the cycles
        :param w: optional
            The user-defined window size
        :param suppress_warnings: optional
            Suppress all warnings
        :param apply_diff: optional
            Apply first order differencing to the time series before applying PSF
        :param diff_periods: optional, default=1
            Periods to shift for calculating difference, to allow for either ordinary or seasonal differencing
        :param detrend:
            Remove a linear trend from the series prior to applying PSF by fitting a simple linear regression model.
            The trend is subsequently re-added to the predictions.
        """
        self.apply_diff = apply_diff
        self.diff_periods = diff_periods
        self.detrend = detrend
        self.k = k
        self.w = w
        self.cycle = cycle
        self.suppress_warnings = suppress_warnings
        self.apply_diff = apply_diff
        self.min_max_scaler = MinMaxScaler()
        self.norm_data = None  # will be instantiated when calling 'fit'
        self.dmin = None  # maximum of data. Will be instantiated when calling 'fit'
        self.dmax = None  # minimum of data. Will be instantiated when calling 'fit'

    def preprocessing(self, data) -> np.array:
        """
        Performs the following steps to prepare the data for the PSF algorithm:
            1. (Optional) Remove a linear trend from the data if self.detrend is
                True
            2. (Optional) Apply first order differencing to the data if
                self.apply_diff is True. Optionally, this can be seasonal
                differencing if self.diff_periods > 1
            3. Normalize the data
            4. Split the data into cycles
        Returns (pd.DataFrame):
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
        fit = len(data) % self.cycle
        if fit > 0 and not self.suppress_warnings:
            warn_str = f"\nTime Series length is not multiple of {self.cycle}. Cutting first {fit} values!"
            warnings.warn(warn_str)
        norm_data = norm_data[fit:]
        split_idxs = np.arange(self.cycle, len(norm_data), self.cycle, dtype=int)
        cycles = np.array_split(norm_data, split_idxs)
        return np.stack(cycles)

    def fit(self, data, k_values=tuple(range(3, 12)), w_values=tuple(range(1, 20))) -> "Psf":
        """
        Performs a hyperparameter search for good values of 'k' (the number of
        clusters) and 'w' (the window size), unless they were already provided
        by the user in '__init__'.

        Args:
            k_values (tuple):
                The range of 'k' values to search
            w_values (tuple):
                The range of 'w' values to search
        Returns (None):
        """
        self.norm_data = self.preprocessing(data)
        # Find optimal number (K) of clusters (or use the value specified by the user).
        if self.k is None:
            self.k = optimum_k(self.norm_data, k_values)
        # Find optimal window size (W) (or use the value specified by the user).
        if self.w is None:
            self.w = optimum_w(self.norm_data, self.k, self.cycle, w_values)
        return self

    def predict(self, n_ahead):
        orig_n_ahead = n_ahead
        n_ahead = int((n_ahead / self.cycle) + 1)
        fit = orig_n_ahead % self.cycle
        if fit > 0 and not self.suppress_warnings:
            warnings.formatwarning = format_warning
            warn_str = f"\nPrediction horizon {orig_n_ahead} is not multiple of {self.cycle}." \
                       f" Using {n_ahead * self.cycle} as intermediate prediction horizon!"
            warnings.warn(warn_str)

        # Step 7. Predict the 'n_ahead' next values for the time series.
        preds = psf_predict(dataset=self.norm_data, n_ahead=self.cycle * n_ahead, cycle=self.cycle, k=self.k, w=self.w,
                            surpress_warnings=self.suppress_warnings)
        return self.postprocessing(preds, orig_n_ahead)

    def postprocessing(self, preds, orig_n_ahead: int) -> np.array:
        preds = np.concatenate(preds)[:orig_n_ahead]  # cut off surplus preds of intermediate prediction horizon
        # Step 8. Denormalize predicted data.
        self.preds = self.min_max_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        if self.apply_diff:
            self.preds = reverse_diff(self.last_data_points, self.preds, self.diff_periods)
        if self.detrend:
            pred_len = len(self.preds)
            pred_idx = np.linspace(self.idx, self.idx + pred_len - 1, pred_len, dtype=int).reshape(-1, 1)
            self.preds += self.trend_mod.predict(pred_idx)
        return self.preds


    def __repr__(self):
        return f"PSF | k = {self.k}, w = {self.w}, cycle = {self.cycle}"


def reverse_diff(orig_vals, diffed, periods):
    length = len(diffed)
    res = np.zeros(length)
    for i in range(periods):
        indices = np.arange(i, length, periods, dtype=int)
        undiffed = orig_vals[i] + np.cumsum(diffed[i::periods])
        res[indices] = undiffed
    return res