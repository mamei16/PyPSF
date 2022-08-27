import warnings

import numpy as np
import pandas as pd

from __optimum import _optimum_k, _optimum_w
from __psf_predict import _psf_predict, format_warning
from sklearn.linear_model import LinearRegression


class Psf:
    """
    Based on https://pypi.org/project/PSF-Py/ and hence transitively based on
    https://cran.r-project.org/web/packages/PSF/index.html
    """
    def __init__(self, data, cycle, k=None, w=None, suppress_warnings=False, apply_diff=False, diff_periods=1,
                 detrend=False):
        """
        :param data:
            The input time series
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
        self.data = data
        if detrend:
            data_len = len(data)
            idx = np.linspace(0, data_len-1, data_len, dtype=int).reshape(-1, 1)
            self.idx = idx[-1]+1
            self.trend_mod = LinearRegression().fit(idx, data)
            self.data = data - self.trend_mod.predict(idx)
        if apply_diff:
            self.last_data_points = self.data[-diff_periods:]
            data = np.array(self.data)
            self.data = data[diff_periods:] - data[:-diff_periods]
        self.apply_diff = apply_diff
        self.diff_periods = diff_periods
        self.detrend = detrend
        self.k = k
        self.w = w
        self.cycle = cycle
        self.suppress_warnings = suppress_warnings
        self.apply_diff = apply_diff
        fit = len(self.data) % self.cycle
        if fit > 0 and not suppress_warnings:
            warn_str = f"\nTime Series length is not multiple of {cycle}. Cutting first {fit} values!"
            warnings.warn(warn_str)
        self.data = self.data[fit:]
        split_idxs = np.arange(self.cycle, len(self.data), self.cycle, dtype=int)
        cycles = np.array_split(self.data, split_idxs)
        self.data = pd.DataFrame(np.stack(cycles))

    def predict(self, n_ahead, k_values=tuple(range(3, 12)), w_values=tuple(range(1, 20))):
        # Check integrity of data (both its size and n.ahead must be multiple of cycle).
        if self.data.isna().any().any():
            raise ValueError('\nTime Series contains NA.')

        orig_n_ahead = n_ahead
        n_ahead = int((n_ahead / self.cycle) + 1)
        fit = orig_n_ahead % self.cycle
        if fit > 0 and not self.suppress_warnings:
            warnings.formatwarning = format_warning
            warn_str = f"\nPrediction horizon {orig_n_ahead} is not multiple of {self.cycle}." \
                       f" Using {n_ahead*self.cycle} as intermediate prediction horizon!"
            warnings.warn(warn_str)

        #  Normalize data.
        dmin = self.data.min().min()
        dmax = self.data.max().max()
        if dmin < 0:
            self.data += np.abs(dmin)
        norm_data = (self.data - dmin) / (dmax - dmin)
        # Find optimal number (K) of clusters (or use the value specified by the user).
        if self.k is None:
            self.k = _optimum_k(norm_data, k_values)
        # Find optimal window size (W) (or use the value specified by the user).
        if self.w is None:
            self.w = _optimum_w(norm_data, self.k, self.cycle, w_values)

        # Step 7. Predict the 'n_ahead' next values for the time series.
        preds = _psf_predict(dataset=norm_data, n_ahead=self.cycle * n_ahead, cycle=self.cycle, k=self.k, w=self.w,
                             surpress_warnings=self.suppress_warnings)
        # Step 8. Denormalize predicted data.
        preds = np.array(preds) * (dmax - dmin) + dmin
        if dmin < 0:
            preds -= np.abs(dmin)
        self.preds = np.concatenate(preds)[:orig_n_ahead]  # cut off surplus preds of intermediate prediction horizon
        if self.apply_diff:
            self.preds = reverse_diff(self.last_data_points, self.preds, self.diff_periods)
        if self.detrend:
            pred_len = len(self.preds)
            pred_idx = np.linspace(self.idx, self.idx+pred_len-1, pred_len, dtype=int).reshape(-1, 1)
            self.preds += self.trend_mod.predict(pred_idx)
        return self.preds

    def model(self):
        return self

    def model_print(self):
        print('\nOriginal time-series : \n', self.data)
        print('\nPredicted Values : \n', self.preds)
        print('\nk = ', self.k)
        print('\nw = ', self.w)
        print('\ncycle = ', self.cycle)


def reverse_diff(orig_vals, diffed, periods):
    length = len(diffed)
    res = np.zeros(length)
    for i in range(periods):
        indices = np.arange(i, length, periods, dtype=int)
        undiffed = orig_vals[i] + np.cumsum(diffed[i::periods])
        res[indices] = undiffed
    return res