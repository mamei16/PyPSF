# PyPSF
This project provides a python implementation of the Pattern Sequence Based Forecasting (PSF) algorithm. For a detailed description of the PSF algorithm and some of the practical issues I encountered when using it, see [this PDF file](https://github.com/mamei16/PyPSF/blob/9b6d395cf2b8288937e7b4bca7ee5752e2e1c435/psf_description.pdf).

## Installation

`pip install pypsf`

### Dependencies
- scikit-learn
- numpy

## Example Usage

```python
import numpy as np
import matplotlib.pyplot as plt
from pypsf import Psf

plt.style.use("dark_background")

t_series = np.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115,
                     126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140, 145, 150,
                     178, 163, 172, 178, 199, 199, 184, 162, 146, 166, 171, 180, 193,
                     181, 183, 218, 230, 242, 209, 191, 172, 194, 196, 196, 236, 235,
                     229, 243, 264, 272, 237, 211, 180, 201, 204, 188, 235, 227, 234,
                     264, 302, 293, 259, 229, 203, 229, 242, 233, 267, 269, 270, 315,
                     364, 347, 312, 274, 237, 278, 284, 277, 317, 313, 318, 374, 413,
                     405, 355, 306, 271, 306, 315, 301, 356, 348, 355, 422, 465, 467,
                     404, 347, 305, 336, 340, 318, 362, 348, 363, 435, 491, 505, 404,
                     359, 310, 337, 360, 342, 406, 396, 420, 472, 548, 559, 463, 407,
                     362, 405, 417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390,
                     432])
train = t_series[:-28]
test = t_series[-28:]

psf = Psf(cycle_length=12, apply_diff=True, diff_periods=12)
psf.fit(train)

pred = psf.predict(len(test))

fig, ax = plt.subplots()
x_train = np.array(range(len(train)))
x_test_pred = np.array(range(len(test))) + x_train[-1]
ax.plot(x_train, train, c="lightblue")
ax.plot(x_test_pred, test, c="lightgreen")
ax.plot(x_test_pred, pred, c="tab:orange")
plt.legend(["Training", "Test", "Prediction"])
plt.tight_layout()
plt.show()
```
![psf_prediction_plot](https://github.com/mamei16/PyPSF/assets/25900898/111befda-6318-4ef5-97a3-71936d980d09)

### Parameters

**class Psf**
- cycle_length: int  
    The cycle length c
- k: int (optional), default None    
    The user-defined number of desired clusters when running K-means on the cycles
- w: int (optional), default None    
    The user-defined window size
- suppress_warnings: bool (optional), default False  
    Suppress all warnings
- apply_diff: bool (optional), default False    
    Apply first order differencing to the time series before applying PSF
- diff_periods: int (optional), default 1  
    Periods to shift for calculating difference, to allow for either ordinary or seasonal differencing. Ignore if apply_diff=False
- detrend: bool (optional), default False  
    Remove a linear trend from the series prior to applying PSF by fitting a simple linear regression model.
    The trend is subsequently re-added to the predictions.

**Psf.fit**
- data:   
    The input time series
- k_values: iterable[int] (optional), default tuple(range(3, 12))  
  The set of candidate values of *k* to test when finding the "best" *k* number of clusters based on the training data
- w_values: iterable[int] (optional), default tuple(range(1, 20))  
  The set of candidate values of *w* to test when finding the "best" window size *w* based on the training data

**Psf.predict**
- n_ahead: int  
  The number of values to predict
 
    
