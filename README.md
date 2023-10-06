# PyPSF
This project provides a valid python implementation of the Pattern Sequence Forecasting (PSF) algorithm. For a detailed description of the PSF algorithm and some of the practical issues I encountered when using it, see [this PDF file](https://github.com/mamei16/PyPSF/blob/9b6d395cf2b8288937e7b4bca7ee5752e2e1c435/psf_description.pdf).

## Installation

`pip install pypsf`

### Dependencies
- scikit-learn
- numpy
- pandas

## Example Usage

```
import numpy as np

from pypsf.psf import Psf


t_series = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
train = t_series[:6]
test = t_series[6:]

psf = Psf(train, cycle=3)

pred = psf.predict(len(test))

print(test) # [1 2 3]
print(pred) # [1. 2. 3.]
```

### Parameters

**class Psf**

- data:   
    The input time series
- cycle: int  
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

**Psf.predict**
- n_ahead: int  
  The number of values to predict
- k_values: iterable[int] (optional), default tuple(range(3, 12))  
  The set of candidate values of *k* to test when finding the "best" *k* number of clusters based on the training data
- w_values: iterable[int] (optional), default tuple(range(1, 20))  
  The set of candidate values of *w* to test when finding the "best" window size *w* based on the training data
 
    
