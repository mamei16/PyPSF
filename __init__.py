from .psf import Psf


'''
Parameters : 
            data :
                The time series.
                
            cycle : int
                Periodicity of time series.
                
            k : int, optional, default: None
                The number of clusters to form as well as the number of centroids to generate using K-Means clustering. If not provided, it is calculated internally
                
            w : int, optional, default: None
                The size of window. If not provided, it is calculated internally
                
Attributes: 
            preds:
                Array of predictions.

            predict : 
                Calculates prediction using PSF.
                
            model : 
                Returns the PSF model.
                
            model_print :
                Prints the model.
'''
__all__ = ['Psf']
