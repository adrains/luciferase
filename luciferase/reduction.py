"""Functions associated with CRIRES+ data reduction or diagnostics.
"""
import numpy as np
from astropy import modeling

def fit_seeing_to_slit_func(
    slit_func,
    arcsec_per_px=0.059,
    resampling_fac=12,
    slit_height_arcsec=10,):
    """Function to fit a 1D Gaussian to CRIRES+ slit func data.

    Parameters
    ----------
    slit_func: numpy array
        1D array of the CRIRES+ slit func for the extracted region.

    arcsec_per_px: float, default: 0.059
        Pixel scale in arcsec/pixel.

    resampling_fac: float, default: 12
        Oversampling factor for the slit extraction, e.g. if 30 pixels have 
        been extracted with an oversampling factor of 12, then the slit_func
        array will have (30+1) x 12 pixels.
    
    slit_height_arcsec: float, default: 10
        Height of the **full** slit. Not currently used.

    Returns
    -------
    fwhm: float
        The full width half max of the fitted Gaussian.
    """
    # Initialise
    fitter = modeling.fitting.LevMarLSQFitter()

    gaussian_model = modeling.models.Gaussian1D()

    # Setup x array in units of arcsec
    n_px = len(slit_func)
    xx = (np.arange(n_px) - n_px/2) * arcsec_per_px / (resampling_fac-1)

    # Do fit
    fitted_model = fitter(gaussian_model, xx, slit_func)
    
    # Calculate FWHM of Gaussian
    # (per https://en.wikipedia.org/wiki/Full_width_at_half_maximum)
    fwhm = 2*np.sqrt(2*np.log(2))*fitted_model.stddev.value

    return fwhm