"""Module for functions related to the SYSREM algorithm for light curve 
detrending for the purpose of exoplanet transmission spectroscopy.

SYSREM:
    https://ui.adsabs.harvard.edu/abs/2005MNRAS.356.1466T/abstract
"""
import warnings
import numpy as np
from astropy.stats import sigma_clip
from tqdm import tqdm
from scipy.interpolate import interp1d
import astropy.constants as const
from numpy.polynomial.polynomial import Polynomial, polyfit

def clean_and_compute_initial_resid(
    spectra,
    e_spectra,
    mjds,
    sigma_threshold_phase=6.0,
    sigma_threshold_spectral=6.0,
    n_max_phase_bad_px=5,
    do_clip_spectral_dimension=False,
    interpolation_method="cubic",
    do_normalise=True,):
    """Function to clean a spectral data cube prior to running SYSREM. We sigma
    clip along the phase dimension (i.e. the time series of each pixel) as well
    as along the spectral dimension. 

    Note that this function expects only spectra from a single spectral
    segment (or a flattened set of many segments.
    
    Parameters
    ----------
    spectra, e_spectra: 3D float array
        Unnormalised spectra and spectroscopic uncertainties of shape 
        [n_phase, n_spec, n_px].

    mjds: 1D float array
        MJDs associated with each phase, of shape [n_phase]. We use this for
        interpolating along the phase axis.
    
    sigma_threshold_phase: float, default: 6.0
        The sigma clipping threshold for when sigma clipping along the *phase*
        dimension. Note that this used for both lower and upper bound clipping.

    sigma_threshold_spectral: float, default: 6.0
        The sigma clipping threshold for when sigma clipping along the 
        *spectral* dimension. Note that we only sigma clip on the upper bound
        so as to not remove deep spectral features.

    n_max_phase_bad_px: int, default: 5.0
        Threshold for the number of bad/clipped phases per spectral pixel, 
        above which we mask out the entire spectral pixel.

    do_clip_spectral_dimension: boolean, default: False
        Whether to do *upper* sigma clip along the spectral dimension.

    interpolation_method: str, default: "cubic"
        Default interpolation method to use with scipy.interp1d. Can be one of: 
        ['linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic',
        'cubic', 'previous', or 'next'].

    do_normalise: boolean, default: True
        Whether to normalise the cleaned + interpolated fluxes or not. This
        should be set to False if intending to use the Piskunov detrending
        algorithm.

    Returns
    -------
    resid_init: 3D float array
        Initial set of residuals to use for SYSREM composed of the cleaned,
        interpolated, and normalised fluxes, of shape [n_phase, n_spec, n_px].
    
    flux, e_flux: 2D float array
        Cleaned and interpolated fluxes + uncertainties of shape 
        [n_phase, n_spec, n_px].
    """
    # Grab shape
    (n_phase, n_spec, n_px) = spectra.shape

    # Duplicate arrays
    flux = spectra.copy()
    e_flux = e_spectra.copy()

    #--------------------------------------------------------------------------
    # Clean + interpolate along phase dimension
    #--------------------------------------------------------------------------
    # Initialise mask for sigma clipping along phase dimension. Note that here
    # we're doing both upper *and* lower bound clipping.
    sc_mask_phase = np.full_like(spectra, False)

    # Loop over all spectral segments and spectral pixels
    for spec_i in range(n_spec):
        desc = "Cleaning spectrum {}/{}".format(spec_i+1, n_spec)
        for px_i in tqdm(range(n_px), desc=desc, leave=False):
            # Sigma clip along the phase dimension to compute the bad pixel
            # mask for px_i
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bad_phase_mask = sigma_clip(
                    data=flux[:,spec_i,px_i],
                    sigma=sigma_threshold_phase,).mask
            
            # If we have more than the threshold number of bad px, mask out the
            # entire column (i.e. mask out px_i)
            if np.sum(bad_phase_mask) > n_max_phase_bad_px:
                sc_mask_phase[:,spec_i,px_i] = True

            # If the first or last phase is a bad px, then mask out the entire
            # column since we can't interpolate them.
            elif bad_phase_mask[0] or bad_phase_mask[-1]:
                sc_mask_phase[:,spec_i,px_i] = True

            # Otherwise interpolate the missing values using the times as X
            # values instead of simply the pixel number
            else:
                # Flux interpolator (interpolating only good px)
                interp_px_time_series_flux = interp1d(
                    x=mjds[~bad_phase_mask],
                    y=flux[:,spec_i,px_i][~bad_phase_mask],
                    kind=interpolation_method,
                    bounds_error=True,)
                
                # Sigma interpolator (interpolating only good px)
                interp_px_time_series_err = interp1d(
                    x=mjds[~bad_phase_mask],
                    y=e_spectra[:,spec_i,px_i][~bad_phase_mask],
                    kind=interpolation_method,
                    bounds_error=True,)
                
                # Interpolate and update
                flux[:,spec_i,px_i][bad_phase_mask] = \
                    interp_px_time_series_flux(mjds[bad_phase_mask])
                
                e_spectra[:,spec_i,px_i][bad_phase_mask] = \
                    interp_px_time_series_err(mjds[bad_phase_mask])
                
                # Update the bad px mask to just have nan values (i.e. filled)
                sc_mask_phase[:,spec_i,px_i] = np.isnan(flux[:,spec_i,px_i])

                # HACK: this should always be 0 right?
                assert np.sum(np.isnan(flux[:,spec_i,px_i])) == 0
    
    #--------------------------------------------------------------------------
    # Clip along spectral dimension
    #--------------------------------------------------------------------------
    # Sigma clip along the spectral dimension. Note that since we're clipping
    # along the spectral dimension, we only clip the *upper* bounds so as to
    # not remove absorption features.
    sc_mask_spec = np.full_like(flux, False)

    if do_clip_spectral_dimension:
        for spec_i in range(n_spec):
            for phase_i in range(n_phase):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sc_mask_spec[phase_i,spec_i,:] = sigma_clip(
                        data=flux[phase_i,spec_i,:],
                        sigma_upper=sigma_threshold_spectral,).mask
        
    # Combine bad px masks
    bad_px_mask = np.logical_or(sc_mask_phase, sc_mask_spec)

    #--------------------------------------------------------------------------
    # Normalisation + wrapping up
    #--------------------------------------------------------------------------
    # Finally we pre-normalise the spectra. Per Birkby+2013, this involves:
    #   1) Dividing each individual *spectrum* by its mean value
    #   2) Subtracting unity
    if do_normalise:
        mean_2D = np.nanmean(flux, axis=2)  # [n_phase, n_spec]
        mean_3D = np.broadcast_to(
            mean_2D[:,:,None], flux.shape)  # [n_phase, n_spec, n_px]
        resid_init = flux / mean_3D - 1
        
        # Also rescale uncertainties
        e_flux /= mean_3D

        # Finally, for any remaining non-interpolated bad px, set the fluxes
        # and uncertainties to default values
        resid_init[bad_px_mask] = 0.0
        flux[bad_px_mask] = 1.0
        e_flux[bad_px_mask] = 1E20

        # No normalisation
    else:
        resid_init = flux.copy()

        # Finally, for any remaining non-interpolated bad px, set the fluxes
        # and uncertainties to default values
        resid_init[bad_px_mask] = np.median(resid_init[~bad_px_mask])
        flux[bad_px_mask] = np.median(flux[~bad_px_mask])
        e_flux[bad_px_mask] = 1E20

    return resid_init, flux, e_flux


def detrend_spectra(
    resid_init,
    e_resid,
    n_iter,
    detrending_algorithm="SYSREM",
    tolerance=1E-6,
    max_converge_iter=100,
    diff_method="max",
    sigma_threshold=3.0,):
    """Function for detrending time-series transmission spectroscopy datasets.
    This function is the master function that hands off to implementations of
    each different algorithm.

    Parameters
    ----------
    resid_init, e_resid: 2D float array
        Initial residual and uncertainty arrays of shape:
        [n_phase, n_spec, n_px].

    sysrem_algorithm: str
        Which SYSREM algorithm to use. Currently either 'SYSREM' or 'PISKUNOV'.

    n_iter: int
        The number of SYSREM iterations to run.

    tolerance: float, default: 1E-6
        The convergence threshold for a given SYSREM iteration.

    max_converge_iter: int, default: 100
        The maximum number of iterations to run each SYSREM iteration for while
        converging.

    diff_method: str, default: 'max'
        Function used to assess convergence: ['mean', 'median', 'min', 'max']

    sigma_threshold: float, default: 3.0
        Sigma threshold used for exluding outliers when using the 'PISKUNOV'
        algorithm.

    Returns
    -------
    resid_all: 3D float array
       Residual array of shape [n_iter+1, n_phase, n_px] where the +1 is so we
       store the starting array of residuals.
    """
    VALID_METHODS = ["SYSREM", "PISKUNOV"]

    if detrending_algorithm.upper() not in VALID_METHODS:
        raise ValueError("Method must be in {}".format(VALID_METHODS))

    if detrending_algorithm.upper() == "SYSREM":
        print("Using SYSREM for detrending.")
        resid = _sysrem_default(
            resid_init=resid_init,
            e_resid=e_resid,
            n_iter=n_iter,
            tolerance=tolerance,
            max_converge_iter=max_converge_iter,
            diff_method=diff_method,)

    elif detrending_algorithm.upper() == "PISKUNOV":
        print("Using Piskunov quadratic detrending method.")
        resid = _sysrem_piskunov(
            spectra=resid_init,
            n_iter=n_iter,
            sigma_threshold=sigma_threshold,)

    return resid


def _sysrem_default(
    resid_init,
    e_resid,
    n_iter,
    tolerance=1E-6,
    max_converge_iter=100,
    diff_method="max",):
    """Function to run the iterative SYSREM algorithm on a set of fluxes and 
    uncertainties. Note that this only runs on a single spectral segment at a
    time.

    SYSREM aims to iteratively minimise the expression:

        S^2 = Σ_ij [(r_ij - c_i * a_j)^2 / σ_ij^2]

    Where:
     - i is the spectral pixel index
     - j is the phase index
     - r_ij are the residuals at px i and phase j
     - c_i is the adopted 'trend' gradient for px i
     - a_j is the adopted 'trend' magnitude at phase j
     - σ_ij are the flux uncertainties on px i and phase j

    The basic steps are as follows:
     1: Clean spectra, mask out bad pixels.
     2: Subtract median spectral px along phase dimension.
     3: Fit residuals for trend magnitude.
     4: Fit for gradient that best minimises residuals when combined with trend
        magnitudes.
     5: Subtract linear trend, repeat 3-5 for as many iterations as required.

    Adapted from:
    www.github.com/AWehrhahn/ChEATS/blob/master/exoplanet_transit_snr/sysrem.py
    
    Itself originally adapted from:
    www.github.com/stephtdouglas/PySysRem/blob/master/sysrem.py

    Parameters
    ----------
    spectra, e_spectra: 2D float array
        Unnormalised spectra and spectroscopic uncertainties of shape 
        [n_phase, n_px].

    n_iter: int
        The number of SYSREM iterations to run.

    tolerance: float, default: 1E-6
        The convergence threshold for a given SYSREM iteration.

    max_converge_iter: int, default: 100
        The maximum number of iterations to run each SYSREM iteration for while
        converging.

    diff_method: str, default: 'max'
        Function used to assess convergence: ['mean', 'median', 'min', 'max']

    Returns
    -------
    resid_all: 3D float array
       Residual array of shape [n_iter+1, n_phase, n_px] where the +1 is so we
       store the starting array of residuals.
    """
    VALID_DIFF_FUNCS = {
        "MEAN":np.nanmean,
        "MEDIAN":np.nanmedian,
        "MIN":np.nanmin,
        "MAX":np.nanmax,}
    
    if diff_method.upper() in VALID_DIFF_FUNCS.keys():
        diff_func = VALID_DIFF_FUNCS[diff_method.upper()]
    else:
        raise ValueError("Invalid diff_method, must be in {}".format(
            VALID_DIFF_FUNCS.keys()))

    n_phase, n_px = resid_init.shape

    # TODO Check to make sure we have proper uncertainties
    pass

    # Intialise resid array for all iterations, shape [n_iter+1, n_phase, n_px]
    resid_all = np.full((n_iter+1, n_phase, n_px), np.nan)

    # Store median subtracted residuals
    resid_all[0] = resid_init
    e_resid_sq = e_resid.copy().T**2

    # Run SYSREM
    for sr_iter_i in range(n_iter):
        print("-"*10, "Iter #{}".format(sr_iter_i+1), "-"*10, sep="\n")
        # Initialise vectors
        cc = np.zeros(n_px)     # gradient
        aa = np.ones(n_phase)   # "airmasses"

        # Grab a temporary handle for the current set of residuals.
        # Note that we're going to use the transpose here to match Stephanie 
        # Douglas' (and Ansgar Wehrhahn's) convention.
        resid = resid_all[sr_iter_i].T.copy()

        # Iterate until convergence for cc an aa vectors
        converge_step_i = 0
        has_converged = False

        while converge_step_i < max_converge_iter and not has_converged:
            # Comnpute an estimate for c for converge_step_i
            c_num = np.nansum(aa * resid / e_resid_sq, axis=1,)
            c_den = np.nansum(aa**2 / e_resid_sq, axis=1,)
            cc_est = np.divide(c_num, c_den,)

            # Compute an estimate for a at converge_step_i
            a_num = np.nansum(cc_est[:, None] * resid / e_resid_sq, axis=0,)
            a_den = np.nansum(cc_est[:, None]**2 / e_resid_sq, axis=0,)
            aa_est = np.divide(a_num, a_den,)

            # Calculate diff
            c_diff = diff_func((cc_est - cc)**2)
            a_diff = diff_func((aa_est - aa)**2)
            diff = c_diff + a_diff
            
            # Update our values
            cc = cc_est.copy()
            aa = aa_est.copy()

            # Update
            converge_step_i += 1
            if diff < tolerance:
                has_converged = True

            print(
                "\t{:3.0f}/{}".format(converge_step_i, max_converge_iter),
                "\tΔc = {:4.8f}".format(c_diff),
                "\tΔa = {:4.8f}".format(a_diff))

        # We've converged, compute our new vector of residuals
        systematic = cc[:, None] * aa[None, :]
        resid_all[sr_iter_i+1] = (resid - systematic).T

    return resid_all


def _sysrem_piskunov(
    spectra,
    n_iter,
    sigma_threshold=3.0,):
    """Nikolai Piskunov's implementation of 'SYSREM'. This algorithm fits each
    spectral pixel with a parabola (i.e. a fit to flux in the phase dimension),
    sigma clips those pixels sigma_threshold aberrant this fit, repeats the
    fit, then divides by the resulting curve. At present this algorithm only
    performs a single 'SYSREM' iteration.

    Note: strictly speaking, this algorithm has some key differences to SYSREM:
     - SYSREM takes into account uncertainties
     - SYSREM fits an arbitrary 'airmass' function to *all* spectral pixels,
       which is then modulated by a per-spectral-px 'extinction' coefficient.
       However, this algorithm simply uses n_px independent parabola fits.
    
    Parameters
    ----------
    spectra: 3D float array
        Observed spectra of shape [n_phase, n_spec, n_px].

    n_iter: int
        Number of iterations to run on the spectra.

    sigma_threshold: float, default: 3.0
        Sigma threshold above which to exclude pixels from the parabola fit.

    Returns
    -------
    resid_all: 4D float array
        Residuals of shape [n_iter+1, n_phase, n_spec, n_px]
    """
    # Grab dimensions for convenience
    (n_phase, n_spec, n_px) = spectra.shape

    # Initialise our residual vector
    resid_all = np.full((n_iter+1, n_phase, n_spec, n_px), np.nan)

    # Add in our starting spectra
    resid_all[0] = spectra.copy()

    # Run for a number of iterations
    for sr_iter_i in range(n_iter):
        # Loop over all spectral segments
        for spec_i in range(n_spec):
            # Grab the flux for this spectral segment [n_phase, n_px]
            o = resid_all[sr_iter_i,:,spec_i,:]

            # Normalise spectrum: dividing by the average flux in each segment
            oo = np.sum(o,axis=1)/n_px
            o /= np.broadcast_to(oo[:,None], (n_phase, n_px)) + 1
            x = np.arange(n_phase)
            
            desc = "Running SYSREM {}/{} for spec {}/{}".format(
                sr_iter_i+1, n_iter, spec_i+1, n_spec)

            # Loop over all pixels
            for px_i in tqdm(range(n_px), desc=desc, leave=False):
                # Sigma clip along phase dimension, clipping > 3 sigma
                y = o[:,px_i]

                # Fit second order polynomial to the data
                coef = polyfit(x=x, y=y, deg=2,)
                calc_poly = Polynomial(coef)
                yy = calc_poly(x)
                
                # Compute standard deviation of residuals
                diff = y-yy
                dev = np.nanstd(diff)
                
                # Compute a bad px mask
                bad_phase_mask = np.abs(diff) > sigma_threshold*dev
                n_bad_phase = np.sum(bad_phase_mask)

                # If there are bad px, refit
                if n_bad_phase > 0:
                    coef = polyfit(
                        x=x[~bad_phase_mask],
                        y=y[~bad_phase_mask],
                        deg=2,)
                    calc_poly = Polynomial(coef)
                    yy = calc_poly(x)
                
                # Divide by best fit parabola + store the result for this pixel
                resid_all[sr_iter_i+1,:,spec_i,px_i] = y/yy-1.0

    return resid_all


def cross_correlate_sysrem_resid(
    waves,
    sysrem_resid,
    template_wave,
    template_spec,
    bcors,
    cc_rv_step=1,
    cc_rv_lims=(-200,200),
    interpolation_method="cubic",):
    """Function to cross correlate a template spectrum against all spectral
    segments, for all phases, and for all SYSREM iterations. This cross
    correlation implementation does not use weight by uncertainties, but rather
    normalises by the two arrays being cross correlated.

    Parameters
    ----------
    waves: 2D float array
        Wavelength scale of shape [n_spec, n_px].

    sysrem_resid: 4D float array
        Combined (i.e. for multiple spectral segments) set of SYSREM residuals
        of shape [n_sysrem_iter, n_phase, n_spec, n_px].

    template_wave, template_spec: 1D float array
        Wavelength scale and spectrum of template spectrum to be interpolated 
        against.

    bcors: 1D float array
        Array of barycentric corrections (-bcor + rv_star) of shape [n_phase].

    cc_rv_step: float, default: 1
        Step size for cross correlation in km/s.

    cc_rv_lims: float tuple, default: (-200,200)
        Lower and upper bounds for cross correlation in km/s.

    interpolation_method: str, default: "cubic"
        Default interpolation method to use with scipy.interp1d. Can be one of: 
        ['linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic',
        'cubic', 'previous', or 'next'].

    Returns
    -------
    cc_rvs: 1D float array
        RV values that were used when cross correlating in km/s.
    
    ccv_per_spec: 4D float array
        Array of cross correlation values of shape:
        [n_sysrem_iter, n_phase, n_spec, n_rv_steps].

    ccv_combined: 3D float array, default: None
        3D float array of the *combined* cross correlations for each SYSREM
        iteration of shape [n_sysrem_iter, n_phase, n_rv_step].
    """
    # Initialise RV vector
    cc_rvs = np.arange(cc_rv_lims[0], cc_rv_lims[1]+cc_rv_step, cc_rv_step)

    # Grab dimensions for convenience
    (n_sysrem_iter, n_phase, n_spec, n_px) = sysrem_resid.shape
    n_rv_steps = len(cc_rvs)

    # Intiialise output array
    ccv_per_spec = np.zeros((n_sysrem_iter, n_phase, n_spec, n_rv_steps))

    # Initialise output array for *global* cross-correlation
    ccv_combined = np.zeros((n_sysrem_iter, n_phase, n_rv_steps))

    #--------------------------------------------------------------------------
    # Create 4D interpolated grid of spectra
    #--------------------------------------------------------------------------
    # Loop over spectral, phase, and RV dimensions to create a template grid.
    # This will be constant for all SYSREM iterations, so we create it once and
    # use it as many times as we need.
    spec_4D = np.ones((n_rv_steps, n_phase, n_spec, n_px,))

    # Initialise template spectrum interpolator.
    temp_interp = interp1d(
        x=template_wave,
        y=template_spec,
        kind=interpolation_method,
        bounds_error=False,
        fill_value=np.nan,)

    for spec_i in range(n_spec):
        desc = "Creating spectral template grid {}/{} for all RVs".format(
            spec_i+1, n_spec)
        
        for rv_i, rv in enumerate(tqdm(cc_rvs, leave=False, desc=desc)):
            for phase_i in range(n_phase):
                # Doppler shift for new wavelength scale
                bcor = bcors[phase_i]
                wave_rv_shift = \
                    waves[spec_i] * (1-(rv+bcor)/(const.c.si.value/1000))

                # Interpolate to wavelength scale
                spec_1D = temp_interp(wave_rv_shift)

                # If this is ever true, our template is insufficiently long
                # in wavelength.
                if np.sum(np.isnan(spec_1D)) > 0:
                    raise ValueError("Nans in interpolated array")
            
                # Store 1D spectrum
                spec_4D[rv_i, phase_i, spec_i, :] = spec_1D

    #--------------------------------------------------------------------------
    # Cross correlation
    #--------------------------------------------------------------------------
    # Loop over each set of residuals for each SYSREM iteration
    for sysrem_iter_i in range(n_sysrem_iter):

        desc = "CCing for SYSREM iter {}/{}".format(
            sysrem_iter_i+1, n_sysrem_iter)

        #----------------------------------------------------------------------
        # Combine per-spectral segment CCs
        #----------------------------------------------------------------------
        # Cross correlate each [n_phase, n_px] slice from spec_4D
        for spec_i in tqdm(range(n_spec), leave=False, desc=desc):
            # Loop over all RVs and cross correlate against each phase
            for rv_i, rv in enumerate(cc_rvs):
                # Store this for computing the *global* cross correlation
                spec_2D = spec_4D[rv_i, :, spec_i, :]
                
                # Calculate the cross correlation weighted by the uncertainties
                # TODO: we add +1 to the residuals so they flucuate about 1.
                # This should be doublechecked.
                resid = sysrem_resid[sysrem_iter_i, :, spec_i].copy() + 1
                
                # We need to mask out entirely nan columns in order for the 
                # cross-correlation maths to behave.
                nm = ~np.all(np.isnan(resid,), axis=0)

                # Calculate cross-correlation, sum along pixel dimension
                cc_num = np.nansum(resid[:,nm] * spec_2D[:,nm], axis=1)
                cc_den = np.sqrt(
                    np.nansum(resid[:,nm]**2, axis=1)
                    * np.nansum(spec_2D[:,nm]**2, axis=1))
                cc_val = cc_num / cc_den

                # Store
                ccv_per_spec[sysrem_iter_i, :, spec_i, rv_i] = cc_val

        #----------------------------------------------------------------------
        # Combine all spectral segments into global CC for sysrem_iter_i
        #----------------------------------------------------------------------
        # Now that we've run the cross correlation for n_spec * n_rv, combine
        # Compute the *global* cross correlation for this SYSREM iteration
        resid_3D = sysrem_resid[sysrem_iter_i].copy() +1
        resid_4D = np.broadcast_to(
            array=resid_3D[None,:,:,:],
            shape=(n_rv_steps, n_phase, n_spec, n_px,),)

        # We still need to mask out all-nan columns here, and the easiest way
        # to do that is to simply concatenate the spectral segment and pixel
        # dimensions into one single long dimension.
        resid_4Dc = resid_4D.reshape((n_rv_steps, n_phase, n_spec*n_px,))
        spec_4Dc = spec_4D.reshape((n_rv_steps, n_phase, n_spec*n_px,))

        # Compute mask for where entire phase dimension is all nans. We assume
        # that this mask will be in common for all SYSREM iterations.
        nm = ~np.all(np.isnan(resid_4Dc[0],), axis=0)

        # Sum over the (now collapsed) px and spectral segment dimensions
        # spec_4D has shape (n_rv_steps, n_phase, n_spec, n_px,)
        cc_num = np.nansum(resid_4Dc[:,:,nm] * spec_4Dc[:,:,nm], axis=2)
        cc_den = np.sqrt(
            np.nansum(resid_4Dc[:,:,nm]**2, axis=2)
            * np.nansum(spec_4Dc[:,:,nm]**2, axis=2))
        cc_val = cc_num / cc_den

        # Store
        ccv_combined[sysrem_iter_i, :, :] = cc_val.T

    #--------------------------------------------------------------------------
    # Normalise all cross correlations
    #--------------------------------------------------------------------------
    for iter_i in range(n_sysrem_iter):
        desc = "Normalising for SYSREM iter".format(
            sysrem_iter_i+1, n_sysrem_iter)
        
        # Normalise the cross correlation for each spectral segment
        for spec_i in tqdm(range(n_spec), desc=desc, leave=False):
            # Sum along the CC direction
            norm_1D = np.nansum(ccv_per_spec[iter_i, :,spec_i,:], axis=1)
            norm_2D = np.broadcast_to(norm_1D[:,None], (n_phase, n_rv_steps))

            ccv_per_spec[iter_i, :,spec_i,:] /= norm_2D

        # Now also normalise the *global* CC, sum along CC dimension
        norm_1D = np.nansum(ccv_combined[iter_i,:,:], axis=1)
        norm_2D = np.broadcast_to(norm_1D[:,None], (n_phase, n_rv_steps))

        ccv_combined[iter_i,:,:] /= norm_2D

    # All done, return our array of cross correlation results
    return cc_rvs, ccv_per_spec, ccv_combined


def compute_Kp_vsys_map(
    cc_rvs,
    ccv_per_spec,
    transit_info,
    ccv_combined=None,
    Kp_lims=(0,400),
    Kp_step=0.5,
    vsys_lims=None,
    interpolation_method="cubic",):
    """Function to compute the Kp-Vsys map given a set of cross correlation
    values from cross_correlate_sysrem_resid.

    V_p = K_p sin(2π * φ) + Vsys - v_bary + v_max

    Parameters
    ----------
    cc_rvs: 1D float array
        Vector of RV steps of length [n_rv_step]
    
    ccv_per_spec: 4D float array
        4D float array of cross correlation results with shape:
        [n_sysrem_iter, n_phase, n_spec, n_rv_step]
    
    transit_info: pandas DataFrame
        Pandas DataFrame with header/computed information about each timestep.

    syst_info: pandas DataFrame
        DataFrame containing planet/star/system properties.

    ccv_combined: 3D float array, default: None
        3D float array of the *combined* cross correlations for each SYSREM
        iteration of shape [n_sysrem_iter, n_phase, n_rv_step].

    Kp_lims: float tuple, default: (0,400)
        Lower and upper bounds of Kp in km/s to sample between.

    Kp_step: float, default: 0.5
        Kp step size to use when sampling in km/s.

    vsys_lims: tuple array or None, default: None
        This is the RV range of the x axis for our Kp-Vsys plots. This should
        be a smaller range than cc_rvs, as the idea is we CC wider than we
        need, then clip off the edges to a) remove edge effects, and b) focus
        on the region the planet is actually in. If None, no limits are
        applied.

    interpolation_method: str, default: "cubic"
        Default interpolation method to use with scipy.interp1d. Can be one of: 
        ['linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic',
        'cubic', 'previous', or 'next'].

    Returns
    -------
    cc_rvs_subset: 1D float array
        Updated vector of RV steps of length per vsys_lims.
    
    Kp_steps: 1D float array
        Array of Kp steps from Kp_lims[0] to Kp_lims[1] in steps of Kp_step.
    
    Kp_vsys_map_per_spec: 4D array
        Grid of Kp_vsys maps of shape: 
        [n_sysrem_iter, n_spec, n_Kp_steps, n_rv_step]

    Kp_vsys_map_combined: 3D float array [optional]
        3D float array of the *combined* Kp-Vsys map of shape: 
        [n_sysrem_iter, n_Kp_steps, n_rv_step].
    """
    # If we've been given a set of combined cross-correlation values, 
    # concatenate these to the end of our array so we can compute a Kp-Vsys map
    # from them as well.
    if ccv_combined is not None:
        cc_values = np.concatenate(
            (ccv_per_spec, ccv_combined[:,:,None,:]), axis=2)
    else:
        cc_values = ccv_per_spec.copy()

    # Grab dimensions for convenience
    (n_sysrem_iter, n_phase, n_spec, n_rv_step) = cc_values.shape

    # Initialise Kp sampling
    Kp_steps = np.arange(Kp_lims[0], Kp_lims[1]+1, Kp_step)

    n_Kp_steps = len(Kp_steps)

    # Initialise output array
    Kp_vsys_map = np.zeros((n_sysrem_iter, n_spec, n_Kp_steps, n_rv_step,))

    # Loop over all sets of sysrem residuals
    for sr_iter_i in range(n_sysrem_iter):
        # Loop over all spectral segments
        for spec_i in range(n_spec):
            desc = \
                "Creating Kp-Vsys map for SYSREM {}/{}, spec {}/{}...".format(
                    sr_iter_i+1, n_sysrem_iter, spec_i+1, n_spec)
            
            # Loop over all Kp values
            for Kp_i, Kp in enumerate(tqdm(Kp_steps, leave=False, desc=desc)):
                # Initialise grid of shifted phases
                cc_values_shifted = np.zeros((n_phase, n_rv_step))

                # Loop over all phases
                for phase_i in range(n_phase):
                    # Grab phases from transit_info, noting the negation
                    phase = -1 * transit_info.iloc[phase_i]["phase_mid"]

                    # Compute the velocity shift between this phase and the
                    # planet rest frame velocity. Note that we assume we've 
                    # aready removed the stellar and barycentric velocities.
                    vp = Kp * np.sin(2*np.pi*phase)

                    # Create interpolator for the CC values of this phase
                    interp_cc_vals = interp1d(
                        x=cc_rvs,
                        y=cc_values[sr_iter_i, phase_i, spec_i, :],
                        bounds_error=False,
                        fill_value=np.nan,
                        kind=interpolation_method,)

                    # Shift CC value to rest frame using calculated vp value
                    cc_values_shifted[phase_i] = interp_cc_vals(cc_rvs - vp)

                # Sum all shifted CC values along the phase axis
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        action='ignore', message='Mean of empty slice')
                    Kp_vsys_map[sr_iter_i, spec_i, Kp_i] = \
                        np.nanmean(cc_values_shifted, axis=0)

    # If we've been provided vsys (i.e. x axis) limits, now that we've
    # created our Kp-Vsys map we can clip the edges off to avoid edge effects.
    if vsys_lims is not None:
        if vsys_lims[0] >= vsys_lims[1]:
            raise ValueError("vsys_lims[0] must be < vsys_lims[1]")
        
        vsys_mask = np.logical_and(
            cc_rvs >= vsys_lims[0],
            cc_rvs <= vsys_lims[1])
        Kp_vsys_map = Kp_vsys_map[:,:,:,vsys_mask]

        # And update cc_rvs
        cc_rvs_subset = np.arange(
            vsys_lims[0], vsys_lims[1]+1, np.diff(cc_rvs)[0])

    # If we were given a combined set of CCVs, removed the combined set of
    # cross correlation values and return that as its own variable.
    if ccv_combined is not None:
        Kp_vsys_map_per_spec = Kp_vsys_map[:,:-1,:,:]
        Kp_vsys_map_combined = Kp_vsys_map[:,-1,:,:]

        return cc_rvs_subset, Kp_steps, Kp_vsys_map_per_spec, \
            Kp_vsys_map_combined

    # Otherwise return as is
    else:
        return cc_rvs_subset, Kp_steps, Kp_vsys_map
    

def combine_kp_vsys_map(Kp_vsys_map_list):
    """Combines Kp-Vsys maps from adjacent nights. We do this by masking out
    negative values and then adding in quadrature.

    Parameters
    ----------
    Kp_vsys_map_list: list of float arrays
        List of float arrays, where the float arrays are assumed to be the same
        shape. Typical shapes:
         - 3D of shape: [n_sysrem_iter, n_Kp_steps, n_rv_step]
         - 4D of shape: [n_sysrem_iter, n_spec, n_Kp_steps, n_rv_step]

    Returns
    -------
    map_combined: float array
        Combined map with the same shape as the input maps from each night.
    """
    # Stack
    map_stacked = np.stack(Kp_vsys_map_list).copy()

    # Mask out negative values
    is_neg = map_stacked < 0
    map_stacked[is_neg] = 0

    # Add in quadtrature
    map_combined = np.sum(map_stacked**2, axis=0)**0.5

    return map_combined