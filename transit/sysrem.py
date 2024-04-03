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

def clean_and_compute_initial_resid(
    spectra,
    e_spectra,
    bad_px_mask_init,
    mjds,
    sigma_threshold_phase=6.0,
    sigma_threshold_spectral=6.0,
    n_max_phase_bad_px=5,
    do_clip_spectral_dimension=False,):
    """Function to clean a spectral data cube prior to running SYSREM. We sigma
    clip along the phase dimension (i.e. the time series of each pixel) as well
    as along the spectral dimension. 

    Note that this function expects only spectra from a single spectral
    segment (or a flattened set of many segments.

    TODO: Interpolate linearly along the phase dimension, no point doing both
    nan and 1E20.
    
    Parameters
    ----------
    spectra, e_spectra: 2D float array
        Unnormalised spectra and spectroscopic uncertainties of shape 
        [n_phase, n_px].
    
    bad_px_mask_init: 2D float array
        Bad px mask of shape [n_phase, n_px].

    mjds: TODO
    
    sigma_threshold_phase: float, default: 5.0
        The sigma clipping threshold for when sigma clipping along the *phase*
        dimension. Note that this used for both lower and upper bound clipping.

    sigma_threshold_spectral: float, default: 6.0
        The sigma clipping threshold for when sigma clipping along the 
        *spectral* dimension. Note that we only sigma clip on the upper bound
        so as to not remove deep spectral features.

    n_max_phase_bad_px: TODO

    Returns
    -------
    residuals: 2D float array
        Initial set of residuals to use for SYSREM composed of the median 
        subtracted fluxes.
    
    flux, e_flux: 2D float array
        Nnormalised spectra and spectroscopic uncertainties with bad pixel
        fluxes and uncertainties set to nan and 1E20 respectively of shape 
        [n_phase, n_px].
    """
    # Grab shape
    (n_phase, n_px) = spectra.shape

    # Duplicate arrays
    flux = spectra.copy()
    e_flux = e_spectra.copy()

    # Sigma clip along phase dimension. Note that here we're doing both upper
    # *and* lower bound clipping.

    sc_mask_phase = np.full_like(bad_px_mask_init, False)

    for px_i in range(n_px):
        # Sigma clip along the phase dimension to compute the bad pixel mask 
        # for px i
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bad_phase_mask = sigma_clip(
                data=flux[:,px_i],
                sigma=sigma_threshold_phase,).mask
        
        # If we have more than the threshold number of bad px, mask out the
        # entire column (i.e. mask out px_i)
        if np.sum(bad_phase_mask) > n_max_phase_bad_px:
            sc_mask_phase[:,px_i] = True

        # If the first or last phase is a bad px, then mask out the entire row
        # since we can't interpolate them.
        elif bad_phase_mask[0] or bad_phase_mask[-1]:
            sc_mask_phase[:,px_i] = True

        # Otherwise interpolate the missing values using the times as X values
        # instead of simply the pixel number
        else:
            # Flux interpolator (interpolating only good px)
            interp_px_time_series_flux = interp1d(
                x=mjds[~bad_phase_mask],
                y=flux[:,px_i][~bad_phase_mask],
                bounds_error=True,)
            
            # Sigma interpolator (interpolating only good px)
            interp_px_time_series_err = interp1d(
                x=mjds[~bad_phase_mask],
                y=e_spectra[:,px_i][~bad_phase_mask],
                bounds_error=True,)
            
            # Interpolate and update
            flux[:,px_i][bad_phase_mask] = \
                interp_px_time_series_flux(mjds[bad_phase_mask])
            
            e_spectra[:,px_i][bad_phase_mask] = \
                interp_px_time_series_err(mjds[bad_phase_mask])
            
            # Update the bad px mask to just have nan values (i.e. filled)
            sc_mask_phase[:,px_i] = np.isnan(flux[:,px_i])

            # HACK: this should always be 0 right?
            assert np.sum(np.isnan(flux[:,px_i])) == 0
        
    # Sigma clip along the spectral dimension. Note that since we're clipping
    # along the spectral dimension, we only clip the *upper* bounds so as to
    # not remove absorption features.
    sc_mask_spec = np.full_like(bad_px_mask_init, False)

    if do_clip_spectral_dimension:
        for phase_i in range(n_phase):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sc_mask_spec[phase_i,:] = sigma_clip(
                    data=flux[phase_i,:],
                    sigma_upper=sigma_threshold_spectral,).mask
        
    # Combine bad px masks
    bad_px_mask = np.logical_or(
        bad_px_mask_init, np.logical_or(sc_mask_phase, sc_mask_spec))

    # Clean spectra
    flux[bad_px_mask] = np.nan
    e_flux[bad_px_mask] = 1E20

    # Median subtract to get initial set of residuals
    residuals = (flux.T - np.nanmedian(flux, axis=1)).T

    return residuals, flux, e_flux


def run_sysrem(
    spectra,
    e_spectra,
    bad_px_mask,
    n_iter,
    mjds,
    tolerance=1E-6,
    max_converge_iter=100,
    diff_method="max",
    sigma_threshold_phase=6.0,
    sigma_threshold_spectral=6.0,):
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

    Parameters
    ----------
    spectra, e_spectra: 2D float array
        Unnormalised spectra and spectroscopic uncertainties of shape 
        [n_phase, n_px].
    
    bad_px_mask_init: 2D float array
        Bad px mask of shape [n_phase, n_px].

    n_iter: int
        The number of SYSREM iterations to run.
    
    mjds: TODO

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

    n_phase, n_px = spectra.shape

    # TODO Check to make sure we have proper ucnertainties
    pass

    # Intialise resid array for all iterations, shape [n_iter+1, n_phase, n_px]
    resid_all = np.full((n_iter+1, n_phase, n_px), np.nan)

    # Use bad px mask to clean input array and compute initial residual vector
    resid_init, flux, e_flux = clean_and_compute_initial_resid(
        spectra=spectra,
        e_spectra=e_spectra,
        bad_px_mask_init=bad_px_mask,
        mjds=mjds,
        sigma_threshold_phase=sigma_threshold_phase,
        sigma_threshold_spectral=sigma_threshold_spectral,)

    # Store median subtracted residuals
    resid_all[0] = resid_init
    e_flux = e_flux.T

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
            c_num = np.nansum(aa * (resid / e_flux**2), axis=1,)
            c_den = np.nansum(aa ** 2 / e_flux**2, axis=1,)
            cc_est = np.divide(c_num, c_den,)

            # Compute an estimate for a at converge_step_i
            a_num = np.nansum(cc_est[:, None] * (resid / e_flux**2), axis=0,)
            a_den = np.nansum(cc_est[:, None] ** 2 / e_flux**2, axis=0,)
            aa_est = np.divide(a_num, a_den,)

            # Calculate diff
            c_diff = diff_func((cc_est - cc)**2)
            a_diff = diff_func((aa_est - aa)**2)
            
            # Update our values
            cc = cc_est
            aa = aa_est

            # Update
            converge_step_i += 1
            if c_diff < tolerance and a_diff < tolerance:
                has_converged = True

            print(
                "\t{:3.0f}/{}".format(converge_step_i, max_converge_iter),
                "\tΔc = {:4.8f}".format(c_diff),
                "\tΔa = {:4.8f}".format(a_diff))

        # We've converged, compute our new vector of residuals
        systematic = cc[:, None] * aa[None, :]
        resid_all[sr_iter_i+1] = (resid - systematic).T

    # We've completed all iterations, return
    return resid_all


def cross_correlate_sysrem_resid(
    waves,
    sysrem_resid,
    sigma_spec,
    template_wave,
    template_spec,
    cc_rv_step=1,
    cc_rv_lims=(-200,200),):
    """Function to cross correlate a template spectrum against all spectral
    segments, for all phases, and for all SYSREM iterations.

    Parameters
    ----------
    waves: 2D float array
        Wavelength scale of shape [n_spec, n_px].

    sysrem_resid: 4D float array
        Combined (i.e. for multiple spectral segments) set of SYSREM residuals
        of shape [n_sysrem_iter, n_phase, n_spec, n_px].

    sigma_spec: 3D float array
        Normalised uncertainties for the spectral datacube of shape
        [n_phase, n_spec, n_px].

    template_wave, template_spec: 1D float array
        Wavelength scale and spectrum of template spectrum to be interpolated 
        against.

    cc_rv_step: float, default: 1
        Step size for cross correlation in km/s.

    cc_rv_lims: float tuple, default: (-200,200)
        Lower and upper bounds for cross correlation in km/s.

    Returns
    -------
    cc_rvs: 1D float array
        RV values that were used when cross correlating in km/s.
    
    cc_values: 4D float array
        Array of cross correlation values of shape:
        [n_sysrem_iter, n_phase, n_spec, n_rv_steps].
    """
    # Initialise RV vector
    cc_rvs = np.arange(cc_rv_lims[0], cc_rv_lims[1]+cc_rv_step, cc_rv_step)

    # Grab dimensions for convenience
    (n_sysrem_iter, n_phase, n_spec, n_px) = sysrem_resid.shape
    n_rv_steps = len(cc_rvs)

    # Intiialise output array
    cc_values = np.zeros((n_sysrem_iter, n_phase, n_spec, n_rv_steps))

    # Initialise template spectrum interpolator. Make sure to subtract 1 from
    # the template spectrum so both it and the residuals fluctuate about zero.
    temp_interp = interp1d(
        x=template_wave,
        y=template_spec-1,
        bounds_error=False,
        fill_value=np.nan,)

    print("Cross correlating...")

    # Loop over each set of residuals for each SYSREM iteration
    for sysrem_iter_i in range(n_sysrem_iter):

        desc = "Cross Correlating for SYSREM iter #{}".format(sysrem_iter_i)

        # Loop over all spectral segments
        for spec_i in tqdm(range(n_spec), leave=False, desc=desc):
            # Loop over all RVs and cross correlate against each phase
            # Note: all phases need the same cross correlation to occur, so we
            # can save on interpolation/loops by just interpolating once per RV
            for rv_i, rv in enumerate(cc_rvs):
                # Doppler shift for new wavelength scale
                wave_rv_shift = waves[spec_i] * (1- rv/(const.c.si.value/1000))

                # Interpolate to wavelength scale
                tspec_rv_shift = temp_interp(wave_rv_shift)

                if np.nansum(tspec_rv_shift) == 0:
                    raise ValueError("All nan interpolated array")

                # Tile this to all phases
                spec_3D = np.broadcast_to(
                    tspec_rv_shift[None,:], (n_phase, n_px))
                
                # Enforce the the phase direction is the same
                ss = set(spec_3D[:,1024])
                assert len(ss) == 1

                # Calculate the cross correlation weighted by the uncertainties
                resid = sysrem_resid[sysrem_iter_i, :, spec_i]
                sigma = sigma_spec[:, spec_i, :]
                cc_val = np.nansum(resid * spec_3D / sigma**2, axis=1)
                
                #cc_num = np.nansum(resid * spec_3D, axis=1)
                #cc_den = np.sqrt(
                #    np.nansum(resid**2, axis=1) * np.nansum(spec_3D**2, axis=1))
                #cc_val = cc_num / cc_den

                # Store
                cc_values[sysrem_iter_i, :, spec_i, rv_i] = cc_val

    # All done, return our array of cross correlation results
    return cc_rvs, cc_values


def compute_Kp_vsys_map(
    cc_rvs,
    cc_values,
    transit_info,
    syst_info,
    Kp_lims=(0,400),
    Kp_step=0.5,):
    """Function to compute the Kp-Vsys map given a set of cross correlation
    values from cross_correlate_sysrem_resid.

    V_p = K_p sin(2π * φ) + Vsys - v_bary + v_max

    Parameters
    ----------
    cc_rvs: 1D float array
        Vector of RV steps of length [n_rv_step]
    
    cc_values: 3D float array
        3D float array of cross correlation results with shape:
        [n_sysrem_iter, n_phase, n_rv_step]
    
    transit_info: pandas DataFrame
        Pandas DataFrame with header/computed information about each timestep.

    syst_info: pandas DataFrame
        DataFrame containing planet/star/system properties.

    Kp_lims: float tuple, default: (0,400)
        Lower and upper bounds of Kp in km/s to sample between.

    Kp_step: float, default: 0.5
        Kp step size to use when sampling in km/s.
    """
    # Grab dimensions for convenience
    (n_sysrem_iter, n_phase, n_rv_step) = cc_values.shape

    # Initialise Kp sampling
    Kp_steps = np.arange(Kp_lims[0], Kp_lims[1], Kp_step)

    n_Kp_steps = len(Kp_steps)

    # Initialise output array
    Kp_vsys_map = np.zeros((n_sysrem_iter, n_Kp_steps, n_rv_step,))

    # Loop over all sets of sysrem residuals
    for sr_iter_i in range(n_sysrem_iter):
        desc = "Creating Kp-Vsys map for SYSREM iter #{}".format(sr_iter_i)

        # Loop over all Kp values
        for Kp_i, Kp in enumerate(tqdm(Kp_steps, leave=False, desc=desc)):
            # Initialise grid of shifted phases
            cc_values_shifted = np.zeros_like(cc_values[sr_iter_i])

            # Loop over all phases
            for phase_i in range(n_phase):
                # Grab relevant parameters from transit_info
                phase = transit_info.iloc[phase_i]["phase_mid"]
                v_bcor = transit_info.iloc[phase_i]["bcor"]
                v_star = syst_info.loc["rv_star", "value"]

                # Compute the velocity shift between this phase and the planet
                # rest frame velocity.
                vp = Kp * np.sin(2*np.pi*phase) + v_star - v_bcor

                # Create interpolator for the CC values of this phase
                interp_cc_vals = interp1d(
                    x=cc_rvs,
                    y=cc_values[sr_iter_i, phase_i],
                    bounds_error=False,
                    fill_value=np.nan,)

                # Shift CC value to the rest frame using calculated vp value
                cc_values_shifted[phase_i] = interp_cc_vals(cc_rvs - vp)

            # Sum all shifted CC values along the phase axis
            Kp_vsys_map[sr_iter_i, Kp_i] = \
                np.nanmean(cc_values_shifted, axis=0)

    return Kp_steps, Kp_vsys_map