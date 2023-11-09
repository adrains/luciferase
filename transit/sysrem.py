"""
"""
import numpy as np
from astropy.stats import sigma_clip
from tqdm import tqdm
from scipy.interpolate import interp1d
import astropy.constants as const

def clean_and_compute_initial_resid(
    spectra,
    e_spectra,
    bad_px_mask_init,
    sigma_threshold_phase=5.0,
    sigma_threshold_spectral=6.0,):
    """Function to clean a spectral data cube prior to running SYSREM. We sigma
    clip along the phase dimension (i.e. the time series of each pixel) as well
    as along the spectral dimension. Note that we only 
    
    Parameters
    ----------

    Returns
    -------
    """
    # Grab shape
    (n_phase, n_px) = spectra.shape

    # Duplicate arrays
    flux = spectra.copy()
    e_flux = e_spectra.copy()

    # Sigma clip along phase dimension. Note that here we're doing both upper
    # *and* lower bound clipping.
    # TODO: we eventually want to interpolate along the spectral dimension
    sc_mask_phase = np.full_like(bad_px_mask_init, False)

    for px_i in range(n_px):
        sc_mask_phase[:,px_i] = sigma_clip(
            data=flux[:,px_i],
            sigma=sigma_threshold_phase,).mask
        
    # Sigma clip along the spectral dimension. Note that since we're clipping
    # along the spectral dimension, we only clip the *upper* bounds so as to
    # not remove absorption features.
    sc_mask_spec = np.full_like(bad_px_mask_init, False)

    for phase_i in range(n_phase):
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
    tolerance=1E-6,
    max_converge_iter=100,
    diff_method="max",):
    """

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
    spectra,
    e_spectra,
    bad_px_mask,
    n_iter,
    tolerance=1E-6,
    max_converge_iter=100,
    diff_method="max"


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
        spectra, e_spectra, bad_px_mask)

    # Store median subtracted residuals
    resid_all[0] = resid_init

    # Run SYSREM
    for sr_iter_i in range(n_iter):
        print("-"*10, "Iter #{}".format(sr_iter_i+1), "-"*10, sep="\n")
        # Initialise vectors
        cc = np.zeros(n_px)     # gradient
        aa = np.ones(n_phase)   # airmasses

        # Grab a temporary handle for the current set of residuals
        resid = resid_all[sr_iter_i]

        # Iterate until convergence for cc an aa vectors
        converge_step_i = 0
        has_converged = False

        while converge_step_i < max_converge_iter and not has_converged:
            # Comnpute an estimate for c for converge_step_i
            c_num = np.nansum(aa * (resid / e_flux**2).T, axis=1,)
            c_den = np.nansum(aa ** 2 / e_flux.T**2, axis=1,)
            cc_est = np.divide(c_num, c_den,)

            # Compute an estimate for a at converge_step_i
            a_num = np.nansum(cc_est[:, None] * (resid / e_flux**2).T, axis=0,)
            a_den = np.nansum(cc_est[:, None] ** 2 / e_flux.T**2, axis=0,)
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
        resid_all[sr_iter_i+1] = resid - systematic.T

    # We've completed all iterations, return
    return resid_all


def cross_correlate_sysrem_resid(
    waves,
    sysrem_resid,
    template_wave,
    template_spec,
    cc_rv_step=1,
    cc_rv_lims=(-200,200),):
    """

    Parameters
    ----------
    waves_1d,
    sysrem_resid,
    template_wave,
    template_spec,
    cc_rv_step=1,
    cc_rv_lims=(-200,200),

    Returns
    -------
    """
    # Initialise RV vector
    cc_rvs = np.arange(cc_rv_lims[0], cc_rv_lims[1]+cc_rv_step, cc_rv_step)

    # Grab dimensions for convenience
    (n_sysrem_iter, n_phase, n_spec, n_px) = sysrem_resid.shape
    n_rv_steps = len(cc_rvs)

    # Intiialise output array
    cc_values = np.zeros((n_sysrem_iter, n_phase, n_spec, n_rv_steps))

    # Initialise template spectrum interpolator
    temp_interp = interp1d(
        x=template_wave,
        y=template_spec,
        bounds_error=False,
        fill_value=np.nan,)

    print("Cross correlating...")

    # Loop over each set of residuals for each SYSREM iteration
    for sysrem_iter_i in range(n_sysrem_iter):

        desc = "Cross Correlating for SYSREM iter #{}".format(sysrem_iter_i)

        # Loop over all spectral segments
        for spec_i in tqdm(range(n_spec), leave=False, desc=desc):
            # Loop over all RVs and cross correlate against each phase
            # Note: all phases need the same cross correlation to occur, so we can
            # save on interpolation/loops by just interpolating once per RV.
            for rv_i, rv in enumerate(cc_rvs):
                # Doppler shift for new wavelength scale
                wave_rv_shift = waves[spec_i] * (1- rv/(const.c.si.value/1000))

                # Interpolate to wavelength scale
                tspec_rv_shift = temp_interp(wave_rv_shift)

                # Tile this to all phases
                tspec_tiled = np.tile(tspec_rv_shift, n_phase).reshape(
                    (n_phase, n_px))

                #import pdb
                #pdb.set_trace()

                # Cross correlate
                cc_values[sysrem_iter_i, :, spec_i, rv_i] = \
                    np.nansum(sysrem_resid[sysrem_iter_i, :, spec_i] * tspec_tiled, axis=1)

    # Normalise by median along rv dimension
    #cc_medians = np.nanmedian(cc_values, axis=2)
    #cc_medians_tiled = np.tile(cc_medians, n_rv_steps)
    #cc_medians_final = np.moveaxis(
    #    cc_medians_tiled.reshape((n_sysrem_iter,n_rv_steps,n_phase)), 1, 2)
    
    #cc_vals_norm = cc_values / cc_medians_final

    # All done, return our array of cross correlation results
    return cc_rvs, cc_values


def compute_Kp_vsys_map(
    cc_rvs,
    cc_values,
    transit_info,
    syst_info,
    Kp_lims=(0,400),
    Kp_step=0.5,):
    """

    V_p = K_p sin(2π * φ) + Vsys - v_bary + v_max

    Parameters
    ----------
    cc_rvs: 1D float array
        Vector of RV steps of length [n_rv_step]
    
    cc_values: 3D float array
        3D float array of cross correlation results with shape:
        [n_sysrem_iter, n_phase, n_rv_step]
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