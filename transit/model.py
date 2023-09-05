"""
Functions to implement a transit model using the Aaronson Method.
 - Aronson, Walden, & Piskunov 2015
  https://ui.adsabs.harvard.edu/abs/2015A%26A...578A.133A/abstract
  https://ui.adsabs.harvard.edu/abs/2015A%26A...581C...1A/abstract (erratum)

Originally written in IDL by Nikolai Piskunov, ported to Python by Adam Rains.
"""
import numpy as np
from tqdm import tqdm
from scipy import linalg, sparse
from transit import utils as tu
from transit import plotting as mplt

# -----------------------------------------------------------------------------
# Observable Computation Functions (flux, tau, trans, scaling, ldc)
# -----------------------------------------------------------------------------
def do_tikhonov_regularise(b, r, lambda_reg, n_wave):
    """Function to implement Tikhonov regularization. TODO: properly test.

    Parameters
    ----------
    b: 1D float array
        Float array TODO of shape [n_wave].

    r: 1D float array
        Float array TODO of shape [n_wave].

    lambda_reg: float
        Tikhonov regularisation parameter lambda.

    Returns
    -------
    spectrum: 1D float array
        Smoothed/regularised spectrum of shape [n_wave].
    """
    # Construct 2D tridiagonal matrix of shape [n_wave, n_wave] from
    # array b. This new matrix will have values only along the super-, 
    # main, and sub-diagonals. This changes what was a simple division
    # into a matrix algebra problem, so instead of being able to
    # directly calculate the planet transmission we must now solve for
    # it. Two ways to think about this:
    #   1) Formally, we've converted the problem to a Tikhonov 
    #      regularization problem. TODO: explain why matrix is the way
    #      it is, etc.
    #   2) Multiplying by a tridiagonal matrix has the effect of
    #      reducing noise, analogous to a low-pass filter. See:
    #      https://math.stackexchange.com/questions/3634766/
    #      why-does-tridiagonal-matrix-reduce-noise.
    off_diag = np.full(shape=n_wave, fill_value=-lambda_reg)
    diag = b.copy()
    diag[0] += lambda_reg
    diag[1:-1] += 2*lambda_reg      # TODO: Ensure this is the same for all 3 times we call this
    diag[-1] += lambda_reg
    bb = sparse.diags(
        diagonals=(off_diag, diag, off_diag),
        offsets=(1,0,-1),
        shape=(n_wave, n_wave),).A
    
    # Compute pivoted LU decomposition of our tridiagonal matrix
    bb_lu, bb_piv = linalg.lu_factor(bb)

    # Solve the matrix equation bb*P = r for P, where P is the planet.
    spectrum = linalg.lu_solve((bb_lu, bb_piv), r)

    return spectrum


def update_stellar_flux(
    waves,
    obs_spec,
    obs_spec_2,
    flux,
    flux_2,
    tau,
    tau_2,
    trans,
    trans_2,
    scale,
    mask,
    mask_2,
    transit_info,
    syst_info,
    lambda_treg,
    flux_limits,):
    """
    Updates stellar flux and its derivative in the *stellar* frame. Note that
    this function uses the appropriate tau vector for the transit in question.

    Parameters
    ----------
    waves: float array
        Wavelength scales for each spectral segment of shape [n_spec, n_px].
    
    obs_spec, obs_spec_2: 2D float array
       3D float array of observed spectra and its derivative of shape
       [n_phase, n_spec, n_px].
    
    flux, flux_2: 2D float array
        Fitted model stellar flux and its derivative of shape [n_spec, n_px].

    tau, tau_2: 2D float array
        Fitted model telluric tau and its derivative of shape [n_spec, n_px].

    trans, trans_2: 2D float array
        Fitted model planet transmission and its derivative of shape 
        [n_spec, n_px].

    scale: 1D float array
        Fitted model scale parameter of shape [n_phase].

    mask, mask_2: 3D float array
        Mask array and its derivative of shape [n_phase, n_spec, n_px]. 
        Contains either 0 or 1. TODO: derivative?

    transit_info: pandas DataFrame
        Transit info DataFrames containing information associated with each 
        transit time step with columns:

        ['mjd_start', 'mjd_mid', 'mjd_end', 'jd_start', 'jd_mid', 'jd_end',
         'airmass', 'bcor', 'hcor', 'ra', 'dec', 'exptime_sec', 'nod_pos',
         'raw_file', 'phase_start', 'is_in_transit_start', 'r_x_start',
         'r_y_start', 'r_z_start', 'v_x_start', 'v_y_start', 'v_z_start',
         's_projected_start', 'scl_start', 'mu_start',
         'planet_area_frac_start', 'phase_mid', 'is_in_transit_mid',
         'r_x_mid', 'r_y_mid', 'r_z_mid', 'v_x_mid', 'v_y_mid', 'v_z_mid',
         's_projected_mid', 'scl_mid', 'mu_mid', 'planet_area_frac_mid',
         'phase_end', 'is_in_transit_end', 'r_x_end', 'r_y_end', 'r_z_end',
         'v_x_end', 'v_y_end', 'v_z_end', 's_projected_end', 'scl_end',
         'mu_end', 'planet_area_frac_end', 'gamma', 'beta', 'delta']

    syst_info: pandas DataFrame
        DataFrame containing planet/star/system properties. The data frame has
        columns ['value', 'sigma', 'reference', 'comment'] and indices:
        ['m_star_msun', 'r_star_rsun', 'k_star_mps', 'dist_pc', 'vsini',
         'rv_star', 'ldc_init_a1', 'ldc_init_a2', 'ldc_init_a3', 'ldc_init_a4',
         'a_planet_au', 'e_planet', 'i_planet_deg', 'omega_planet_deg',
         'w_planet_deg', 'k_star_mps', 'transit_dur_hours', 'jd0_days', 
         'period_planet_days', 'm_planet_mearth', 'r_planet_rearth', 
         'r_planet_atmo_r_earth'].

    lambda_treg: float or None
        Tikhonov regularisation parameter lambda. Defaults to None, in
        which case no regularisation is applied.

    flux_limits: float/None tuple
        Lower and upper limits to enforce for the modelled stellar flux as
        (flux_low, flux_high) where a value of None for either the lower or
        upper limit is used to not apply a limit.

    Updated
    -------
    flux, flux_2: 2D float array
        Fitted model stellar flux and its derivative of shape [n_spec, n_px].
    """
    # -------------------------------------------------------------------------
    # Initialise variable references for convenience
    # -------------------------------------------------------------------------
    n_phase = obs_spec.shape[0]
    n_spec = obs_spec.shape[1]
    n_wave = obs_spec.shape[2]

    transit_nums = transit_info["transit_num"].values

    ldc_cols = ["ldc_init_a1", "ldc_init_a2", "ldc_init_a3", "ldc_init_a4",]
    a_limb = syst_info.loc[ldc_cols, "value"].values

    mus = transit_info["mu_mid"].values
    airmasses = transit_info["airmass"].values

    area_frac = transit_info["planet_area_frac_mid"].values

    # Grab our RVs for convenience
    gamma = transit_info["gamma"].values       # gamma = rv_star + rv_bcor
    beta = transit_info["beta"].values         # beta = gamma + rv_projected
    delta = transit_info["delta"].values       # delta = gamma + rv_planet

    # -------------------------------------------------------------------------
    # Loop over all spectral segments
    # -------------------------------------------------------------------------
    # Pre-compute the limb darkening function:
    # sum [a_k * mu^(k/2)] / sum [2*a_k / (k + 4)]
    # TODO: is there a factor of 2 missing here?
    limb = np.zeros(n_phase)
    ipow = np.arange(len(a_limb))+1
    N = np.pi*(1-np.sum(ipow*a_limb/(ipow+4)))
    for phase_i in range(n_phase):
        limb[phase_i] = ((1 - np.sum(a_limb*(1 - np.sqrt(mus[phase_i]**ipow))))
                         / N * area_frac[phase_i])

    for spec_i in range(0, n_spec):
        #a = np.zeros(n_wave-1)
        b = np.zeros(n_wave)
        #c = np.zeros(n_wave-1)
        r = np.zeros(n_wave)

        for phase_i in range(0, n_phase):
            # RV shift tellurics into stellar rv frame, making sure to use the
            # tellurics from the appropriate transit.
            transit_i = transit_nums[phase_i]

            t = tu.doppler_shift(
                x=waves[spec_i],
                y=tau[transit_i, spec_i],
                gamma=gamma[phase_i],
                y2=tau_2[transit_i, spec_i],)

            # RV shift mask into stellar rv frame
            m = tu.doppler_shift(
                x=waves[spec_i],
                y=mask[phase_i, spec_i],
                gamma=gamma[phase_i],
                y2=mask_2[phase_i,spec_i],)

            # RV shift observations into stellar rv frame
            o = tu.doppler_shift(
                x=waves[spec_i],
                y=obs_spec[phase_i,spec_i,],
                gamma=gamma[phase_i],
                y2=obs_spec_2[phase_i,spec_i,],)

            # RV shift planet transmission into stellar rv frame
            p = tu.doppler_shift(
                x=waves[spec_i],
                y=trans[spec_i],
                gamma=gamma[phase_i]-delta[phase_i],
                y2=trans_2[spec_i],)

            # Calculate equation 26, flux = E / G = r / b
            b += (1-limb[phase_i]*p) * scale[phase_i]**2 \
                * np.exp(-2*t*airmasses[phase_i]) * m
            r += o * scale[phase_i] * np.exp(-t*airmasses[phase_i]) * m

        # Calculate stellar spectrum *with* regularisation (i.e. smoothing)
        if lambda_treg is not None:
            new_flux = do_tikhonov_regularise(b, r, lambda_treg, n_wave)
        
        # Calculate stellar spectrum *without* regularisation
        else:
            new_flux = r/b

        # [Optional] Clip fluxes
        if flux_limits[0] is not None or flux_limits[1] is not None:
            new_flux[:] = np.clip(
                a=new_flux,
                a_min=flux_limits[0],
                a_max=flux_limits[1],)
        
        # Update flux by reference
        flux[spec_i] = new_flux

        # Update flux derivative
        flux_2[spec_i] = tu.bezier_init(waves[spec_i], flux[spec_i])


def init_tau_newton_raphson(
    obs_spec,
    model,
    tau,
    mask,
    transit_info,
    telluric_trans_limits,):
    """Initialise tau before running the Newton-Raphson method. This function
    should be called before update_tau_newton_raphson for the first time.

    Parameters
    ----------
    obs_spec: 2D float array
       3D float array of observed spectra of shape [n_phase, n_spec, n_px].

    model: 3D float array
        Combined model (star + tellurics + planet) at each phase of shape 
        [n_phase, n_spec, n_px].

    tau: 3D float array
        Fitted model telluric tau of shape [n_transit, n_spec, n_px].

    mask: 3D float array
        Mask array of shape [n_phase, n_spec, n_px]. Contains either 0 or 1.

    transit_info: pandas DataFrame
        Transit info DataFrames containing information associated with each 
        transit time step with columns:

        ['mjd_start', 'mjd_mid', 'mjd_end', 'jd_start', 'jd_mid', 'jd_end',
         'airmass', 'bcor', 'hcor', 'ra', 'dec', 'exptime_sec', 'nod_pos',
         'raw_file', 'phase_start', 'is_in_transit_start', 'r_x_start',
         'r_y_start', 'r_z_start', 'v_x_start', 'v_y_start', 'v_z_start',
         's_projected_start', 'scl_start', 'mu_start',
         'planet_area_frac_start', 'phase_mid', 'is_in_transit_mid',
         'r_x_mid', 'r_y_mid', 'r_z_mid', 'v_x_mid', 'v_y_mid', 'v_z_mid',
         's_projected_mid', 'scl_mid', 'mu_mid', 'planet_area_frac_mid',
         'phase_end', 'is_in_transit_end', 'r_x_end', 'r_y_end', 'r_z_end',
         'v_x_end', 'v_y_end', 'v_z_end', 's_projected_end', 'scl_end',
         'mu_end', 'planet_area_frac_end', 'gamma', 'beta', 'delta']

    telluric_trans_limits: float/None tuple
        Lower and upper limits to enforce for modelled telluric transmission
        (trans_low, trans_high) where a value of None for either the lower or
        upper limit is used to not apply a limit.

    Updated
    -------
    tau: 2D float array
        Fitted model telluric tau and its derivative of shape [n_spec, n_px].

    model: 3D float array
        Combined model (star + tellurics + planet) at each phase of shape 
        [n_phase, n_spec, n_px].
    """
    # -------------------------------------------------------------------------
    # Initialise for convenience
    # -------------------------------------------------------------------------
    n_transit = tau.shape[0]
    n_spec = obs_spec.shape[1]
    n_wave = obs_spec.shape[2]

    # Masks to reference obs_spec/model/transit info per transit
    night_masks = [transit_info["transit_num"].values == trans_i
        for trans_i in range(n_transit)]
    
    # Number of phases per night
    n_phase_nightly = [np.sum(nm) for nm in night_masks]

    airmasses = transit_info["airmass"].values

    # -------------------------------------------------------------------------
    # Loop over all transits
    # -------------------------------------------------------------------------
    for transit_i in range(0, n_transit):
        # Grab mask and n_phase for this transit
        nm = night_masks[transit_i]
        n_phase = n_phase_nightly[transit_i]

        # Expand airmass to all wavelengths with shape [n_phase, n_wave]
        am_all = np.tile(airmasses[nm], n_wave).reshape((n_wave, n_phase)).T

        # ---------------------------------------------------------------------
        # Loop over all spectral segments
        # ---------------------------------------------------------------------
        for spec_i in range(0, n_spec):
            # Pull out subarrays for convenience
            obs_spec_n = obs_spec[nm, spec_i]
            model_n = model[nm, spec_i]
            tau_n = tau[transit_i, spec_i]
            mask_n = mask[nm, spec_i]

            # Expand tau to all phases with shape [n_phase, n_wave]
            tau_n_all = np.tile(tau_n, n_phase).reshape((n_phase, n_wave))

            # Initial guess, assuming all Z eq 1, sum over all phases (axis=0)
            # TODO: should we be assuming this is one?
            #   - if we're coming here initially, we've already assumed z=1
            a = np.sum(
                mask_n*model_n*obs_spec_n/np.exp(-tau_n_all*am_all),axis=0)
            b = np.sum(mask_n*model_n**2/np.exp(-2*tau_n_all*am_all), axis=0)

            ibad = np.where(b == 0)
            nbad = np.sum(ibad)

            if nbad > 0:
                b[ibad] = 1

            # Calculate (and clip) tellurics to calculcate a new value for tau
            tellurics = a/b
            clipped_tellurics = np.clip(
                a=tellurics/np.max(tellurics),
                a_min=telluric_trans_limits[0],
                a_max=telluric_trans_limits[1],)
            
            tau_n_new = -np.log(clipped_tellurics) / np.min(airmasses)
            
            # Expand delta tau to all wavelengths with shape [n_phase, n_wave]
            delta_tau = np.tile(
                tau_n-tau_n_new, n_phase).reshape((n_phase, n_wave))

            # Calculate updated model
            model_n_new = model_n * np.exp(delta_tau*am_all)

            # Update (by reference) tau and model vectors
            tau[transit_i, spec_i] = tau_n_new
            model[nm, spec_i] = model_n_new


def update_tau_newton_raphson(
    obs_spec,
    model,
    tau,
    mask,
    transit_info,
    lambda_treg,
    tolerance,
    telluric_tau_limits,
    max_tau_nr_iter=1E20,):
    """
    Iteratively update tau (and the model) until the difference between 
    subsequent updates is below the adopted tolerance. This is done per transit
    per spectral segment.
    
    M = [F - I*P] e^(-tau * Z) * S

    e^(-tau^0_lambda) = [sum(F-I*P)^2 * S^2] / [sum(F-I*P)*S*O]

    TODO: should this also update the tau second derivative once tau has
    converged?

    Parameters
    ----------
    obs_spec: 2D float array
       3D float array of observed spectra of shape [n_phase, n_spec, n_px].

    model: 3D float array
        Combined model (star + tellurics + planet) at each phase of shape 
        [n_phase, n_spec, n_px].

    tau: 2D float array
        Fitted model telluric tau and its derivative of shape [n_spec, n_px].

    mask: 3D float array
        Mask array of shape [n_phase, n_spec, n_px]. Contains either 0 or 1.

    transit_info: pandas DataFrame
        Transit info DataFrames containing information associated with each 
        transit time step with columns:

        ['mjd_start', 'mjd_mid', 'mjd_end', 'jd_start', 'jd_mid', 'jd_end',
         'airmass', 'bcor', 'hcor', 'ra', 'dec', 'exptime_sec', 'nod_pos',
         'raw_file', 'phase_start', 'is_in_transit_start', 'r_x_start',
         'r_y_start', 'r_z_start', 'v_x_start', 'v_y_start', 'v_z_start',
         's_projected_start', 'scl_start', 'mu_start',
         'planet_area_frac_start', 'phase_mid', 'is_in_transit_mid',
         'r_x_mid', 'r_y_mid', 'r_z_mid', 'v_x_mid', 'v_y_mid', 'v_z_mid',
         's_projected_mid', 'scl_mid', 'mu_mid', 'planet_area_frac_mid',
         'phase_end', 'is_in_transit_end', 'r_x_end', 'r_y_end', 'r_z_end',
         'v_x_end', 'v_y_end', 'v_z_end', 's_projected_end', 'scl_end',
         'mu_end', 'planet_area_frac_end', 'gamma', 'beta', 'delta']

    lambda_treg: float or None
        Tikhonov regularisation parameter lambda. Set to None for no smoothing.

    tolerance: float, default: 1E-5
        The tolerance value used to determine when the fit has reached a
        satisfactory level of precision.

    telluric_tau_limits: float/None tuple
        Lower and upper limits to enforce for the modelled telluric tau array
        (tau_low, tau_high) where a value of None for either the lower or
        upper limit is used to not apply a limit.
    
    max_tau_nr_iter: int, default: 1E20
        Maximum number of Newton's method iterations.

    Updated
    -------
    tau: 2D float array
        Fitted model telluric tau and its derivative of shape [n_spec, n_px].

    model: 3D float array
        Combined model (star + tellurics + planet) at each phase of shape 
        [n_phase, n_spec, n_px].
    """
    # -------------------------------------------------------------------------
    # Initialise for convenience
    # -------------------------------------------------------------------------
    n_transit = tau.shape[0]
    n_spec = obs_spec.shape[1]
    n_wave = obs_spec.shape[2]

    # Masks to reference obs_spec/model/transit info per transit
    night_masks = [transit_info["transit_num"].values == trans_i
        for trans_i in range(n_transit)]
    
    # Number of phases per night
    n_phase_nightly = [np.sum(nm) for nm in night_masks]

    airmasses = transit_info["airmass"].values

    total_iterations = 0

    # -------------------------------------------------------------------------
    # Loop over all transits
    # -------------------------------------------------------------------------
    for transit_i in range(0, n_transit):
        # Grab mask and n_phase for this transit
        nm = night_masks[transit_i]
        n_phase = n_phase_nightly[transit_i]

        # Expand airmass to all wavelengths with shape [n_phase, n_wave]
        am_all = np.tile(airmasses[nm], n_wave).reshape((n_wave, n_phase)).T

        # ---------------------------------------------------------------------
        # Loop over all spectral segments
        # ---------------------------------------------------------------------
        for spec_i in range(0, n_spec):
            # Pull out subarrays for convenience
            obs_spec_n = obs_spec[nm, spec_i]
            mask_n = mask[nm, spec_i]

            # -----------------------------------------------------------------
            # Iterate with Newton's method until tau converges for this segment
            # -----------------------------------------------------------------
            iter_count = 0
            has_converged = False

            print("")   # Empty line to separate segments
            
            while (not has_converged) and (iter_count < max_tau_nr_iter):
                # Pull out sub-arrays for convenience
                model_n = model[nm, spec_i]
                tau_n = tau[transit_i, spec_i]

                # Calculate num/denominator, sum along phase axis (axis 0)
                b = np.sum(
                    mask_n*(2*model_n-obs_spec_n)*model_n*am_all**2, axis=0)
                r = np.sum(mask_n*(model_n-obs_spec_n)*model_n*am_all, axis=0)

                # Calculate delta_tau as:
                #   delta tau = [sum(M-O) * M * z] / [sum(2M - O)*M * z^2]
                # *With* regularisation
                if lambda_treg is not None:
                    delta_tau = do_tikhonov_regularise(
                        b, r, lambda_treg, n_wave)

                # Or *without* regulatisation
                else:
                    delta_tau = r/b

                # Store old value of tau
                tau_n_old = tau_n.copy()

                # Clip tau to limits, calculate updated tau
                tau_n_new = np.clip(
                    a=tau_n_old+delta_tau,
                    a_min=telluric_tau_limits[0],
                    a_max=telluric_tau_limits[1],)

                # Print iteration updates
                delta_tau_clipped = tau_n_old - tau_n_new
                med_delta_tau = np.median(np.abs(delta_tau_clipped))
                std_delta_tau = np.std(delta_tau_clipped)
                sum_delta_tau = np.sum(delta_tau_clipped)
                max_delta_tau = np.max(np.abs(delta_tau_clipped))

                print(
                    "\tTransit #{:2}\t".format(transit_i),
                    "\tSegment #{:2}\t".format(spec_i),
                    "iteration #{:4}\t".format(iter_count),
                    "| median Δ | = {:18.14f}\t".format(med_delta_tau),
                    "| max Δ | = {:18.14f}\t".format(max_delta_tau),
                    "sum Δ = {:20.14f}\t".format(sum_delta_tau),
                    "std Δ = {:18.14f}\t".format(std_delta_tau),)

                # Calculcate updated model
                model_n_new = model_n * np.exp(delta_tau_clipped * am_all)

                # Update tau and model by reference
                tau[transit_i, spec_i] = tau_n_new
                model[nm, spec_i] = model_n_new

                # Finally, check convergence
                max_resid = np.nanmax(np.abs(delta_tau_clipped))

                if np.isnan(max_resid):
                    raise Exception(
                        "Maximum residual is nan for iter {}".format(
                        iter_count))
                if max_resid < tolerance:
                    has_converged = True
                else:
                    iter_count += 1

            if iter_count >= max_tau_nr_iter:
                print("\tHit max iterations, continuing")

            total_iterations += iter_count + 1

    print("\n\tConverged with {} iterations\n".format(total_iterations))


def update_transmission(
    waves,
    obs_spec,
    obs_spec_2,
    flux,
    flux_2,
    tau,
    tau_2,
    trans,
    trans_2,
    scale,
    transit_info,
    syst_info,
    lambda_treg,
    trans_limits,):
    """
    Updates the exoplanet transmission and its derivative in-place. To do this,
    we shift things into the planet rv frame.

    Parameters
    ----------
    waves: float array
        Wavelength scales for each spectral segment of shape [n_spec, n_px].
    
    obs_spec, obs_spec_2: 2D float array
       3D float array of observed spectra and its derivative of shape
       [n_phase, n_spec, n_px].
    
    flux, flux_2: 2D float array
        Fitted model stellar flux and its derivative of shape [n_spec, n_px].

    tau, tau_2: 2D float array
        Fitted model telluric tau and its derivative of shape [n_spec, n_px].

    trans, trans_2: 2D float array
        Fitted model planet transmission and its derivative of shape 
        [n_spec, n_px].

    scale: 1D float array
        Fitted model scale parameter of shape [n_phase].

    transit_info: pandas DataFrame
        Transit info DataFrames containing information associated with each 
        transit time step with columns:

        ['mjd_start', 'mjd_mid', 'mjd_end', 'jd_start', 'jd_mid', 'jd_end',
         'airmass', 'bcor', 'hcor', 'ra', 'dec', 'exptime_sec', 'nod_pos',
         'raw_file', 'phase_start', 'is_in_transit_start', 'r_x_start',
         'r_y_start', 'r_z_start', 'v_x_start', 'v_y_start', 'v_z_start',
         's_projected_start', 'scl_start', 'mu_start',
         'planet_area_frac_start', 'phase_mid', 'is_in_transit_mid',
         'r_x_mid', 'r_y_mid', 'r_z_mid', 'v_x_mid', 'v_y_mid', 'v_z_mid',
         's_projected_mid', 'scl_mid', 'mu_mid', 'planet_area_frac_mid',
         'phase_end', 'is_in_transit_end', 'r_x_end', 'r_y_end', 'r_z_end',
         'v_x_end', 'v_y_end', 'v_z_end', 's_projected_end', 'scl_end',
         'mu_end', 'planet_area_frac_end', 'gamma', 'beta', 'delta']

    syst_info: pandas DataFrame
        DataFrame containing planet/star/system properties. The data frame has
        columns ['value', 'sigma', 'reference', 'comment'] and indices:
        ['m_star_msun', 'r_star_rsun', 'k_star_mps', 'dist_pc', 'vsini',
         'rv_star', 'ldc_init_a1', 'ldc_init_a2', 'ldc_init_a3', 'ldc_init_a4',
         'a_planet_au', 'e_planet', 'i_planet_deg', 'omega_planet_deg',
         'w_planet_deg', 'k_star_mps', 'transit_dur_hours', 'jd0_days', 
         'period_planet_days', 'm_planet_mearth', 'r_planet_rearth', 
         'r_planet_atmo_r_earth'].

    lambda_treg: float or None
        Tikhonov regularisation parameter lambda. Set to None for no smoothing.

    trans_limits: float/None tuple
        Lower and upper limits to enforce for the modelled planet transmission
        (trans_low, trans_high) where a value of None for either the lower or
        upper limit is used to not apply a limit.

    Updated
    -------
    trans, trans_2: 2D float array
        Fitted model planet transmission and its derivative of shape 
        [n_spec, n_px].
    """
    # -------------------------------------------------------------------------
    # Grab parameters for convenience
    # -------------------------------------------------------------------------
    n_phase = obs_spec.shape[0]
    n_spec = obs_spec.shape[1]
    n_wave = obs_spec.shape[2]

    ldc_cols = ["ldc_init_a1", "ldc_init_a2", "ldc_init_a3", "ldc_init_a4",]
    a_limb = syst_info.loc[ldc_cols, "value"].values

    mus = transit_info["mu_mid"].values
    airmasses = transit_info["airmass"].values
    
    transit_nums = transit_info["transit_num"].values

    # gamma [star]    = (v_bary + v_star) / c
    # beta  [shadow]  = (v_bary + v_star + vsini * x_planet) / c
    # delta [planet]  = (v_bary + v_star + v^i_planet) / c
    gamma = transit_info["gamma"].values
    beta = transit_info["beta"].values
    delta = transit_info["delta"].values

    area_frac = transit_info["planet_area_frac_mid"].values

    # -------------------------------------------------------------------------
    # Precalculate limb darkening
    # -------------------------------------------------------------------------
    n_limb = len(a_limb)
    ipow = np.arange(n_limb) + 1
    N = np.pi*(1-np.sum(ipow*a_limb/(ipow+4)))

    in_transit = transit_info["is_in_transit_mid"].values
    phase_i_min = np.min(np.arange(n_phase)[in_transit])
    phase_i_max = np.max(np.arange(n_phase)[in_transit])

    # -------------------------------------------------------------------------
    # Loop over all spectral segments
    # -------------------------------------------------------------------------
    for spec_i in range(0, n_spec):
        # Initialise vectors for this spectral segment
        r = np.zeros(n_wave)
        b = np.zeros(n_wave)

        # Only loop over the phases that are mid-transit
        for phase_i in range(phase_i_min, phase_i_max+1):
            # Shift observed spectrum to planet rv frame (delta) from telescope
            # frame (rest frame)
            o = tu.doppler_shift(
                x=waves[spec_i],
                y=obs_spec[phase_i, spec_i],
                gamma=delta[phase_i],
                y2=obs_spec_2[phase_i, spec_i])

            # Shift model flux vector to planet rv frame (delta) from stellar
            # frame (gamma)
            f1 = tu.doppler_shift(
                x=waves[spec_i],
                y=flux[spec_i],
                gamma=-gamma[phase_i]+delta[phase_i],
                y2=flux_2[spec_i],)

            # Shift model flux shadow vector to planet rv frame (delta) from
            # shadow frame (beta)
            f2 = tu.doppler_shift(
                x=waves[spec_i],
                y=flux[spec_i],
                gamma=-beta[phase_i]+delta[phase_i],
                y2=flux_2[spec_i])

            # Apply limb darkening to the shadow
            if in_transit[phase_i]:
                intens = (
                    f2 * (1-np.sum(a_limb*(1-np.sqrt(mus[phase_i]**ipow)))) / N
                    * area_frac[phase_i])
            else:
                intens = np.zeros(n_wave)

            # Shift telluric vector to planet rv frame (delta) from telescope
            # frame (rest frame), making sure to use the telluric vector for
            # the appropriate transit.
            transit_i = transit_nums[phase_i]

            t = tu.doppler_shift(
                x=waves[spec_i],
                y=tau[transit_i, spec_i],
                gamma=delta[phase_i],
                y2=tau_2[transit_i, spec_i],)

            # Calculate the numerator (r) and denominator (b) of equation 12
            # from the Aronson paper.
            t = np.exp(-t*airmasses[phase_i])*scale[phase_i]
            r += (f1*t - o)*intens*t
            b += intens**2 * t**2

        # TODO: no idea what this means?
        #if lambda_treg is not None:
        #    r += lambda_treg * syst_info.loc["rp_rstar", "value"]

        # Ensure r is positive
        r[r < 0] = 0

        # Calculate planet transmission *with* regularisation (i.e. smoothing)
        if lambda_treg is not None:
            new_trans = do_tikhonov_regularise(b, r, lambda_treg, n_wave)
        
        # Calculate planet transmission *without* regularisation
        else:
            new_trans = r/b

        # [Optional] Enforce limits
        if trans_limits[0] is not None or trans_limits[1] is not None:
            new_trans[:] = np.clip(
                a=new_trans,
                a_min=trans_limits[0],
                a_max=trans_limits[1],)
            
        # Update transmission by reference
        trans[spec_i] = new_trans

        # Update the derivative
        trans_2[spec_i] = tu.bezier_init(waves[spec_i], trans[spec_i])


def update_scaling(
    model,
    obs_spec,
    scale,
    scale_limits,):
    """
    Updates our scaling vector and the main model.

    S_j = sum (O^j_lambda * M^j_lambda) / M^2_j

    Note that since this an iterative problem, we use our old value of S_j to
    scale M^j_lambda.

    Per new paper, a and b are quadratic polynomial coefficients. TODO: expand
    on this/generalise so that we can use an arbitrary polynomial. For the case
    of multiple transits, this will need be called twice since the scaling from
    night-to-night will be decoupled.

    Parameters
    ----------
    model: 3D float array
        Combined model (star + tellurics + planet) at each phase of shape 
        [n_phase, n_spec, n_px].

    obs_spec, obs_spec_2: 2D float array
       3D float array of observed spectra and its derivative of shape
       [n_phase, n_spec, n_px].

    scale_old: 1D float array
        Old model scale parameter of shape [n_phase].

    scale_limits: float/None tuple
        Lower and upper limits to enforce for the modelled scale vector
        (scale_low, scale_high) where a value of None for either the lower or
        upper limit is used to not apply a limit.

    Updated
    -------
    scale: 1D float array
        Updated model scale parameter of shape [n_phase].

    model: 3D float array
        Combined model (star + tellurics + planet) at each phase of shape 
        [n_phase, n_spec, n_px].
    """
    # Initialisation:
    n_phase = obs_spec.shape[0]
    n_spec = obs_spec.shape[1]
    n_wave = obs_spec.shape[2]

    a = np.zeros(n_phase)
    b = np.zeros(n_phase)

    for phase_i in range(n_phase):
        # Scale model vector at this phase by old scaling param
        scaled_model = model[phase_i] / scale[phase_i]

        a[phase_i] += np.sum(scaled_model*obs_spec[phase_i])
        b[phase_i] += np.sum(scaled_model**2)

    # Calculcate the new scale vector
    scale_old = scale.copy()
    scale_new = a/b

    # Normalize to avoid degeneracy between flux and scaling
    scale_new /= np.max(scale_new)

    # [Optional] Enforce scale limits
    if scale_limits[0] is not None or scale_limits[1] is not None:
        scale_new = np.clip(
            a=scale_new, a_min=scale_limits[0], a_max=scale_limits[1])

    # Update scale by reference
    scale[:] = scale_new

    # Compute the fractional change in scale to update the model
    scale_fac = scale_new / scale_old

    # Tile this to shape [n_phase, n_spec, n_wave], update model by reference
    sf = np.tile(scale_fac, n_spec*n_wave).reshape((n_wave, n_spec, n_phase)).T
    model *= sf


# -----------------------------------------------------------------------------
# Modelling functions
# -----------------------------------------------------------------------------
def create_transit_model_array(
    waves,
    flux,
    flux_2,
    tau,
    trans,
    trans_2,
    scale,
    transit_info,
    syst_info,
    model_limits,):
    """Compute the combined Aronson model from the model stellar flux, planet
    absorption, and telluric absorption (and their derivatives) for use when
    directly comparing with the observed data. To do this, each component needs
    to be shifted to its respective velocity frame.

   Parameters
    ----------
    waves: float array
        Wavelength scales for each spectral segment of shape [n_spec, n_px].
    
    flux, flux_2: 2D float array
        Fitted model stellar flux and its derivative of shape [n_spec, n_px].

    tau: 2D float array
        Fitted model telluric tau of shape [n_spec, n_px].

    trans, trans_2: 2D float array
        Fitted model planet transmission and its derivative of shape 
        [n_spec, n_px].

    scale: 1D float array
        Fitted model scale parameter of shape [n_phase].

    transit_info: pandas DataFrame
        Transit info DataFrames containing information associated with each 
        transit time step with columns:

        ['mjd_start', 'mjd_mid', 'mjd_end', 'jd_start', 'jd_mid', 'jd_end',
         'airmass', 'bcor', 'hcor', 'ra', 'dec', 'exptime_sec', 'nod_pos',
         'raw_file', 'phase_start', 'is_in_transit_start', 'r_x_start',
         'r_y_start', 'r_z_start', 'v_x_start', 'v_y_start', 'v_z_start',
         's_projected_start', 'scl_start', 'mu_start',
         'planet_area_frac_start', 'phase_mid', 'is_in_transit_mid',
         'r_x_mid', 'r_y_mid', 'r_z_mid', 'v_x_mid', 'v_y_mid', 'v_z_mid',
         's_projected_mid', 'scl_mid', 'mu_mid', 'planet_area_frac_mid',
         'phase_end', 'is_in_transit_end', 'r_x_end', 'r_y_end', 'r_z_end',
         'v_x_end', 'v_y_end', 'v_z_end', 's_projected_end', 'scl_end',
         'mu_end', 'planet_area_frac_end', 'gamma', 'beta', 'delta']

    syst_info: pandas DataFrame
        DataFrame containing planet/star/system properties. The data frame has
        columns ['value', 'sigma', 'reference', 'comment'] and indices:
        ['m_star_msun', 'r_star_rsun', 'k_star_mps', 'dist_pc', 'vsini',
         'rv_star', 'ldc_init_a1', 'ldc_init_a2', 'ldc_init_a3', 'ldc_init_a4',
         'a_planet_au', 'e_planet', 'i_planet_deg', 'omega_planet_deg',
         'w_planet_deg', 'k_star_mps', 'transit_dur_hours', 'jd0_days', 
         'period_planet_days', 'm_planet_mearth', 'r_planet_rearth', 
         'r_planet_atmo_r_earth'].

    model_limits: float/None tuple
        Lower and upper limits to enforce for the model vector
        (model_low, model_high) where a value of None for either the lower or
        upper limit is used to not apply a limit.

    Returns
    -------
    model: 3D float array
        Combined model (star + tellurics + planet) at each phase of shape 
        [n_phase, n_spec, n_px].

    Updated
    -------
    model: 3D float array
        Combined model (star + tellurics + planet) at each phase of shape 
        [n_phase, n_spec, n_px].
    """
    # -------------------------------------------------------------------------
    # Initialise things and define variables for convenience
    # -------------------------------------------------------------------------
    n_spec = waves.shape[0]
    n_wave = waves.shape[1]
    n_phase = len(transit_info)

    ldc_cols = ["ldc_init_a1", "ldc_init_a2", "ldc_init_a3", "ldc_init_a4",]
    a_limb = syst_info.loc[ldc_cols, "value"].values

    mus = transit_info["mu_mid"].values
    mu_wgt = transit_info["planet_area_frac_mid"].values

    airmasses = transit_info["airmass"].values

    transit_nums = transit_info["transit_num"].values

    # Grab our RVs for convenience
    gamma = transit_info["gamma"].values       # gamma = rv_star + rv_bcor
    beta = transit_info["beta"].values         # beta = gamma + rv_projected
    delta = transit_info["delta"].values       # delta = gamma + rv_planet

    # Initialise model array
    model = np.zeros((n_phase, n_spec, n_wave))
    
    # Pre-compute the limb darkening function:
    # sum [a_k * mu^(k/2)] / sum [2*a_k / (k + 4)]
    ipow = np.arange(len(a_limb))+1
    N = np.pi * (1 - np.sum(ipow*a_limb/(ipow+4)))

    # -------------------------------------------------------------------------
    # Create model array for all phases and spectral segments
    # -------------------------------------------------------------------------
    # To create our model, we must iterate over each phase and spectral 
    # segment and combine our stellar flux, tau, and telluric arrays with the
    # appropriate velocity shifts and mu values.
    desc = "Creating model for all phases"

    for phase_i in tqdm(range(0, n_phase), desc=desc, leave=False):
        for spec_i in range(0, n_spec):
            # Stellar flux RV shifted using stellar + barycentric velocity
            f1 = tu.doppler_shift(
                x=waves[spec_i],
                y=flux[spec_i],
                gamma=-gamma[phase_i],
                y2=flux_2[spec_i])

            # Stellar flux RV shifted to the *projected* velocity velocity of
            # the star at mu position associated with the planet.
            f2 = tu.doppler_shift(
                x=waves[spec_i],
                y=flux[spec_i],
                gamma=-beta[phase_i],
                y2=flux_2[spec_i])

            # Only add the planetary signal in during the transit itself
            if mus[phase_i] > 0:
                #ipwr = np.arange(n_mus)

                # Calculate planet flux RV shifted to planet frame
                planet_trans = tu.doppler_shift(
                    x=waves[spec_i],
                    y=trans[spec_i],
                    gamma=-delta[phase_i],
                    y2=trans_2[spec_i],)

                intens = (f2 * (
                    1 - np.sum(a_limb*(1 - np.sqrt(mus[phase_i]**ipow))))
                    /N*planet_trans*mu_wgt[phase_i])

            else:
                intens = np.zeros(n_wave)

            transit_i = transit_nums[phase_i]

            # Combine each of our separate components, making sure to use the
            # appropriate telluric vector
            model[phase_i, spec_i] = (scale[phase_i] * (f1-intens) 
                * np.exp(-tau[transit_i, spec_i]*airmasses[phase_i]))

    # [Optional] Enforce model limits
    if model_limits[0] is not None or model_limits[1] is not None:
        model[:] = \
            np.clip(a=model, a_min=model_limits[0], a_max=model_limits[1],)

    return model


def run_transit_model_iteration(
    waves,
    obs_spec,
    obs_spec_2,
    flux,
    flux_2,
    tau,
    tau_2,
    trans,
    trans_2,
    scale,
    model,
    mask,
    mask_2,
    transit_info,
    syst_info,
    tau_nr_tolerance,
    stellar_flux_limits,
    telluric_trans_limits,
    telluric_tau_limits,
    planet_trans_limits,
    scale_limits,
    model_limits,
    lambda_treg_star,
    lambda_treg_tau,
    lambda_treg_planet,
    is_first,
    do_fix_flux_vector,
    do_fix_trans_vector,
    do_fix_tau_vector,
    do_fix_scale_vector,
    max_tau_nr_iter,):
    """Runs a single iteration of our inverse model. Each iteration does the
    following:
     1) Update stellar flux, flux derivative, and model.
     2) Update tau, tau derivative, and model.
     3) Update scaling and model.
     4) Update planet transmission, derivative, and model.

    All arrays are updated in-place.

    Parameters
    ----------
    waves: float array
        Wavelength scales for each spectral segment of shape [n_spec, n_px].
    
    obs_spec, obs_spec_2: 2D float array
       3D float array of observed spectra and its derivative of shape
       [n_phase, n_spec, n_px].
    
    flux, flux_2: 2D float array
        Fitted model stellar flux and its derivative of shape [n_spec, n_px].

    tau, tau_2: 2D float array
        Fitted model telluric tau and its derivative of shape [n_spec, n_px].

    trans, trans_2: 2D float array
        Fitted model planet transmission and its derivative of shape 
        [n_spec, n_px].

    scale: 1D float array
        Fitted model scale parameter of shape [n_phase].

    model: 3D float array
        Fitted model (star + tellurics + planet) matrix of shape
        [n_phase, n_spec, n_px].

    mask, mask_2: 3D float array
        Mask array and its derivative of shape [n_phase, n_spec, n_px]. 
        Contains either 0 or 1. TODO: derivative?

    transit_info: pandas DataFrame
        Transit info DataFrames containing information associated with each 
        transit time step with columns:

        ['mjd_start', 'mjd_mid', 'mjd_end', 'jd_start', 'jd_mid', 'jd_end',
         'airmass', 'bcor', 'hcor', 'ra', 'dec', 'exptime_sec', 'nod_pos',
         'raw_file', 'phase_start', 'is_in_transit_start', 'r_x_start',
         'r_y_start', 'r_z_start', 'v_x_start', 'v_y_start', 'v_z_start',
         's_projected_start', 'scl_start', 'mu_start',
         'planet_area_frac_start', 'phase_mid', 'is_in_transit_mid',
         'r_x_mid', 'r_y_mid', 'r_z_mid', 'v_x_mid', 'v_y_mid', 'v_z_mid',
         's_projected_mid', 'scl_mid', 'mu_mid', 'planet_area_frac_mid',
         'phase_end', 'is_in_transit_end', 'r_x_end', 'r_y_end', 'r_z_end',
         'v_x_end', 'v_y_end', 'v_z_end', 's_projected_end', 'scl_end',
         'mu_end', 'planet_area_frac_end', 'gamma', 'beta', 'delta']

    syst_info: pandas DataFrame
        DataFrame containing planet/star/system properties. The data frame has
        columns ['value', 'sigma', 'reference', 'comment'] and indices:
        ['m_star_msun', 'r_star_rsun', 'k_star_mps', 'dist_pc', 'vsini',
         'rv_star', 'ldc_init_a1', 'ldc_init_a2', 'ldc_init_a3', 'ldc_init_a4',
         'a_planet_au', 'e_planet', 'i_planet_deg', 'omega_planet_deg',
         'w_planet_deg', 'k_star_mps', 'transit_dur_hours', 'jd0_days', 
         'period_planet_days', 'm_planet_mearth', 'r_planet_rearth', 
         'r_planet_atmo_r_earth'].

    tau_nr_tolerance: float
        Convergence tolerance for the Newton-Raphson iteration scheme for tau
        and the model convergence tolerance respectively.

    stellar_flux_limits, telluric_trans_limits, telluric_tau_limits, 
    planet_limits, scale_limits/model_limits: float/None tuple
        Lower and upper limits to enforce for the our respective model arrays
        (low, high) where a value of None for either the lower or upper limit
        is used to not apply a limit.

    lambda_treg_star, lambda_treg_tau, lambda_treg_planet: float or None
        Tikhonov regularisation parameter lambda for the stellar flux, telluric
        tau, and planet transmission spectra respectively. Defaults to None, in
        which case no regularisation is applied.

    is_first: boolean
        Whether this is the first iteration which affects the initialisation
        for the Newton-Raphson iterative scheme for tau.

    do_fix_flux_vector, do_fix_tau_vector, do_fix_trans_vector,
    do_fix_scale_vector: boolean
        If true, we don't iteratre the respective vector when modelling. This
        is useful for debugging.

    max_tau_nr_iter: int
        Maximum number of Newton's method iterations.

    Updates
    -------
    flux: 2D float array
        Fitted model stellar flux of shape [n_spec, n_px].

    trans: 2D float array
        Fitted model planet transmission of shape [n_spec, n_px].

    tau: 2D float array
        Fitted model telluric tau of shape [n_spec, n_px].

    scale: 1D float array
        Fitted model scale parameter of shape [n_phase].

    model: 3D float array
        Fitted model (star + tellurics + planet) matrix of shape
        [n_phase, n_spec, n_px].

    mask: 3D float array
        Mask array of shape [n_phase, n_spec, n_px]. Contains either 0 or 1.
    """
    # -------------------------------------------------------------------------
    # Define variables for convenience
    # -------------------------------------------------------------------------
    n_spec = obs_spec.shape[1]
    n_transits = tau.shape[0]

    # -------------------------------------------------------------------------
    # Update stellar flux & model by reference
    # -------------------------------------------------------------------------
    # TODO I don't think this is needed since we update the derivative along
    # with the stellar flux in update_stellar_flux?
    # flux_2 = flux.copy()
    if not do_fix_flux_vector:
        print("- Updating stellar flux...")
        update_stellar_flux(
            waves=waves,
            flux=flux,
            flux_2=flux_2,
            tau=tau,
            tau_2=tau_2,
            trans=trans,
            trans_2=trans_2,
            scale=scale,
            mask=mask,
            mask_2=mask_2,
            obs_spec=obs_spec,
            obs_spec_2=obs_spec_2,
            transit_info=transit_info,
            syst_info=syst_info,
            lambda_treg=lambda_treg_star,
            flux_limits=stellar_flux_limits,)

        model[:,:,:] = create_transit_model_array(
            waves=waves,
            flux=flux,
            flux_2=flux_2,
            tau=tau,
            trans=trans,
            trans_2=trans_2,
            scale=scale,
            transit_info=transit_info,
            syst_info=syst_info,
            model_limits=model_limits,)
        
    else:
        print("- Fixed flux, skipping")

    # -------------------------------------------------------------------------
    # Update tau & model by reference
    # -------------------------------------------------------------------------
    # Telluric has a non-linear equation so we use a Newton-Raphson solver
    # We do this for each order on each detector separately as the telluric
    # features do not overlap
    if not do_fix_tau_vector:
        if is_first:
            print("- Initialising tau...")
            init_tau_newton_raphson(
                obs_spec=obs_spec,
                model=model,
                tau=tau,
                mask=mask,
                transit_info=transit_info,
                telluric_trans_limits=telluric_trans_limits,)

        print("- Updating tau...")
        update_tau_newton_raphson(
            obs_spec=obs_spec,
            model=model,
            tau=tau,
            mask=mask,
            transit_info=transit_info,
            lambda_treg=lambda_treg_tau,
            tolerance=tau_nr_tolerance,
            telluric_tau_limits=telluric_tau_limits,
            max_tau_nr_iter=max_tau_nr_iter,)

        # Update tau derivative from our new tau array
        for transit_i in range(0, n_transits):
            for spec_i in range(0, n_spec):
                tau_2[transit_i, spec_i] = \
                    tu.bezier_init(waves[spec_i], tau[transit_i, spec_i],)
    
        # TODO: ensure model is consistently updated
    else:
        print("- Fixed tau, skipping")

    # -------------------------------------------------------------------------
    # Update scaling & model by reference
    # -------------------------------------------------------------------------
    if not do_fix_scale_vector:
        print("- Updating scaling...")
        update_scaling(model, obs_spec, scale, scale_limits)
    else:
        print("- Fixed scale, skipping")

    # -------------------------------------------------------------------------
    # Update transmission & model by reference
    # -------------------------------------------------------------------------
    if not do_fix_trans_vector:
        print("- Updating transmission...")
        update_transmission(
            waves=waves,
            obs_spec=obs_spec,
            obs_spec_2=obs_spec_2,
            flux=flux,
            flux_2=flux_2,
            tau=tau,
            tau_2=tau_2,
            trans=trans,
            trans_2=trans_2,
            scale=scale,
            transit_info=transit_info,
            syst_info=syst_info,
            lambda_treg=lambda_treg_planet,
            trans_limits=planet_trans_limits,)

        model[:,:,:] = create_transit_model_array(
            waves=waves,
            flux=flux,
            flux_2=flux_2,
            tau=tau,
            trans=trans,
            trans_2=trans_2,
            scale=scale,
            transit_info=transit_info,
            syst_info=syst_info,
            model_limits=model_limits,)

    else:
        print("- Fixed transmission, skipping")

    # All done, nothing to return since we modified in place


def run_transit_model(
    waves,
    obs_spec,
    transit_info,
    syst_info,
    lambda_treg_star=None,
    lambda_treg_tau=None,
    lambda_treg_planet=None,
    tau_nr_tolerance=1E-5,
    model_converge_tolerance=1E-5,
    stellar_flux_limits=(None, None),
    telluric_trans_limits=(None, None),
    telluric_tau_limits=(None,None),
    planet_trans_limits=(None, None),
    scale_limits=(None, None),
    model_limits=(None, None),
    max_model_iter=4000,
    max_tau_nr_iter=300,
    do_plot=False,
    print_every_n_iterations=100,
    do_fix_flux_vector=False,
    do_fix_trans_vector=False,
    do_fix_tau_vector=False,
    do_fix_scale_vector=False,
    fixed_flux=None,
    fixed_trans=None,
    fixed_tau=None,
    fixed_scale=None,
    init_with_flux_vector=False,
    init_with_tau_vector=False,
    init_with_trans_vector=False,
    init_with_scale_vector=False,):
    """
    We solve iteratively 4 systems of equations:
     1) for the stellar spectrum flux,
     2) for the telluric spectrum optical thickness tau,
     3) for the spectrum scaling factor, and
     4) for the atmospheric transmission of the exoplanet

    We can formulate the problem to solve for a fifth equation, the limb
    darkening, but this is not currently implemented and it is sufficient to
    assume constant limb darkening coefficients.

    We repeat updating these function until convergence is achieved.
    In the process we also construct the forward model for the observations.

    Parameters
    ----------
    waves: float array
        Wavelength scales for each spectral segment of shape [n_spec, n_px].
    
    obs_spec_list: 2D float array
       3D float array of observed spectra of shape  [n_phase, n_spec, n_px].
    
    transit_info: pandas DataFrame
        Transit info DataFrames containing information associated with each 
        transit time step with columns:

        ['mjd_start', 'mjd_mid', 'mjd_end', 'jd_start', 'jd_mid', 'jd_end',
         'airmass', 'bcor', 'hcor', 'ra', 'dec', 'exptime_sec', 'nod_pos',
         'raw_file', 'phase_start', 'is_in_transit_start', 'r_x_start',
         'r_y_start', 'r_z_start', 'v_x_start', 'v_y_start', 'v_z_start',
         's_projected_start', 'scl_start', 'mu_start',
         'planet_area_frac_start', 'phase_mid', 'is_in_transit_mid',
         'r_x_mid', 'r_y_mid', 'r_z_mid', 'v_x_mid', 'v_y_mid', 'v_z_mid',
         's_projected_mid', 'scl_mid', 'mu_mid', 'planet_area_frac_mid',
         'phase_end', 'is_in_transit_end', 'r_x_end', 'r_y_end', 'r_z_end',
         'v_x_end', 'v_y_end', 'v_z_end', 's_projected_end', 'scl_end',
         'mu_end', 'planet_area_frac_end', 'gamma', 'beta', 'delta']

    syst_info: pandas DataFrame
        DataFrame containing planet/star/system properties. The data frame has
        columns ['value', 'sigma', 'reference', 'comment'] and indices:
        ['m_star_msun', 'r_star_rsun', 'k_star_mps', 'dist_pc', 'vsini',
         'rv_star', 'ldc_init_a1', 'ldc_init_a2', 'ldc_init_a3', 'ldc_init_a4',
         'a_planet_au', 'e_planet', 'i_planet_deg', 'omega_planet_deg',
         'w_planet_deg', 'k_star_mps', 'transit_dur_hours', 'jd0_days', 
         'period_planet_days', 'm_planet_mearth', 'r_planet_rearth', 
         'r_planet_atmo_r_earth'].

    lambda_treg_star, lambda_treg_tau, lambda_treg_planet: float or None
        Tikhonov regularisation parameter lambda for the stellar flux, telluric
        tau, and planet transmission spectra respectively. Defaults to None, in
        which case no regularisation is applied.
    
    tau_nr_tolerance, model_converge_tolerance: float, default: 1E-5
        Convergence tolerance for the Newton-Raphson iteration scheme for tau
        and the model convergence tolerance respectively.
    
    stellar_flux_limits, telluric_trans_limits, telluric_tau_limits, 
    planet_limits, scale_limits/model_limits: float/None tuple
        Lower and upper limits to enforce for the our respective model arrays
        (low, high) where a value of None for either the lower or upper limit
        is used to not apply a limit. Default (None, None).

    max_model_iter, max_tau_nr_iter: int, default: 4000, 300
        Maximum number of loop iterations for fitting for the modelling and
        tau Newton's method respectively.

    do_plot: boolean, default: False
        TODO: not implemented.

    print_every_n_iterations: int, default: 100
        How many iterations to print fitting updates after.

    do_fix_flux_vector, do_fix_tau_vector, do_fix_trans_vector, 
    do_fix_scale_vector: boolean, default: False
        If true, we don't iterate the respective vector when modelling. This
        is useful for debugging.
    
    fixed_flux, fixed_tau, fixed_trans, fixed_scale: float array or None
        Arrays to fix flux, trans, tau, or scale to. Must have the same shape
        as their respective arrays.

    init_with_flux_vector, init_with_tau_vector, init_with_trans_vector,
    init_with_scale_vector: boolean, default: False
        If true, we initialise our flux, tau, trans, or scale vectors to their
        equivalent components. So long as the equivalent 
        do_fix_<component>_vector boolean is False, the arrays are still 
        allowed to float during fitting.

    Returns
    -------
    flux: 2D float array
        Fitted model stellar flux of shape [n_spec, n_px].

    trans: 2D float array
        Fitted model planet transmission of shape [n_spec, n_px].

    tau: 2D float array
        Fitted model telluric tau of shape [n_spec, n_px].

    scale: 1D float array
        Fitted model scale parameter of shape [n_phase].

    model: 3D float array
        Fitted model (star + tellurics + planet) matrix of shape
        [n_phase, n_spec, n_px].

    mask: 3D float array
        Mask array of shape [n_phase, n_spec, n_px]. Contains either 0 or 1.
    """
    # -------------------------------------------------------------------------
    # Error checking
    # -------------------------------------------------------------------------
    # TODO: error checking on fixed parameters
    # - Check dimensions
    # - Check booleans correspond to arrays
    # - Check we haven't fixed all our arrays
    # - Check that limits are appropriate

    # -------------------------------------------------------------------------
    # Initialise variables for convenience 
    # -------------------------------------------------------------------------
    n_transits = len(set(transit_info["transit_num"]))
    n_spec = obs_spec.shape[1]
    n_wave = obs_spec.shape[2]

    # We have two sets of phases to keep track of: the total number of phases,
    # and the number of phases observed on each night.
    n_phase_all = obs_spec.shape[0]

    # Masks to reference obs_spec/model/transit info per transit
    night_masks = [transit_info["transit_num"].values == trans_i
        for trans_i in range(n_transits)]

    # Get minimum airmass for each night
    airmasses = transit_info["airmass"].values
    am_min_i = np.argmin(airmasses)
    am_min_transit_i = transit_info.iloc[am_min_i]["transit_num"]
    am_nightly_min_i = [np.argmin(airmasses[nm]) for nm in night_masks]

    # -------------------------------------------------------------------------
    # Initialise planet transmission
    # -------------------------------------------------------------------------
    # Initialise with our fixed value
    if init_with_trans_vector or do_fix_trans_vector:
        trans = fixed_trans.copy()

    # Otherwise initialise transmission array to be the fractional light
    #  blocked by the planet (normalised to the stellar radius)
    else:
        rp_rstar_area_frac = syst_info.loc["rp_rstar", "value"]**2
        trans = np.zeros((n_spec, n_wave)) + rp_rstar_area_frac

    # -------------------------------------------------------------------------
    # Initialise telluric optical depth
    # -------------------------------------------------------------------------
    # Initialise with our fixed value
    if init_with_tau_vector or do_fix_tau_vector:
        tau = fixed_tau.copy()

    # Otherwise, for our initial guess for the tellurics, start with the lowest
    # observed airmass observation from each transit and assume that all
    # absorption is due to tellurics--that is normalise this transmission and 
    # convert to an optical depth.
    else:
        tau = []

        for trans_i, nm in enumerate(night_masks):
            tell_trans = obs_spec[nm][am_nightly_min_i[trans_i]].copy()

            # Clip telluric transmission for zeros/unreasonable optical depths
            tell_trans[:] = np.clip(
                a=tell_trans,
                a_min=telluric_trans_limits[0],
                a_max=telluric_trans_limits[1],)

            # Normalise -- TODO: smooth instead
            for spec_i in range(0, n_spec):
                tell_trans[spec_i,:] = \
                    tell_trans[spec_i]/ np.max(tell_trans[spec_i])

            tau.append(
                -np.log(tell_trans) / airmasses[nm][am_nightly_min_i[trans_i]])

        tau = np.stack(tau)

    # -------------------------------------------------------------------------
    # Initialise stellar flux
    # -------------------------------------------------------------------------
    # Initialise with fixed value
    if init_with_flux_vector or do_fix_flux_vector:
        flux = fixed_flux.copy()
    
    # Otherwise initial guess for stellar flux is just the airmass corrected 
    # observation from the minimum airmass epoch across all transits.
    else:
        flux = (obs_spec[am_min_i,:,:]
            / np.exp(-tau[am_min_transit_i]*airmasses[am_min_i]))

        # Clip fluxes
        if (stellar_flux_limits[0] is not None 
            or stellar_flux_limits[1] is not None):
            flux[:] = np.clip(
                a=flux,
                a_min=stellar_flux_limits[0],
                a_max=stellar_flux_limits[1],)

    # -------------------------------------------------------------------------
    # Derivatives
    # -------------------------------------------------------------------------
    # Initialise derivatives arrays
    obs_spec_2 = obs_spec.copy()
    flux_2 = flux.copy()
    trans_2 = trans.copy()
    tau_2 = tau.copy()

    # Initialise observational derivatives for every phase and spectral segment
    for phase_i in range(0, n_phase_all):
        for spec_i in range(0, n_spec):
            # Initialise for each phase and spectral segment
            obs_spec_2[phase_i,spec_i,:] = tu.bezier_init(
                x=waves[spec_i, :],
                y=obs_spec[phase_i,spec_i,:],)
    
    # Initialise derivatives for flux/tau/planet for every spectral segment
    for spec_i in range(0, n_spec):
        flux_2[spec_i] = tu.bezier_init(x=waves[spec_i], y=flux[spec_i],)
        trans_2[spec_i] = tu.bezier_init(x=waves[spec_i], y=trans[spec_i],)

        # Telluric derivative has to be updated on a per-transit basis
        for transit_i in range(0, n_transits):
            tau_2[transit_i, spec_i] = \
                tu.bezier_init(x=waves[spec_i], y=tau[transit_i, spec_i],)

    # -------------------------------------------------------------------------
    # Model and Scaling
    # -------------------------------------------------------------------------
    # Initialise with fixed scale vector
    if init_with_scale_vector or do_fix_scale_vector:
        scale = fixed_scale.copy()

        model = create_transit_model_array(
            waves=waves,
            flux=flux,
            flux_2=flux_2,
            tau=tau,
            trans=trans,
            trans_2=trans_2,
            scale=scale,
            transit_info=transit_info,
            syst_info=syst_info,
            model_limits=model_limits,)
    
    # Otherwise have our initial guess for scaling vector simply account for 
    # observed flux variability
    else:
        # TODO: avoid including transit itself in scale to have a better
        # initial guess.
        scale = np.ones(n_phase_all)

        # Initialise model based on the initial guesses for all components
        model = create_transit_model_array(
            waves=waves,
            flux=flux,
            flux_2=flux_2,
            tau=tau,
            trans=trans,
            trans_2=trans_2,
            scale=scale,
            transit_info=transit_info,
            syst_info=syst_info,
            model_limits=model_limits,)

        # Adjust the scale to match roughly variation of the observed flux and 
        for phase_i in range(0, n_phase_all):
            m1 = model[phase_i]
            o1 = obs_spec[phase_i]
            k1 = np.where(m1 > 0.)

            scale[phase_i] = np.median(o1[k1]/m1[k1])
        
        # TODO: not sure where this came from? Probably a good idea though?
        #scale /= np.max(scale)

        # Recompute the model
        model[:,:,:] = create_transit_model_array(
            waves=waves,
            flux=flux,
            flux_2=flux_2,
            tau=tau,
            trans=trans,
            trans_2=trans_2,
            scale=scale,
            transit_info=transit_info,
            syst_info=syst_info,
            model_limits=model_limits,)

    # -------------------------------------------------------------------------
    # Mask
    # -------------------------------------------------------------------------
    # Initialize the masks
    mask = np.ones_like(model)
    mask_2 = mask.copy()

    for phase_i in range(0, n_phase_all):
        for spec_i in range(0, n_spec):
            mask_2[phase_i, spec_i] = tu.bezier_init(
                x=waves[spec_i],
                y=mask[phase_i, spec_i],)

    # -------------------------------------------------------------------------
    # Print initial error + info
    # -------------------------------------------------------------------------
    #init_error = np.sum(mask*np.abs(obs_spec-model))/np.sum(mask)
    #print("-"*140, "\nModelling: Initial Values\n", "-"*140, sep="",)
    #print("Initial error: {}\n".format(init_error))
    print("-"*140, "\nStarting modelling\n", "-"*140, sep="",)
    print("N transits \t= {:1.0f}".format(n_transits))
    print("N phases \t= {:3.0f}".format(n_phase_all))
    print("Max iter \t= {}".format(max_model_iter))
    print("Stellar λ_reg \t= {}".format(lambda_treg_star))
    print("Telluric λ_reg \t= {}".format(lambda_treg_tau))
    print("Planet λ_reg \t= {}\n".format(lambda_treg_planet))
    print("Initial statistics:")
    # Store references to arrays in dict, and copies in another, for checking
    # convergence at the end of each iteration.
    arrays = {
        "obs_spec":obs_spec,
        "model":model,
        "flux":flux,
        "tau":tau,
        "trans":trans,
        "scale":scale,}
    
    arrays_old = {
        "obs_spec":obs_spec.copy(),
        "model":model.copy(),
        "flux":flux.copy(),
        "tau":tau.copy(),
        "trans":trans.copy(),
        "scale":scale.copy(),}

    for name in arrays.keys():
        min_array = np.nanmin(arrays[name])
        max_array = np.nanmax(arrays[name])
        median_array = np.nanmedian(arrays[name])
        mean_array = np.nanmean(arrays[name])
        std_array = np.nanstd(arrays[name])
        n_nans = np.sum(np.isnan(arrays[name]))
        print("{: <8} -->".format(name),
            " min = {:10.4f},".format(min_array),
            " median = {:10.4f},".format(median_array),
            " mean = {:15.4f},".format(mean_array),
            " std = {:15.4f},".format(std_array),
            " max = {:15.4f},".format(max_array),
            " # nans = {}".format(n_nans),
            sep="",)

    # -------------------------------------------------------------------------
    # Iterate Until Convergence
    # -------------------------------------------------------------------------
    # Initialise counter and while loop conditions
    iter_count = 0
    has_converged = False
    hit_max_iterations = False
    
    while (not has_converged) and (not hit_max_iterations):
        print("-"*160, "\nIteration #{}\n".format(iter_count), "-"*160, sep="")
        # Save previous estimates for comparison
        arrays_old["obs_spec"] = obs_spec.copy()
        arrays_old["model"] = model.copy()
        arrays_old["flux"] = flux.copy()
        arrays_old["tau"] = tau.copy()
        arrays_old["trans"] = trans.copy()
        arrays_old["scale"] = scale.copy()

        # Initialise boolean for running Newton's method on Tau
        is_first = True if iter_count == 0 else False

        # Run single iteration, updating arrays in place
        run_transit_model_iteration(
            waves=waves,
            obs_spec=obs_spec,
            obs_spec_2=obs_spec_2,
            flux=flux,
            flux_2=flux_2,
            tau=tau,
            tau_2=tau_2,
            trans=trans,
            trans_2=trans_2,
            scale=scale,
            model=model,
            mask=mask,
            mask_2=mask_2,
            transit_info=transit_info,
            syst_info=syst_info,
            tau_nr_tolerance=tau_nr_tolerance,
            stellar_flux_limits=stellar_flux_limits,
            telluric_trans_limits=telluric_trans_limits,
            telluric_tau_limits=telluric_tau_limits,
            planet_trans_limits=planet_trans_limits,
            scale_limits=scale_limits,
            model_limits=model_limits,
            lambda_treg_star=lambda_treg_star,
            lambda_treg_tau=lambda_treg_tau,
            lambda_treg_planet=lambda_treg_planet,
            is_first=is_first,
            do_fix_flux_vector=do_fix_flux_vector,
            do_fix_trans_vector=do_fix_trans_vector,
            do_fix_tau_vector=do_fix_tau_vector,
            do_fix_scale_vector=do_fix_scale_vector,
            max_tau_nr_iter=max_tau_nr_iter,)

        # -----------------------------------------
        # Check array equivalence [TODO: remove]
        # -----------------------------------------
        assert np.sum(obs_spec - arrays["obs_spec"]) == 0
        assert np.sum(model - arrays["model"]) == 0
        assert np.sum(flux - arrays["flux"]) == 0
        assert np.sum(tau - arrays["tau"]) == 0
        assert np.sum(trans - arrays["trans"]) == 0
        assert np.sum(scale - arrays["scale"]) == 0

        # Print update for every Nth iteration
        if (iter_count % print_every_n_iterations) == 0:
            print("\nIteration {} finished, ".format(iter_count),
                  "obs-model median resid = {:0.6f},".format(
                    np.nanmedian(obs_spec-model)))

            for name in arrays.keys():
                min_array = np.nanmin(arrays[name])
                max_array = np.nanmax(arrays[name])
                median_array = np.nanmedian(arrays[name])
                mean_array = np.nanmean(arrays[name])
                std_array = np.nanstd(arrays[name])
                n_nans = np.sum(np.isnan(arrays[name]))
                delta_array = np.median(arrays_old[name]-arrays[name])
                print("{: <8} -->".format(name),
                    " min = {:20.4f},".format(min_array),
                    " median = {:20.4f},".format(median_array),
                    " mean = {:20.4f},".format(mean_array),
                    " std = {:20.4f},".format(std_array),
                    " max = {:20.4f},".format(max_array),
                    #" # nans = {:20}".format(n_nans),
                    " change = {:+20.8f},".format(delta_array),
                    sep="",)

        # End the loop if we hit the maximum number of iterations
        if iter_count >= max_model_iter:
            hit_max_iterations = True

        # Continue to iterate so long as either flux or tau is not yet below
        # our adopted tolerance. TODO: account for possibility of both tau and
        # flux being fixed.
        delta_flux = np.nanmean(np.abs(arrays_old["flux"]-arrays["flux"]))
        delta_tau = np.nanmean(np.abs(arrays_old["tau"]-arrays["tau"]))
        delta_trans = np.nanmean(np.abs(arrays_old["trans"]-arrays["trans"]))
        delta_scale = np.nanmean(np.abs(arrays_old["scale"]-arrays["scale"]))

        print("\nMean change:")
        print("|Δ flux|\t= {:11.8f}".format(delta_flux),)
        print("|Δ tau|\t\t= {:11.8f}".format(delta_tau),)
        print("|Δ trans|\t= {:11.8f}".format(delta_trans),)
        print("|Δ scale|\t= {:11.8f}".format(delta_scale),)

        if (delta_flux < model_converge_tolerance
            and delta_tau < model_converge_tolerance
            and delta_trans < model_converge_tolerance):
            # Update convergence
            has_converged = True

        # Save a per iteration plot
        if do_plot:
            mplt.plot_iteration()

        # Finally, update iteration count
        iter_count += 1

    # All done, print a summary
    # TODO: add in booleans
    print('Iterations: ' + str(iter_count), np.mean(abs(obs_spec-model)),)

    return flux, trans, tau, scale, model, mask