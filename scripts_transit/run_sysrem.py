"""Script to run SYSREM on a datacube (either real or simulated) of
exoplanet transmission spectra. The resulting residuals (along with the 
continuum normalised spectra used to generate them) are then saved back to the
same fits file as extra extensions.
"""
import numpy as np
import transit.utils as tu
import transit.sysrem as sr
import transit.plotting as tplt
import luciferase.utils as lu

#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------
# Import settings file
simulation_settings_file = "scripts_transit/sysrem_settings.yml"
ss = tu.load_yaml_settings(simulation_settings_file)

# Import stellar spectrum
wave_stellar, spec_stellar = lu.load_plumage_template_spectrum(
    template_fits=ss.stellar_template_fits,
    do_convert_air_to_vacuum_wl=False,)

# Load data from fits file--this can either be real or simulated data
waves, fluxes_list, sigmas_list, det, orders, transit_info_list, syst_info = \
    tu.load_transit_info_from_fits(ss.save_path, ss.label, ss.n_transit)

# Currently both detrending algorithms do not require normalisation to work
if ss.detrending_algorithm == "PISKUNOV":
    do_normalise = False
else:
    do_normalise = False

#------------------------------------------------------------------------------
# Main Operation
#------------------------------------------------------------------------------
for transit_i in range(ss.n_transit):
    # Grab dimensions of flux datacube
    (n_phase, n_spec, n_px) = fluxes_list[transit_i].shape

    # Decide what data we're using
    # 1) Load continuum normalised data
    if ss.do_use_continuum_normalised_data:
        print("Usinng continuum normalised spectra.")
        fluxes_norm, sigmas_norm, strong_telluic_mask, poly_coeff = \
            tu.load_normalised_spectra_from_fits(
                fits_load_dir=ss.save_path,
                label=ss.label,
                n_transit=ss.n_transit,
                transit_i=transit_i,)

    # 2) Avoid continuum normalisation for debugging
    else:
        print("Using unnormalised spectra.")
        mf = np.nanmedian(fluxes_list[transit_i], axis=2)
        mf_3D = np.broadcast_to(mf[:,:,None], (n_phase, n_spec, n_px))
        fluxes_norm = fluxes_list[transit_i].copy() / mf_3D
        sigmas_norm = sigmas_list[transit_i].copy() / mf_3D
        strong_telluic_mask = np.full_like(fluxes_norm, False)

    # Clean and prepare our fluxes for input to SYSREM. This involves:
    # - sigma clipping along phase and spectral dimension
    # - interpolate along the phase dimension
    resid_init, flux, e_flux_init = sr.clean_and_compute_initial_resid(
        spectra=fluxes_norm,
        e_spectra=sigmas_norm,
        strong_telluic_mask=strong_telluic_mask,
        mjds=transit_info_list[transit_i]["mjd_mid"].values,
        sigma_threshold_phase=ss.sigma_threshold_phase,
        sigma_threshold_spectral=ss.sigma_threshold_spectral,
        do_normalise=do_normalise,)

    #--------------------------------------------------------------------------
    # Run SYSREM for this night
    #--------------------------------------------------------------------------
    if ss.run_sysrem_order_by_order and ss.detrending_algorithm != "PISKUNOV":
        print("Running detrending order-by-order:")
        resid_all = np.full(
            (ss.n_sysrem_iter+1, n_phase, n_spec, n_px), np.nan)

        # Run SYSREM on one spectral segment at a time
        for spec_i in range(n_spec):
            fmt_txt = "\nSpectral segment {:0.0f}, λ~{:0.0f} nm\n".format(
                    spec_i, np.mean(waves[spec_i]))
            print("-"*80, fmt_txt, "-"*80,sep="")

            resid = sr.detrend_spectra(
                resid_init=resid_init[:,spec_i,:],
                e_resid=e_flux_init[:,spec_i,:],
                detrending_algorithm=ss.detrending_algorithm,
                n_iter=ss.n_sysrem_iter,
                tolerance=ss.sysrem_convergence_tol,
                max_converge_iter=ss.sysrem_max_convergence_iter,
                diff_method=ss.sysrem_diff_method,
                sigma_threshold=ss.sigma_threshold_sysrem_piskunov,)
            
            resid_all[:,:,spec_i,:] = resid
    
    else:
        print("Running detrending on all orders:")
        resid_init_reshaped = resid_init.reshape((n_phase, n_spec*n_px))
        sigmas_init_reshaped = e_flux_init.reshape((n_phase, n_spec*n_px))

        resid = sr.detrend_spectra(
            resid_init=resid_init,
            e_resid=e_flux_init,
            detrending_algorithm=ss.detrending_algorithm,
            n_iter=ss.n_sysrem_iter,
            tolerance=ss.sysrem_convergence_tol,
            max_converge_iter=ss.sysrem_max_convergence_iter,
            diff_method=ss.sysrem_diff_method,
            sigma_threshold=ss.sigma_threshold_sysrem_piskunov,)

        resid_all = resid.reshape((ss.n_sysrem_iter+1, n_phase, n_spec, n_px))

    # Save residuals
    tu.save_sysrem_residuals_to_fits(
        fits_load_dir=ss.save_path,
        label=ss.label,
        n_transit=ss.n_transit,
        sysrem_resid=resid_all,
        transit_i=transit_i,)

    # Plotting
    plot_label = "n{}_{}".format(transit_i, ss.label)

    tplt.plot_sysrem_residuals(waves, resid_all, plot_label=plot_label,)