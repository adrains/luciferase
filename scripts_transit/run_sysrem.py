"""Script to run SYSREM on a datacube (either real or simulated) of
exoplanet transmission spectra. The resulting residuals (along with the 
continuum normalised spectra used to generate them) are then saved back to the
same fits file as extra extensions.
"""
import numpy as np
import transit.utils as tu
import transit.simulator as sim
import luciferase.spectra as ls
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

#------------------------------------------------------------------------------
# Main Operation
#------------------------------------------------------------------------------
for transit_i in range(ss.n_transit):
    # Grab dimensions of flux datacube
    (n_phase, n_spec, n_px) = fluxes_list[transit_i].shape

    # Sort to be in proper wavelength order so nothing unexpected happens
    wave_ii = np.argsort(np.median(waves, axis=1))
    waves = waves[wave_ii]
    waves_1d = np.reshape(waves, n_spec*n_px)

    for night_i in range(len(fluxes_list)):
        fluxes_list[night_i] = fluxes_list[night_i][:, wave_ii, :]
        sigmas_list[night_i] = sigmas_list[night_i][:, wave_ii, :]

        # HACK: clean sigma=0 values
        is_zero = sigmas_list[night_i] == 0
        sigmas_list[night_i][is_zero] = 1E5

    #--------------------------------------------------------------------------
    # Import telluric vector for this night
    #--------------------------------------------------------------------------
    # Import telluric vector
    telluric_wave, telluric_tau, _ = sim.load_telluric_spectrum(
        molecfit_fits=ss.molecfit_fits[transit_i],
        tau_fill_value=ss.tau_fill_value,)

    # TODO: properly interpolate telluric vector

    telluric_wave /= 10
    telluric_trans = 10**-telluric_tau

    #--------------------------------------------------------------------------
    # Continuum normalise spectra
    #--------------------------------------------------------------------------
    print("Continuum normalising spectra...")
    fluxes_norm, sigmas_norm, poly_coeff = \
        ls.continuum_normalise_all_spectra_with_telluric_model(
            waves_sci=waves,
            fluxes_sci=fluxes_list[transit_i],
            sigmas_sci=sigmas_list[transit_i],
            wave_telluric=telluric_wave,
            trans_telluric=telluric_trans,
            wave_stellar=wave_stellar,
            spec_stellar=spec_stellar,
            bcors=transit_info_list[transit_i]["bcor"].values,
            rv_star=syst_info.loc["rv_star", "value"],)

    # Construct bad px mask from tellurics
    print("Constructing bad px mask from tellurics...")
    bad_px_mask_1D = telluric_trans < ss.telluric_trans_bad_px_threshold
    bad_px_mask_3D = np.tile(
        bad_px_mask_1D, n_phase).reshape(n_phase, n_spec, n_px)

    # Save normalised spectra back to fits file
    tu.save_normalised_spectra_to_fits(
        fits_load_dir=ss.save_path,
        label=ss.label,
        n_transit=ss.n_transit,
        waves_norm=waves,
        fluxes_norm=fluxes_norm,
        sigmas_norm=sigmas_norm,
        bad_px_mask_norm=bad_px_mask_3D,
        transit_i=transit_i,)

    #--------------------------------------------------------------------------
    # Run SYSREM for this night
    #--------------------------------------------------------------------------
    print("Running SYSREM...")
    resid_all = np.full((ss.n_sysrem_iter+1, n_phase, n_spec, n_px), np.nan)

    # Run SYSREM on one spectral segment at a time
    for spec_i in range(n_spec):
        fmt_txt = "\nSpectral segment {:0.0f}, λ~{:0.0f} nm\n".format(
                spec_i, np.mean(waves[spec_i]))
        print("-"*80, fmt_txt, "-"*80,sep="")

        resid = sr.run_sysrem(
            spectra=fluxes_norm[:,spec_i,:],
            e_spectra=sigmas_norm[:,spec_i,:],
            bad_px_mask=bad_px_mask_3D[:,spec_i,:],
            n_iter=ss.n_sysrem_iter,
            mjds=transit_info_list[transit_i]["mjd_mid"].values,
            tolerance=ss.sysrem_convergence_tol,
            max_converge_iter=ss.sysrem_max_convergence_iter,
            diff_method=ss.sysrem_diff_method,
            sigma_threshold_phase=ss.sigma_threshold_phase,
            sigma_threshold_spectral=ss.sigma_threshold_spectral,)
        
        resid_all[:,:,spec_i,:] = resid

    # Save residuals
    tu.save_sysrem_residuals_to_fits(
        fits_load_dir=ss.save_path,
        label=ss.label,
        n_transit=ss.n_transit,
        sysrem_resid=resid_all,
        transit_i=transit_i,)

    # Plotting
    tplt.plot_sysrem_residuals(waves, resid_all)