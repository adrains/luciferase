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
# TODO: this should be revisted and relates to whether we are treating the
# removed SYSREM components as additive or multiplicative. 
do_normalise = False

# Load molecfit telluric model for masking of deep tellurics in SYSREM.
telluric_wave, _, _, telluric_trans = tu.load_telluric_spectrum(
    molecfit_fits=ss.molecfit_fits[0],
    tau_fill_value=ss.tau_fill_value,
    convert_to_angstrom=False,
    convert_to_nm=True,
    output_transmission=True,)

#------------------------------------------------------------------------------
# Main Operation
#------------------------------------------------------------------------------
for transit_i in range(ss.n_transit):
    # Grab dimensions of flux datacube
    (n_phase, n_spec, n_px) = fluxes_list[transit_i].shape

    # Decide which set of fluxes to use
    # 1) Load continuum normalised fluxes
    if ss.do_use_continuum_normalised_data:
        print("Using continuum normalised spectra.")
        fluxes_norm, sigmas_norm, _, _ = \
            tu.load_normalised_spectra_from_fits(
                fits_load_dir=ss.save_path,
                label=ss.label,
                n_transit=ss.n_transit,
                transit_i=transit_i,)

    # 2) Simply normalise each spectral segment by its median value.
    else:
        print("Using unnormalised spectra.")
        mf = np.nanmedian(fluxes_list[transit_i], axis=2)
        mf_3D = np.broadcast_to(mf[:,:,None], (n_phase, n_spec, n_px))
        fluxes_norm = fluxes_list[transit_i].copy() / mf_3D
        sigmas_norm = sigmas_list[transit_i].copy() / mf_3D

    #--------------------------------------------------------------------------
    # [Optional] Telluric masking
    #--------------------------------------------------------------------------
    # Mask out strong telluric features when running SYSREM
    if ss.do_mask_strong_tellurics_in_sysrem:
        telluic_mask_1D = telluric_trans < ss.telluric_trans_bad_px_threshold
        telluic_mask_3D = np.tile(
            telluic_mask_1D, n_phase).reshape(n_phase, n_spec, n_px)
    # No masking
    else:
        telluic_mask_3D = np.full(fluxes_norm.shape, False)

    #--------------------------------------------------------------------------
    # [Optional] Split of A/B sequences
    #--------------------------------------------------------------------------
    # Split A/B frames into separate sequences
    if ss.split_AB_sequences:
        print("Splitting A/B observations into separate sequences.")
        nod_pos = transit_info_list[transit_i]["nod_pos"].values
        sequence_masks = [nod_pos == "A", nod_pos == "B"]
        sequences = ["A", "B"]
    
    # OR run with interleaved A/B sequences
    else:
        print("Running with interleaved A/B sequences.")
        sequence_masks = [np.full(n_phase, True)]
        sequences = ["AB"]

    #--------------------------------------------------------------------------
    # Clean and run SYSREM for this night
    #--------------------------------------------------------------------------
    # Loop over each of our sequences
    for seq, seq_mask in zip(sequences, sequence_masks):
        # Clean and prepare our fluxes for input to SYSREM. This involves:
        # - sigma clipping along phase and spectral dimension
        # - interpolate along the phase dimension
        resid_init, flux, e_flux_init = sr.clean_and_compute_initial_resid(
            spectra=fluxes_norm[seq_mask],
            e_spectra=sigmas_norm[seq_mask],
            mjds=transit_info_list[transit_i]["mjd_mid"].values[seq_mask],
            sigma_threshold_phase=ss.sigma_threshold_phase,
            sigma_threshold_spectral=ss.sigma_threshold_spectral,
            do_normalise=do_normalise,)

        n_phase_seq = np.sum(seq_mask)

        # Mask out strong tellurics by setting these pixels to nans
        # Note: telluic_mask_3D will be all False if not doing masking.
        resid_init[telluic_mask_3D[seq_mask]] = np.nan

        #----------------------------------------------------------------------
        # Option 1: Order-by-order detrending
        #----------------------------------------------------------------------
        if ss.run_sysrem_order_by_order and ss.detrending_algorithm != "PISKUNOV":
            print("Running detrending order-by-order:")
            resid_all = np.full(
                (ss.n_sysrem_iter+1, n_phase_seq, n_spec, n_px), np.nan)

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
                
                resid_all[:, :,spec_i,:] = resid
        #----------------------------------------------------------------------
        # Option 2: Simultaneous detrending
        #----------------------------------------------------------------------
        else:
            print("Running detrending on all orders:")

            resid = sr.detrend_spectra(
                resid_init=resid_init,
                e_resid=e_flux_init,
                detrending_algorithm=ss.detrending_algorithm,
                n_iter=ss.n_sysrem_iter,
                tolerance=ss.sysrem_convergence_tol,
                max_converge_iter=ss.sysrem_max_convergence_iter,
                diff_method=ss.sysrem_diff_method,
                sigma_threshold=ss.sigma_threshold_sysrem_piskunov,)

            resid_all = resid.reshape(
                (ss.n_sysrem_iter+1, n_phase_seq, n_spec, n_px))

        # Save residuals
        tu.save_sysrem_residuals_to_fits(
            fits_load_dir=ss.save_path,
            label=ss.label,
            n_transit=ss.n_transit,
            sysrem_resid=resid_all,
            transit_i=transit_i,
            sequence=seq)

        # Plotting
        plot_label = "n{}_{}_{}".format(transit_i, seq, ss.label)

        tplt.plot_sysrem_residuals(waves, resid_all, plot_label=plot_label,)