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

# Currently both detrending algorithms do not require normalisation (note: not
# in a continuum normalisation way, but in a SYSREM residual prep way) to work
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

(n_spec, n_px) = waves.shape
telluric_trans_2D = telluric_trans.reshape(n_spec, n_px)

#------------------------------------------------------------------------------
# Main Operation
#------------------------------------------------------------------------------
for transit_i in range(ss.n_transit):
    # Grab dimensions of flux datacube
    (n_phase, n_spec, n_px) = fluxes_list[transit_i].shape
    
    #--------------------------------------------------------------------------
    # Create telluric mask
    #--------------------------------------------------------------------------
    # We need this for masking out tellurics for a) determining median flux
    # values for spectral normalisation, and b) masking out strong tellurics
    # before running SYSREM.
    telluric_mask_2D = telluric_trans_2D < ss.telluric_trans_bad_px_threshold
    telluric_mask_3D = np.broadcast_to(
        telluric_mask_2D[None,:,:], (n_phase, n_spec, n_px))

    #--------------------------------------------------------------------------
    # [Optional] Mask out edge pixels
    #--------------------------------------------------------------------------
    if ss.do_mask_detector_edges:
        px_to_mask = np.logical_or(
            np.arange(n_px) < ss.edge_px_to_mask,
            np.arange(n_px) > n_px - ss.edge_px_to_mask,)
        
        fluxes_list[transit_i][:,:,px_to_mask] = np.nan
        sigmas_list[transit_i][:,:,px_to_mask] = np.nan

    #--------------------------------------------------------------------------
    # Import and prepare fluxes for this night
    #--------------------------------------------------------------------------
    # Option 1) Load *physically meaningfully* continuum normalised fluxes
    if ss.do_use_continuum_normalised_data:
        print("Using continuum normalised spectra.")
        fluxes_norm, sigmas_norm, _, _ = \
            tu.load_normalised_spectra_from_fits(
                fits_load_dir=ss.save_path,
                label=ss.label,
                n_transit=ss.n_transit,
                transit_i=transit_i,)

    # Option 2) First do a median normalisation (masking out strong tellurics)
    # then, optionally, do an additional continuum normalisation that either
    # assumes there are only two continua (A and B) and one must be corrected 
    # to the other, or that the continuum is time varying.
    else:
        print("Continuum normalising data.")
        fluxes_norm, sigmas_norm = sr.continuum_normalise_datacube(
            waves=waves,
            fluxes=fluxes_list[transit_i],
            sigmas=sigmas_list[transit_i],
            telluric_trans=telluric_trans_2D,
            nodding_pos=transit_info_list[transit_i]["nod_pos"].values,
            telluric_trans_bad_px_threshold=ss.telluric_trans_bad_px_threshold,
            continuum_correction_kind=ss.continuum_correction_kind,
            continuum_ratio_smoothing_resolution=\
                ss.continuum_corr_smoothing_resolution,
            do_diagnostic_plotting=True,)

    #--------------------------------------------------------------------------
    # [Optional] Regrid the data to be in the stellar rest frame
    #--------------------------------------------------------------------------
    if ss.run_sysrem_in_stellar_frame:
        print("Regridding data to stellar rest frame.")
        # To regrid to the stellar frame, we opt to do the minimal possible
        # interpolation here in that we just interpolate to the RV frame of the
        # first exposure and shift every subsequent exposure by delta bcor. We
        # then leave doing the full interpolation to the cross-correlation
        # step.
        bcors = transit_info_list[transit_i]["bcor"].values
        bcors -= bcors[0]
        rv_bcors =  -1*bcors

        # Shift spectra into stellar frame
        fluxes_norm, sigmas_norm = sr.change_flux_restframe(
            waves=waves,
            fluxes=fluxes_norm,
            sigmas=sigmas_norm,
            rv_bcors=rv_bcors,
            interpolation_method="linear",)

    else:
        print("Leaving data in telluric rest frame.")

    #--------------------------------------------------------------------------
    # [Optional] Split A/B sequences
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
    # Clean
    #--------------------------------------------------------------------------
    # TODO WHY FIRST
    # Clean and prepare our fluxes for input to SYSREM. This involves:
    # - sigma clipping along phase and spectral dimension
    # - interpolate along the phase dimension
    resid_init_full, flux, e_flux_init = sr.clean_and_compute_initial_resid(
        spectra=fluxes_norm,
        e_spectra=sigmas_norm,
        mjds=transit_info_list[transit_i]["mjd_mid"].values,
        sigma_threshold_phase=ss.sigma_threshold_phase,
        sigma_threshold_spectral=ss.sigma_threshold_spectral,
        do_normalise=do_normalise,)

    #--------------------------------------------------------------------------
    # Run SYSREM for this night
    #--------------------------------------------------------------------------
    for seq, seq_mask in zip(sequences, sequence_masks):
        # Now that we've cleaned the full sequence, select subsequences as req
        resid_init = resid_init_full.copy()

        # Now set all values from the opposite sequence to nan
        resid_init[~seq_mask,:,:] = np.nan

        #----------------------------------------------------------------------
        # [Optional] Mask out strong tellurics
        #----------------------------------------------------------------------
        # Mask out strong tellurics by setting these pixels to nans
        # Note: telluric_mask_3D will be all False if not doing masking.
        if ss.do_mask_strong_tellurics_in_sysrem:
            #telluric_mask_seq = telluric_mask_3D[seq_mask]
            resid_init[telluric_mask_3D] = np.nan

        #----------------------------------------------------------------------
        # Option 1: Order-by-order detrending
        #----------------------------------------------------------------------
        if ss.run_sysrem_order_by_order and ss.detrending_algorithm != "PISKUNOV":
            print("Running detrending order-by-order:")
            resid_all = np.full(
                (ss.n_sysrem_iter+1, n_phase, n_spec, n_px), np.nan)

            # Run SYSREM on one spectral segment at a time
            for spec_i in range(n_spec):
                fmt_txt = "\nSpectral segment {:0.0f}, λ~{:0.0f} nm\n".format(
                        spec_i, np.mean(waves[spec_i]))
                print("-"*80, fmt_txt, "-"*80,sep="")

                # Only run SYSREM on unmasked regions
                if ss.do_mask_strong_tellurics_in_sysrem: 
                    unmasked_px = ~telluric_mask_2D[spec_i,:]
                    resid_init_masked = resid_init[:,spec_i,unmasked_px]
                    e_flux_masked = e_flux_init[:,spec_i,unmasked_px]
                else:
                    resid_init_masked = resid_init[:,spec_i]
                    e_flux_masked = e_flux_init[:,spec_i]

                resid = sr.detrend_spectra(
                    resid_init=resid_init_masked,
                    e_resid=e_flux_masked,
                    detrending_algorithm=ss.detrending_algorithm,
                    n_iter=ss.n_sysrem_iter,
                    tolerance=ss.sysrem_convergence_tol,
                    max_converge_iter=ss.sysrem_max_convergence_iter,
                    diff_method=ss.sysrem_diff_method,
                    sigma_threshold=ss.sigma_threshold_sysrem_piskunov,)
                
                if ss.do_mask_strong_tellurics_in_sysrem:
                    resid_all[:, :,spec_i, unmasked_px] = resid
                else:
                    resid_all[:, :,spec_i] = resid
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
                (ss.n_sysrem_iter+1, n_phase, n_spec, n_px))

        # Save residuals
        rv_frame = "stellar" if ss.run_sysrem_in_stellar_frame else "telluric"
        tu.save_sysrem_residuals_to_fits(
            fits_load_dir=ss.save_path,
            label=ss.label,
            n_transit=ss.n_transit,
            sysrem_resid=resid_all[:,seq_mask],    # Only save correct seq
            transit_i=transit_i,
            sequence=seq,
            rv_frame=rv_frame,)

        # Plotting
        plot_title = "Night {} ({}): {} [{}]".format(
            transit_i+1, seq, ss.label, rv_frame)
        plot_label = "{}_n{}_{}_{}".format(
            rv_frame, transit_i+1, seq, ss.label)

        tplt.plot_sysrem_residuals(
            waves,
            resid_all[:,seq_mask],
            plot_label=plot_label,
            plot_title=plot_title,)