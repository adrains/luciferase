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

# Load molecfit telluric model for masking of deep tellurics in SYSREM.
telluric_wave, _, _, telluric_trans = tu.load_telluric_spectrum(
    molecfit_fits=ss.molecfit_fits[0],
    tau_fill_value=ss.tau_fill_value,
    convert_to_angstrom=False,
    convert_to_nm=True,
    output_transmission=True,)

(n_spec, n_px) = waves.shape
telluric_trans_2D = telluric_trans.reshape(n_spec, n_px)

# Initialise lists for storing SYSREM coefficient arrays (and associated phases
# + labels) for plotting later
labels_per_seq = []
phases_per_seq = []
coeff_phase_all_seq = []
coeff_wave_all_seq = []

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
    # Run SYSREM for this night
    #--------------------------------------------------------------------------
    for seq, seq_mask in zip(sequences, sequence_masks):
        # Compute initial residuals (without normalisation, this is just the
        # spectra as is) for the sequence in question.
        # TODO: reintroduce potential for pre-normalisation
        resid_init = fluxes_norm[seq_mask].copy()
        e_resid = sigmas_norm[seq_mask].copy()

        # Grab number of phases in this sequence
        (n_phase_seq, _, _) = resid_init.shape

        #----------------------------------------------------------------------
        # Mask out NaNs and strong tellurics
        #----------------------------------------------------------------------
        # Mask out any columns with NaNs
        is_nan_3D = np.isnan(fluxes_norm)         # [n_phase_seq, n_spec, n_px]
        is_nan_column = np.any(is_nan_3D, axis=0) # [n_spec, n_px]

        # Combine telluric and bad column masks, tile to 3D
        bad_px_mask_2D = np.logical_or(telluric_mask_2D, is_nan_column)
        bad_px_3D = np.broadcast_to(
            bad_px_mask_2D[None,:,:], (n_phase_seq, n_spec, n_px))
        
        # Set any masked px to nans
        resid_init[bad_px_3D] = np.nan

        #----------------------------------------------------------------------
        # Option 1: Order-by-order detrending
        #----------------------------------------------------------------------
        if ss.run_sysrem_order_by_order and ss.detrending_algorithm != "PISKUNOV":
            print("Running detrending order-by-order:")
            # Initialise output arrays to hold residuals and fitted SYSREM
            # coefficients in phase and wavelength
            resid_all = np.full(
                (ss.n_sysrem_iter+1, n_phase_seq, n_spec, n_px), np.nan)
            
            coeff_phase_all = np.full(
                (ss.n_sysrem_iter, n_spec, n_phase_seq), np.nan)
            
            coeff_wave_all = np.full(
                (ss.n_sysrem_iter, n_spec, n_px), np.nan)

            # Run SYSREM on one spectral segment at a time
            for spec_i in range(n_spec):
                fmt_txt = "\nSpectral segment {:0.0f}, λ~{:0.0f} nm\n".format(
                        spec_i, np.mean(waves[spec_i]))
                print("-"*80, fmt_txt, "-"*80,sep="")

                # Only run SYSREM on unmasked regions
                unmasked_px = ~telluric_mask_2D[spec_i,:]
                resid_init_masked = resid_init[:,spec_i,unmasked_px]
                e_flux_masked = e_resid[:,spec_i,unmasked_px]

                # Run SYSREM
                resid, coeff_phase, coeff_wave = sr.detrend_spectra(
                    resid_init=resid_init_masked,
                    e_resid=e_flux_masked,
                    detrending_algorithm=ss.detrending_algorithm,
                    n_iter=ss.n_sysrem_iter,
                    tolerance=ss.sysrem_convergence_tol,
                    max_converge_iter=ss.sysrem_max_convergence_iter,
                    diff_method=ss.sysrem_diff_method,
                    sigma_threshold=ss.sigma_threshold_sysrem_piskunov,)
                
                # Update the residual + coeff arrays using our adopted px mask
                # for only those pixels that we ran SYSREM on--everything else
                # will be NaN.
                resid_all[:, :, spec_i, unmasked_px] = resid
                coeff_wave_all[:, spec_i, unmasked_px] = coeff_wave
                coeff_phase_all[:, spec_i] = coeff_phase

        #----------------------------------------------------------------------
        # Option 2: Simultaneous detrending
        #----------------------------------------------------------------------
        else:
            print("Running detrending on all orders:")

            resid, coeff_phase, coeff_wave = sr.detrend_spectra(
                resid_init=resid_init,
                e_resid=e_resid,
                detrending_algorithm=ss.detrending_algorithm,
                n_iter=ss.n_sysrem_iter,
                tolerance=ss.sysrem_convergence_tol,
                max_converge_iter=ss.sysrem_max_convergence_iter,
                diff_method=ss.sysrem_diff_method,
                sigma_threshold=ss.sigma_threshold_sysrem_piskunov,)

            resid_all = resid.reshape(
                (ss.n_sysrem_iter+1, n_phase, n_spec, n_px))
            
            # TODO: store coefficients
            pass

        #----------------------------------------------------------------------
        # Storing results
        #----------------------------------------------------------------------
        # Store coefficients
        labels_per_seq.append("{}{}".format(transit_i+1, seq))
        phases_per_seq.append(np.arange(0, n_phase)[seq_mask])
        coeff_phase_all_seq.append(coeff_phase_all)
        coeff_wave_all_seq.append(coeff_wave_all)

        # Save residuals
        # TODO: save coefficients as well
        rv_frame = "stellar" if ss.run_sysrem_in_stellar_frame else "telluric"
        tu.save_sysrem_residuals_to_fits(
            fits_load_dir=ss.save_path,
            label=ss.label,
            n_transit=ss.n_transit,
            sysrem_resid=resid_all,
            transit_i=transit_i,
            sequence=seq,
            rv_frame=rv_frame,)

        #----------------------------------------------------------------------
        # Plots and diagnostics
        #----------------------------------------------------------------------
        # Plot titles
        plot_title = "Night {} ({}): {} [{}]".format(
            transit_i+1, seq, ss.label, rv_frame)
        plot_label = "{}_n{}_{}_{}".format(
            rv_frame, transit_i+1, seq, ss.label)

        plot_folder = "plots/{}/".format(ss.label)

        # Plot the residuals themselves (n_sr_iter x n_spec panels)
        tplt.plot_sysrem_residuals(
            waves,
            resid_all,
            plot_label=plot_label,
            plot_title=plot_title,
            plot_folder=plot_folder,)
        
        # Plot standard deviation of each segment/phase as a function of SYSREM
        # iteration (n_spec panels)
        tplt.plot_sysrem_std(
            waves,
            resid_all,
            plot_label=plot_label,
            plot_title=plot_title,
            plot_folder=plot_folder,)

# Plot comparison figures of the SYSREM phase and wavelength coefficients for
# all sequences.
plot_label = "{}_{}".format(rv_frame, ss.label)

tplt.plot_sysrem_coefficients(
    waves=waves,
    phases_list=phases_per_seq,
    labels_per_seq_list=labels_per_seq,
    coeff_phase_all_list=coeff_phase_all_seq,
    coeff_wave_all_list=coeff_wave_all_seq,
    plot_label=plot_label,
    plot_folder=plot_folder,)