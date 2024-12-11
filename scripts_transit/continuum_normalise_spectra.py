"""Script to continuum normalise spectra and perform cleaning + masking.
"""
import numpy as np
import transit.utils as tu
import luciferase.utils as lu
import luciferase.spectra as lsp
import transit.plotting as tplt

#------------------------------------------------------------------------------
# Settings + import
#------------------------------------------------------------------------------
simulation_settings_file = "scripts_transit/sysrem_settings.yml"
ss = tu.load_yaml_settings(simulation_settings_file)

waves, fluxes_list, sigmas_list, det, orders, transit_info_list, syst_info = \
    tu.load_transit_info_from_fits(ss.save_path, ss.label, ss.n_transit)

# -----------------------------------------------------------------------------
# Load template spectra for continuum normalisation
# -----------------------------------------------------------------------------
# Molecfit telluric model (continuum normalisation + cross-correlation)
telluric_wave, _, _, telluric_trans = tu.load_telluric_spectrum(
    molecfit_fits=ss.molecfit_fits[0],
    tau_fill_value=ss.tau_fill_value,
    convert_to_angstrom=False,
    convert_to_nm=True,
    output_transmission=True,)

# Stellar spectrum (continuum normalisation)
wave_stellar, spec_stellar = lu.load_plumage_template_spectrum(
    template_fits=ss.stellar_template_fits,
    do_convert_air_to_vacuum_wl=True,)

# -----------------------------------------------------------------------------
# Info print
# -----------------------------------------------------------------------------
print("-"*80, "\nContinuum Normalisation Settings\n", "-"*80, sep="")
if ss.save_path == "":
    print("\tSpectra fits\t\t\ttransit_data_{}_n{}.fits".format(
        ss.label, ss.n_transit))
else:
    print("\tSpectra fits\t\t\t{}/transit_data_{}_n{}.fits".format(
        ss.save_path, ss.label, ss.n_transit))

print("\tSigma Threshold Phase\t\t{}".format(ss.sigma_threshold_phase))
print("\tSigma Threshold Column\t\t{}".format(ss.sigma_threshold_column))
print("\tBad px extrapolation?\t\t{}".format(
    ss.do_bad_px_corr_via_extrapolation))
print("\tSplit A/B seq during norm?\t{}".format(
    ss.do_split_sequences_continuum_norm))
print("-"*80)

# -----------------------------------------------------------------------------
# Continuum normalise data + save
# -----------------------------------------------------------------------------
for transit_i in range(ss.n_transit):
    print("\nContinuum normalising transit {}/{}...".format(
        transit_i+1, ss.n_transit))
    
    # Grab shape for this night
    (n_phase, n_spec, n_px) = fluxes_list[transit_i].shape
    
    plot_folder = "plots/{}/".format(ss.label)

    # Run continuum normalisation once for each A/B sequence
    if ss.do_split_sequences_continuum_norm:
        # Create arrays
        fluxes_norm = np.zeros_like(fluxes_list[transit_i])
        sigmas_norm = np.zeros_like(sigmas_list[transit_i])
        poly_coeff = np.zeros((n_phase, n_spec, 2))

        # Grab sequences
        nod_pos = transit_info_list[transit_i]["nod_pos"].values
        sequence_masks = [nod_pos == "A", nod_pos == "B"]
        sequences = ["A", "B"]

        # Loop over both sequences
        for seq, sm in zip(sequences, sequence_masks):
            print("\tNormalising {} sequence".format(seq))
            fluxes_seq, sigmas_seq, poly_coeff_seq = \
                lsp.continuum_normalise_all_spectra_with_telluric_model(
                    waves_sci=waves,
                    fluxes_sci=fluxes_list[transit_i][sm],
                    sigmas_sci=sigmas_list[transit_i][sm],
                    wave_telluric=telluric_wave,
                    trans_telluric=telluric_trans,
                    wave_stellar=wave_stellar,
                    spec_stellar=spec_stellar,
                    bcors=transit_info_list[transit_i]["bcor"].values[sm],
                    rv_star=syst_info.loc["rv_star", "value"],
                    airmasses=\
                        transit_info_list[transit_i]["airmass"].values[sm],
                    seq=seq,
                    plot_folder=plot_folder,)
            
            # Store
            fluxes_norm[sm] = fluxes_seq
            sigmas_norm[sm] = sigmas_seq
            poly_coeff[sm] = poly_coeff_seq

    # Continuum normalise the entire cube at once
    else:
        print("\tNormalising combined A/B sequence")
        fluxes_norm, sigmas_norm, poly_coeff = \
            lsp.continuum_normalise_all_spectra_with_telluric_model(
                waves_sci=waves,
                fluxes_sci=fluxes_list[transit_i],
                sigmas_sci=sigmas_list[transit_i],
                wave_telluric=telluric_wave,
                trans_telluric=telluric_trans,
                wave_stellar=wave_stellar,
                spec_stellar=spec_stellar,
                bcors=transit_info_list[transit_i]["bcor"].values,
                rv_star=syst_info.loc["rv_star", "value"],
                airmasses=transit_info_list[transit_i]["airmass"].values,
                seq="AB",
                plot_folder=plot_folder,)

    # -------------------------------------------------------------------------
    # Clean continuum normalised spectra
    # -------------------------------------------------------------------------
    print("\tCleaning and interpolating...")
    mjds = transit_info_list[transit_i]["mjd_mid"].values
    nod_pos = transit_info_list[transit_i]["nod_pos"].values
    is_A = nod_pos == "A"

    spectra_clean, e_spectra_clean, interp_px_mask = \
        tu.clean_and_interpolate_spectra(
            spectra=fluxes_norm,
            e_spectra=sigmas_norm,
            mjds=mjds,
            is_A=is_A,
            sigma_threshold_phase=ss.sigma_threshold_phase,
            sigma_threshold_column=ss.sigma_threshold_column,
            do_extrapolation=ss.do_bad_px_corr_via_extrapolation,)

    plot_label = "{}_N{}".format(ss.label, transit_i+1)

    tplt.plot_cleaned_continuum_normalised_spectra(
        waves=waves,
        spectra_init=fluxes_norm,
        spectra_clean=spectra_clean,
        sequences=["A","B"],
        sequence_masks=[is_A, ~is_A],
        plot_label=plot_label,
        plot_title=plot_label,
        plot_folder=plot_folder,)

    # -------------------------------------------------------------------------
    # Save cleaned spectra
    # -------------------------------------------------------------------------
    # Save normalised spectra back to fits file
    tu.save_normalised_spectra_to_fits(
        fits_load_dir=ss.save_path,
        label=ss.label,
        n_transit=ss.n_transit,
        fluxes_norm=spectra_clean,
        sigmas_norm=e_spectra_clean,
        bad_px_mask_norm=interp_px_mask,
        poly_coeff=poly_coeff,
        transit_i=transit_i,)