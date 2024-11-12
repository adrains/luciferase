"""Script to continuum normalise input datacube.
"""
import numpy as np
import transit.utils as tu
import luciferase.utils as lu
import luciferase.spectra as lsp

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
# Continuum normalise data + save
# -----------------------------------------------------------------------------
print("Continuum normalising spectra...")
for transit_i in range(ss.n_transit):
    # Grab shape for this night
    (n_phase, n_spec, n_px) = fluxes_list[transit_i].shape
    
    # Run continuum normalisation once for each A/B sequence
    if ss.do_split_sequences_continuum_norm:
        # Create arrays
        fluxes_norm = np.zeros()
        sigmas_norm = np.zeros()
        poly_coeff = np.zeros()
    # Continuum normalise the entire cube at once
    else:
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
                airmasses=transit_info_list[transit_i]["airmass"].values,)

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
        fluxes_norm=fluxes_norm,
        sigmas_norm=sigmas_norm,
        bad_px_mask_norm=bad_px_mask_3D,
        poly_coeff=poly_coeff,
        transit_i=transit_i,)