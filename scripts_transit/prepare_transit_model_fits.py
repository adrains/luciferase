"""Script to initialise spectra and transit parameters in preparation for
running the Aronson method. For this we require:
 - A CSV file containing literature information about the system and planet
   properties with the columns [parameter, value, sigma, reference, comment]
 - A set of directories containing spectra reduced using the CRIRES+ pipeline,
   where each directory corresponds to a set of observations for a single 
   transit.

This script produces a fits file containing the formated spectra and
information necessary to run the Aronson method.
"""
import numpy as np
import transit.utils as tu
import transit.plotting as tplt
import transit.tables as ttab
import luciferase.spectra as ls
import luciferase.utils as lu

# -----------------------------------------------------------------------------
# Setup and Options
# -----------------------------------------------------------------------------
# Import our settings from a separate YAML file
data_prep_settings_file = "scripts_transit/data_preparation_settings.yml"
ds = tu.load_yaml_settings(data_prep_settings_file)

# -----------------------------------------------------------------------------
# Load in all sets of transits
# -----------------------------------------------------------------------------
# Load planetary properties file
print("Loading properties file...")
syst_info = tu.load_planet_properties(ds.planet_properties_file)

# Initial arrays
waves_all = []
fluxes_all = []
sigmas_all = []
detectors_all = []
orders_all = []
transit_info_all = []
nod_positions_all = []

# Cleaned + split arrays
fluxes_cleaned_all = []
sigmas_cleaned_all = []
bad_px_mask_all = []

cc_rv_shifts_all = []
cc_ds_shifts_all = []

# Extract data for each transit
for transit_i, trans_dir in enumerate(ds.planet_root_dirs):
    # Extract sorted time series data
    print("Loading spectra for transit #{}...".format(transit_i))
    waves, fluxes, sigmas, detectors, orders, transit_info = \
        tu.extract_nodding_time_series(
            root_dir=trans_dir,
            nod_ab_wildcard=ds.nod_ab_wildcard)
    
    # Grab dimensions for convenience
    (n_phase, n_spec, n_px) = fluxes.shape

    # TODO: we might need to do a cross correlation and interpolate the A/B
    # frame wavelength scale in cases where the PSF is smaller than the slit 
    # width.
    pass

    # Save all the arrays for this transit. Note that for convience when using
    # interpolate_wavelength_scale later we broadcast wave_out to all phases.
    waves_all.append(waves)
    fluxes_all.append(fluxes)
    sigmas_all.append(sigmas)
    detectors_all.append(detectors)
    orders_all.append(orders)
    transit_info_all.append(transit_info)

    # For convenience of regridding later, we also save the nod positions
    nod_positions_all.append(transit_info["nod_pos"].values)

# -----------------------------------------------------------------------------
# Template spectra
# -----------------------------------------------------------------------------
# Molecfit telluric model (continuum normalisation + cross-correlation)
telluric_wave, _, _, telluric_trans = tu.load_telluric_spectrum(
    molecfit_fits=ds.molecfit_fits[0],
    tau_fill_value=ds.tau_fill_value,
    convert_to_angstrom=False,
    convert_to_nm=True,
    output_transmission=True,)

# Stellar spectrum (continuum normalisation)
wave_stellar, spec_stellar = lu.load_plumage_template_spectrum(
    template_fits=ds.stellar_template_fits,
    do_convert_air_to_vacuum_wl=True,)

# -----------------------------------------------------------------------------
# Interpolate all fluxes across phase and transit onto common wavelength scale
# -----------------------------------------------------------------------------
# Stack all transits along the phase dimension to regid on inter-night basis
if ds.n_transit > 1:
    wave_stacked = np.vstack(waves_all)
    fluxes_stacked = np.vstack(fluxes_all)
    sigmas_stacked = np.vstack(sigmas_all)
    detectors_stacked = np.vstack(detectors_all)
    orders_stacked = np.vstack(orders_all)
    nod_positions_stacked = np.hstack(nod_positions_all)
else:
    wave_stacked = waves_all[0]
    fluxes_stacked = fluxes_all[0]
    sigmas_stacked = sigmas_all[0]
    detectors_stacked = detectors_all[0]
    orders_stacked = orders_all[0]
    nod_positions_stacked = nod_positions_all[0]

# Sort in wavelength order
wave_ii = np.argsort(np.median(wave_stacked[0], axis=1))
wave_stacked = wave_stacked[:,wave_ii,:]
fluxes_stacked = fluxes_stacked[:,wave_ii,:]
sigmas_stacked = sigmas_stacked[:,wave_ii,:]
detectors_stacked = detectors_stacked[:,wave_ii]
orders_stacked = orders_stacked[:,wave_ii]

# [Optional] Load in continuum polynomial coefficients, which we need to scale
# the molecfit transmission spectrum to match the observations. Note this is
# currently circular/iterative in that this script needs to be run twice now:
# once with the use_telluric_model_as_ref set to False, at which point we save
# the polynomial coefficients to the fits file, then again with it set to true
# so that we can use the polynomial coefficients when doing the 
# cross-correlation using the telluric spectrum as a template. TODO: fix.
if ds.use_telluric_model_as_ref:
    _, _, _, poly_coeff = tu.load_normalised_spectra_from_fits(
        fits_load_dir=ds.save_path,
        label=ds.star_name,
        n_transit=ds.n_transit,
        transit_i=0,)
    
    poly_coeff = np.nanmean(poly_coeff, axis=0) # Shape: [n_phase, n_coeff]

else:
    poly_coeff = None

# Using the stacked data, regrid onto a single common wavelength scale
# Regrid the stacked data onto a single wavelength scale and correct offsets in
# the wavelength scale between A/B nodding positions. This function has a few
# options:
#   1) Regrid using a Molecfit telluric model (use_telluric_model_as_ref = True
#      poly_coeff != None)
#   2) Regrid using the spectrum from a specific phase + A/B position
#   2.1) If do_rigid_regrid_per_detector = True, then three cross-correlations
#        are done: one for each detector.
#   2.2) If do_rigid_regrid_per_detector = False, then n_orders_to_group is
#        used to determine how many adjacent orders on the same detector are
#        used for each cross-correlation.
wave_adopt, fluxes_interp_all, sigmas_interp_all, rvs_all, cc_rvs, cc_all = \
    tu.regrid_all_phases(
        waves=wave_stacked,
        fluxes=fluxes_stacked,
        sigmas=sigmas_stacked,
        detectors=detectors_stacked,
        nod_positions=nod_positions_stacked,
        reference_nod_pos=ds.reference_nod_pos,
        cc_range_kms=ds.cc_range_kms,
        cc_step_kms=ds.cc_step_kms,
        do_sigma_clipping=ds.do_sigma_clipping,
        sigma_clip_level_upper=ds.sigma_clip_level_upper,
        sigma_clip_level_lower=ds.sigma_clip_level_lower,
        make_debug_plots=ds.make_debug_plots,
        interpolation_method=ds.interpolation_method,
        do_rigid_regrid_per_detector=ds.do_rigid_regrid_per_detector,
        n_ref_div=ds.n_ref_div,
        n_orders_to_group=ds.n_orders_to_group,
        use_telluric_model_as_ref=ds.use_telluric_model_as_ref,
        telluric_wave=telluric_wave,
        telluric_trans=telluric_trans,
        continuum_poly_coeff=poly_coeff,)

# -----------------------------------------------------------------------------
# Clean spectra + calculate timestep info
# -----------------------------------------------------------------------------
# Split our stacked arrays back into individual transits, clean/clip the data,
# and calculate information for each time step.
phase_low = 0
phase_high = fluxes_all[0].shape[0]

for transit_i in range(ds.n_transit):
    # Grab high extent
    phase_high = phase_low + fluxes_all[transit_i].shape[0]

    # Extract the fluxes and sigmas for our current transit
    fluxes = fluxes_interp_all[phase_low:phase_high]
    sigma = sigmas_interp_all[phase_low:phase_high]
    transit_info = transit_info_all[transit_i]

    # Calculate mus and rvs and add them to our existing dataframes
    print("Calculating time step info...")
    tu.calculate_transit_timestep_info(
        transit_info=transit_info,
        syst_info=syst_info,
        do_consider_vsini=ds.do_consider_vsini,)

    # Sigma clip observations
    print("Sigma clipping and cleaning data...")
    fluxes_interp_clipped, bad_px_mask = tu.sigma_clip_observations(
        obs_spec=fluxes,
        bad_px_replace_val="interpolate",
        time_steps=transit_info["jd_mid"].values,)

    print("{:,} bad pixels found.".format(np.sum(bad_px_mask)))
    
    fluxes_cleaned_all.append(fluxes_interp_clipped)
    sigmas_cleaned_all.append(sigma)
    bad_px_mask_all.append(bad_px_mask)
    cc_rv_shifts_all.append(rvs_all[phase_low:phase_high])

    # Update low bound
    phase_low = phase_high
    
# -----------------------------------------------------------------------------
# Consistency check
# -----------------------------------------------------------------------------
# Sanity check: order and detector arrays should be the same across all phases
# and all transits assuming our results were taken using the same instrument 
# settings. In this case, we can get away with only saving one master set with
# length equal to n_spec = n_order * n_detector
N_SPEC = detectors_stacked.shape[1]

for spec_i in range(N_SPEC):
    if len(set(detectors_stacked[:,spec_i])) > 1:
        raise Exception("Detector #s aren't equal across transit and phase.")
    
    if len(set(orders_stacked[:,spec_i])) > 1:
        raise Exception("Order #s aren't equal across transit and phase.")

# All detectors and order #s are consistent, save only one set of length n_spec
detectors = detectors_stacked[0]
orders = orders_stacked[0]

# Diagnostic plots
fluxes_cleaned_stacked = np.vstack(fluxes_cleaned_all)
tplt.plot_regrid_diagnostics_rv(rvs_all, wave_adopt, detectors)
tplt.plot_regrid_diagnostics_img(fluxes_cleaned_stacked, detectors, wave_adopt)

# -----------------------------------------------------------------------------
# Save data cube + timestep info
# -----------------------------------------------------------------------------
# Save all arrays and dataframes
tu.save_transit_info_to_fits(
    waves=wave_adopt,
    obs_spec_list=fluxes_cleaned_all,
    sigmas_list=sigmas_cleaned_all,
    n_transits=ds.n_transit,
    detectors=detectors,
    orders=orders,
    transit_info_list=transit_info_all,
    syst_info=syst_info,
    fits_save_dir=ds.save_path,
    label=ds.star_name,
    cc_rv_shifts_list=cc_rv_shifts_all,)

# -----------------------------------------------------------------------------
# Continuum normalise data + save
# -----------------------------------------------------------------------------
print("Continuum normalising spectra...")
for transit_i in range(ds.n_transit):
    # Grab shape for this night
    (n_phase, n_spec, n_px) = fluxes_all[transit_i].shape

    for night_i in range(len(fluxes_cleaned_all)):
        # HACK: clean sigma=0 values. TODO: move this earlier.
        is_zero = sigmas_cleaned_all[night_i] == 0
        sigmas_cleaned_all[night_i][is_zero] = 1E5

    fluxes_norm, sigmas_norm, poly_coeff = \
        ls.continuum_normalise_all_spectra_with_telluric_model(
            waves_sci=wave_adopt,
            fluxes_sci=fluxes_cleaned_all[transit_i],
            sigmas_sci=sigmas_cleaned_all[transit_i],
            wave_telluric=telluric_wave,
            trans_telluric=telluric_trans,
            wave_stellar=wave_stellar,
            spec_stellar=spec_stellar,
            bcors=transit_info_all[transit_i]["bcor"].values,
            rv_star=syst_info.loc["rv_star", "value"],
            airmasses=transit_info_all[transit_i]["airmass"].values,)

    # Construct bad px mask from tellurics
    print("Constructing bad px mask from tellurics...")
    bad_px_mask_1D = telluric_trans < ds.telluric_trans_bad_px_threshold
    bad_px_mask_3D = np.tile(
        bad_px_mask_1D, n_phase).reshape(n_phase, n_spec, n_px)

    # Save normalised spectra back to fits file
    tu.save_normalised_spectra_to_fits(
        fits_load_dir=ds.save_path,
        label=ds.star_name,
        n_transit=ds.n_transit,
        fluxes_norm=fluxes_norm,
        sigmas_norm=sigmas_norm,
        bad_px_mask_norm=bad_px_mask_3D,
        poly_coeff=poly_coeff,
        transit_i=transit_i,)

# -----------------------------------------------------------------------------
# LaTeX table
# -----------------------------------------------------------------------------
ttab.make_observations_table(
    transit_info_all, fluxes_cleaned_all, sigmas_cleaned_all)