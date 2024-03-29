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

# -----------------------------------------------------------------------------
# Setup and Options
# -----------------------------------------------------------------------------
# Save path
save_path = ""
star_name = "wasp107"

# Location of the 
planet_properties_file = "scripts_transit/planet_data_wasp107.csv"

# Data directories where each folder contains observations for a single transit
planet_root_dirs = [
    #"/home/arains/gto/220310_WASP107",
    #"/home/arains/gto/230222_WASP107",
    #"/Users/arains/data/220310_WASP107/",
    #"/Users/arains/data/230222_WASP107/",
    "/Users/arains/data/wasp107b_final/wasp107b_N1_20220310/",
    "/Users/arains/data/wasp107b_final/wasp107b_N2_20230222/",
]

N_TRANSITS = len(planet_root_dirs)

# -----------------------------------------------------------------------------
# Load in all sets of transits
# -----------------------------------------------------------------------------
# Load planetary properties file
print("Loading properties file...")
syst_info = tu.load_planet_properties(planet_properties_file)

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
for transit_i, trans_dir in enumerate(planet_root_dirs):
    # Extract sorted time series data
    print("Loading spectra for transit #{}...".format(transit_i))
    waves, fluxes, sigmas, detectors, orders, transit_info = \
        tu.extract_nodding_time_series(root_dir=trans_dir)
    
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
# Interpolate all fluxes across phase and transit onto common wavelength scale
# -----------------------------------------------------------------------------
# Stack all transits along the phase dimension to regid on inter-night basis
if N_TRANSITS > 1:
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

# Using the stacked data, regrid onto a single common wavelength scale
# Regrid the stacked data onto a single wavelength scale and correct offsets in
# the wavelength scale between A/B nodding positions.rvs and ds are the radial
# velocities and doppler shifts respectively found for each spectral segment, 
# but if do_rigid_regrid_per_detector is True then we simply use the median of
# this on a per-detector basis for the regridding.
wave_adopt, fluxes_interp_all, sigmas_interp_all, rvs_all, cc_rvs, cc_all = \
    tu.regrid_all_phases(
        waves=wave_stacked,
        fluxes=fluxes_stacked,
        sigmas=sigmas_stacked,
        detectors=detectors_stacked,
        nod_positions=nod_positions_stacked,
        reference_nod_pos="A",
        cc_range_kms=(-20,20),
        cc_step_kms=0.1,
        do_sigma_clipping=True,
        sigma_clip_level_upper=3,
        sigma_clip_level_lower=3,
        make_debug_plots=False,
        interpolation_method="linear",
        do_rigid_regrid_per_detector=False,
        n_ref_div=5,
        n_orders_to_group=1,)

# Determine bad pixel mask for edge pixels (imin, imax)
#print("Determining pixel limits...")
#px_min, px_max = tu.compute_detector_limits(fluxes_stacked)

# -----------------------------------------------------------------------------
# Main operation
# -----------------------------------------------------------------------------
# Split our stacked arrays back into individual transits, clean/clip the data,
# and calculate information for each time step.
phase_low = 0
phase_high = fluxes_all[0].shape[0]

for transit_i in range(N_TRANSITS):
    # Grab high extent
    phase_high = phase_low + fluxes_all[transit_i].shape[0]

    # Extract the fluxes and sigmas for our current transit
    fluxes = fluxes_interp_all[phase_low:phase_high]
    sigma = sigmas_interp_all[phase_low:phase_high]
    transit_info = transit_info_all[transit_i]

    # Calculate mus and rvs and add them to our existing dataframes
    print("Calculating time step info...")
    tu.calculate_transit_timestep_info(transit_info, syst_info)

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
# Saving fits
# -----------------------------------------------------------------------------
# Save all arrays and dataframes
tu.save_transit_info_to_fits(
    waves=wave_adopt,
    obs_spec_list=fluxes_cleaned_all,
    sigmas_list=sigmas_cleaned_all,
    n_transits=N_TRANSITS,
    detectors=detectors,
    orders=orders,
    transit_info_list=transit_info_all,
    syst_info=syst_info,
    fits_save_dir=save_path,
    label=star_name,
    cc_rv_shifts_list=cc_rv_shifts_all,)

# -----------------------------------------------------------------------------
# LaTeX table
# -----------------------------------------------------------------------------
ttab.make_observations_table(
    transit_info_all, fluxes_cleaned_all, sigmas_cleaned_all)