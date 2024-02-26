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
    "/Users/arains/data/220310_WASP107/",
    "/Users/arains/data/230222_WASP107/",
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

# Regridding diagnostic arrays
cc_rv_shifts_all = []
cc_ds_shifts_all = []

# Cleaned arrays
fluxes_cleaned_all = []
sigmas_cleaned_all = []
bad_px_mask_all = []

# Extract data for each transit
for transit_i, trans_dir in enumerate(planet_root_dirs):
    # Extract sorted time series data
    print("Loading spectra for transit #{}...".format(transit_i))
    waves, fluxes, sigmas, detectors, orders, transit_info = \
        tu.extract_nodding_time_series(root_dir=trans_dir)
    
    # Grab dimensions for convenience
    (n_phase, n_spec, n_px) = fluxes.shape

    # Regrid this night onto a single wavelength scale and correct offsets in
    # the wavelength scale over the night and between A/B nodding positions.
    # rvs and ds are the radial velocities and doppler shifts respectively
    # found for each spectral segment, but if do_rigid_regrid_per_detector is
    # True then we simply use the median of this on a per-detector basis for
    # the regridding.
    wave_out, fluxes_corr, sigmas_corr, rvs, ds, cc = tu.regrid_single_night(
        waves=waves,
        fluxes=fluxes, 
        sigmas=sigmas,
        detectors=detectors,
        orders=orders,
        nod_positions=transit_info["nod_pos"].values,
        reference_nod_pos="A",
        make_debug_plots=True,
        interpolation_method="linear",
        do_rigid_regrid_per_detector=False,)

    # TODO: we might need to do a cross correlation and interpolate the A/B
    # frame wavelength scale in cases where the PSF is smaller than the slit 
    # width.
    pass

    # Save all the arrays for this transit. Note that for convience when using
    # interpolate_wavelength_scale later we broadcast wave_out to all phases.
    waves_all.append(np.broadcast_to(wave_out, (n_phase, n_spec, n_px)))
    fluxes_all.append(fluxes)
    sigmas_all.append(sigmas)
    detectors_all.append(detectors)
    orders_all.append(orders)
    transit_info_all.append(transit_info)

    # Store the regridding diagnostics
    cc_rv_shifts_all.append(rvs)
    cc_ds_shifts_all.append(ds)

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
else:
    wave_stacked = waves_all[0]
    fluxes_stacked = fluxes_all[0]
    fluxes_stacked = sigmas_all[0]
    detectors_stacked = detectors_all[0]
    orders_stacked = orders_all[0]

# Determine bad pixel mask for edge pixels (imin, imax)
print("Determining pixel limits...")
px_min, px_max = tu.compute_detector_limits(fluxes_stacked)

# Interpolate wavelength scale
print("Interpolating wavelength scale...")
wave_interp, fluxes_interp, sigmas_interp = tu.interpolate_wavelength_scale(
    waves=wave_stacked,
    fluxes=fluxes_stacked,
    sigmas=sigmas_stacked,
    px_min=px_min,
    px_max=px_max,)

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
    fluxes = fluxes_interp[phase_low:phase_high]
    sigma = sigmas_interp[phase_low:phase_high]
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

# -----------------------------------------------------------------------------
# Saving fits
# -----------------------------------------------------------------------------
# Save all arrays and dataframes
tu.save_transit_info_to_fits(
    waves=wave_interp,
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