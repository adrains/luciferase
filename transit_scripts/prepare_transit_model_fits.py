"""Script to initialise spectra and transit parameters in preparation for
running the Aronson method. For this we require:
 - A CSV file containing literature information about the system and planet
   properties with the columns [parameter, value, sigma, reference, comment]
 - A directory containing spectra reduced using the CRIRES+ pipeline

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
planet_properties_file = "transit_scripts/planet_data_wasp107.csv"

# Data directory
planet_root_dir = "/Users/arains/data/220310_WASP107/"

# -----------------------------------------------------------------------------
# Running things
# -----------------------------------------------------------------------------
# Load planetary properties file
print("Loading properties file...")
syst_info = tu.load_planet_properties(planet_properties_file)

# Extract sorted time series data
print("Loading spectra...")
waves, fluxes, sigmas, detectors, orders, transit_info = \
    tu.extract_nodding_time_series(root_dir=planet_root_dir)

# TODO: we might need to do a cross correlation and interpolate the A/B frame
# wavelength scale in cases where the PSF is smaller than the slit width.
pass

# Determine bad pixel mask for edge pixels (imin, imax)
print("Determining pixel limits...")
px_min, px_max = tu.compute_detector_limits(fluxes)

# Interpolate wavelength scale
print("Interpolating wavelength scale...")
wave_new, obs_spec = tu.interpolate_wavelength_scale(
    waves=waves,
    fluxes=fluxes,
    px_min=px_min,
    px_max=px_max,)

# Calculate mus and rvs and add them to our existing dataframes
print("Calculating time step info...")
tu.calculate_transit_timestep_info(transit_info, syst_info)

# Sigma clip observations
print("Sigma clipping and cleaning data...")
obs_spec_clipped, bad_px_mask = tu.sigma_clip_observations(
    obs_spec=obs_spec,
    bad_px_replace_val="interpolate",
    time_steps=transit_info["jd_mid"].values,)

print("{:,} bad pixels found.".format(np.sum(bad_px_mask)))

# Save all arrays and dataframes
tu.save_transit_info_to_fits(
    waves=wave_new,
    obs_spec=obs_spec,
    sigmas=sigmas,
    detectors=detectors,
    orders=orders,
    transit_info=transit_info,
    syst_info=syst_info,
    fits_save_dir=save_path,
    star_name=star_name,)