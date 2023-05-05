"""
Script to run the transit modelling using the Aronson method. Assumes that a 
data file has already been prepared using prepare_transit_model_fits.py.
"""
import numpy as np
import transit.model as tmod
import transit.utils as tu

# -----------------------------------------------------------------------------
# Read in pre-prepared fits file
# -----------------------------------------------------------------------------
save_path = ""
file_label = "simulated_wasp107"
n_transit = 2

# Load in prepared fits file for our transit
waves, fluxes_list, sigmas_list, det, orders, transit_info_list, syst_info = \
    tu.load_transit_info_from_fits(save_path, file_label, n_transit)

# Run on only a single transit for now
obs_spec = fluxes_list[0]
transit_info = transit_info_list[0]

# Run on a subset of all spectral segments. By default the segments are
# ordered by detector, so the three segments making up each order will be
# separated by 6 in the array.
run_on_sub_slice = True
segments_to_keep = [0, 6, 12]
segment_mask = np.full(waves.shape[0], False)
segment_mask[segments_to_keep] = True

if run_on_sub_slice:
    waves = waves[segment_mask]
    obs_spec = obs_spec[:,segment_mask]

# -----------------------------------------------------------------------------
# Inverse model settings
# -----------------------------------------------------------------------------
lambda_treg_star = 1
lambda_treg_tau = None      #1E-3
lambda_treg_planet = None   #1E5
tau_nr_tolerance = 1E-5
model_converge_tolerance = 1E-5
max_iter = 4000
do_plot = False
print_every_n_iterations = 1

# Set a reasonable limit for the maximum fluxes
max_flux = 2*np.max(fluxes_list[0])

# Limits (low, high). Set to (None, None) for no limits. Note that to prevent
# division by zero and overflow errors min telluric_trans should be > 0.
stellar_flux_limits = (1E-5, max_flux)
telluric_trans_limits = (4.5E-5, max_flux)          # Max optical depth ~10
telluric_tau_limits = (0, -np.log(4.5E-5))
planet_trans_limits = (1E-5, None)
scale_limits = (1E-5, None)
model_limits = (1E-5, None)

# -----------------------------------------------------------------------------
# Running modelling
# -----------------------------------------------------------------------------
flux, trans, tau, scale, model, mask = tmod.run_transit_model(
    waves=waves,
    obs_spec=obs_spec,
    transit_info=transit_info,
    syst_info=syst_info,
    lambda_treg_star=lambda_treg_star,
    lambda_treg_tau=lambda_treg_tau,
    lambda_treg_planet=lambda_treg_planet,
    tau_nr_tolerance=tau_nr_tolerance,
    model_converge_tolerance=model_converge_tolerance,
    stellar_flux_limits=stellar_flux_limits,
    telluric_trans_limits=telluric_trans_limits,
    telluric_tau_limits=telluric_tau_limits,
    planet_trans_limits=planet_trans_limits,
    scale_limits=scale_limits,
    model_limits=model_limits,
    max_iter=max_iter,
    do_plot=do_plot,
    print_every_n_iterations=print_every_n_iterations,)

# Save the results
pass

# Diagnostic plots
pass