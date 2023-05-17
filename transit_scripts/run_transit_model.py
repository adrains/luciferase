"""
Script to run the transit modelling using the Aronson method. Assumes that a 
data file has already been prepared using prepare_transit_model_fits.py.
"""
import numpy as np
import transit.model as tmod
import transit.utils as tu
import transit.plotting as tplt
import astropy.constants as const

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

# HACK to put mu_wgt in Nik's format. TODO discuss with Nik the correct
# formalism, specifically whether mu_wgt should be normalised or not.
R_SUN = const.R_sun.cgs.value
R_EARTH = const.R_earth.cgs.value

r_planet = (syst_info.loc["r_planet_rearth", "value"]*R_EARTH
    / (syst_info.loc["r_star_rsun", "value"]*R_SUN))

transit_info["planet_area_frac_mid"] = \
    transit_info["planet_area_frac_mid"].values * (np.pi * r_planet**2)

# -----------------------------------------------------------------------------
# Inverse model settings
# -----------------------------------------------------------------------------
lambda_treg_star = 1
lambda_treg_tau = 100      # 1E-3
lambda_treg_planet = 1   # 1E5
tau_nr_tolerance = 1E-5
model_converge_tolerance = 1E-5
max_iter = 3000
do_plot = False
print_every_n_iterations = 1

# Set a reasonable limit for the maximum fluxes. Recommend not to use currently
# it seems to prevent the model fully exploring the parameter space.
max_flux = None # 2*np.max(fluxes_list[0])

# Limits (low, high). Set to (None, None) for no limits. Note that to prevent
# division by zero and overflow errors min telluric_trans should be > 0.
stellar_flux_limits = (1E-5, None)
telluric_trans_limits = (4.5E-5, None)          # Max optical depth ~10
telluric_tau_limits = (0, -np.log(4.5E-5))
planet_trans_limits = (1E-5, None)
scale_limits = (1E-5, 1)
model_limits = (1E-5, None)

# -----------------------------------------------------------------------------
# Debugging settings
# -----------------------------------------------------------------------------
# Load in the flux, tau, and trans vectors used to simulate this transit
component_flux, component_tau, component_trans = \
    tu.load_simulated_transit_components_from_fits(
        fits_load_dir=save_path,
        label=file_label,
        n_transit=n_transit,)

fixed_flux = component_flux[segment_mask]
fixed_tau = component_tau[segment_mask]
fixed_trans = component_trans[segment_mask]
fixed_scale = np.ones(obs_spec.shape[0])

# Setting any of these to true will fix that particular vector during fitting
do_fix_flux_vector = False
do_fix_tau_vector = False
do_fix_trans_vector = False
do_fix_scale_vector = False

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
    print_every_n_iterations=print_every_n_iterations,
    do_fix_flux_vector=do_fix_flux_vector,
    do_fix_trans_vector=do_fix_trans_vector,
    do_fix_tau_vector=do_fix_tau_vector,
    do_fix_scale_vector=do_fix_scale_vector,
    fixed_flux=fixed_flux,
    fixed_trans=fixed_trans,
    fixed_tau=fixed_tau,
    fixed_scale=fixed_scale,)

# Save the results
tu.save_transit_model_results_to_fits(
    fits_load_dir="",
    label=file_label,
    n_transit=n_transit,
    flux=flux,
    trans=trans,
    tau=tau,
    scale=scale,
    mask=mask,)

# Diagnostic plots
tplt.plot_component_spectra(waves, flux, tau, trans,)