"""
Script to run the transit modelling using the Aronson method. Assumes that a 
data file has already been prepared using prepare_transit_model_fits.py.
"""
import numpy as np
import pandas as pd
import transit.model as tmod
import transit.utils as tu
import transit.plotting as tplt
import astropy.constants as const
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Read in pre-prepared fits file
# -----------------------------------------------------------------------------
save_path = ""

# Parameters of the simulation
planet_transmission_boost_fac = 1
target_snr = 100
label = "simulated_R100000_wasp107"
n_transit = 2

# Build simulation filename
fn_label = "{}_trans_boost_x{:0.0f}_SNR{:0.0f}".format(
    label, planet_transmission_boost_fac, target_snr)

# Load in prepared fits file for our transit
waves, fluxes_list, sigmas_list, det, orders, transit_info_list, syst_info = \
    tu.load_transit_info_from_fits(save_path, fn_label, n_transit)

# Combine transits - stack observations allong phase dimension
obs_spec = np.vstack(fluxes_list)

# Add in transit number column to each dataframe, then stack
for t_i in range(n_transit):
    transit_info_list[t_i]["transit_num"] = t_i
    transit_info_list[t_i].reset_index(inplace=True)
    transit_info_list[t_i].rename(columns={"index":"obs_i"}, inplace=True)

transit_info = pd.concat(transit_info_list, axis=0)
transit_info.index = np.arange(len(transit_info))

# Reorder columns so we have transit_num, obs_i, then everything else
columns = np.concatenate(
    (["transit_num", "obs_i"], transit_info.columns.values[1:-1]))
transit_info = transit_info[columns]

# -----------------------------------------------------------------------------
# Running on subset of data
# -----------------------------------------------------------------------------
# Run on a subset of all spectral segments. By default the segments are
# ordered by detector, so the three segments making up each order will be
# separated by 6 in the array.
run_on_sub_slice = False
segments_to_keep = [2, 8, 14]

if run_on_sub_slice:
    segment_mask = np.full(waves.shape[0], False)
    segment_mask[segments_to_keep] = True
    waves = waves[segment_mask]
    obs_spec = obs_spec[:,segment_mask]

else:
    segment_mask = np.full(waves.shape[0], True)

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
lambda_treg_star = 100
lambda_treg_tau = 100
lambda_treg_planet = 1E6
tau_nr_tolerance = 1E-5
model_converge_tolerance = 1E-5
max_model_iter = 100
max_tau_nr_iter = 300
do_plot = False
print_every_n_iterations = 1

# Set a reasonable limit for the maximum fluxes. Recommend not to use currently
# it seems to prevent the model fully exploring the parameter space.
max_flux = np.max(obs_spec)

# Limits (low, high). Set to (None, None) for no limits. Note that to prevent
# division by zero and overflow errors min telluric_trans should be > 0.
stellar_flux_limits = (1E-5, max_flux)
telluric_trans_limits = (4.5E-5, None)          # Max optical depth ~10
telluric_tau_limits = (0, -np.log(4.5E-5))
planet_trans_limits = (1E-5, 1)
scale_limits = (1E-5, 1)
model_limits = (1E-5, None)

# -----------------------------------------------------------------------------
# Debugging settings
# -----------------------------------------------------------------------------
# Load in the flux, tau, and trans vectors used to simulate this transit.
# Note that the flux and trans vectors will have shape [n_spec, n_px], the tau
# vector will have shape [n_trans, n_spec, n_px] and the scale vector will be
# a list of length n_trans containing (likely unequal) vectors of n_phase.
component_flux, component_tau, component_trans, component_scale = \
    tu.load_simulated_transit_components_from_fits(
        fits_load_dir=save_path,
        label=fn_label,
        n_transit=n_transit,)

# Slice these so we're only considering the segments we're interested in, and
# concatenate the scale vector to be 1D.
fixed_flux = component_flux[segment_mask]
fixed_tau = component_tau[:, segment_mask]
fixed_trans = component_trans[segment_mask]
fixed_scale = np.concatenate(component_scale)

# Setting any of these to true will initialise *and* fix that particular vector
# to the fixed value above.
do_fix_flux_vector = False
do_fix_tau_vector = False
do_fix_trans_vector = False
do_fix_scale_vector = True

# Setting any of these to true will initialise that particular vector to the 
# fixed value above, *but* leave that vector free to float during fitting so
# long as do_fix_<vector> is False. This allows e.g. the telluric vector to be
# initialised to a molecfit fit, but still allow it to be optimised to the data
init_with_flux_vector = True
init_with_tau_vector = True
init_with_trans_vector = False
init_with_scale_vector = False

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
    max_model_iter=max_model_iter,
    max_tau_nr_iter=max_tau_nr_iter,
    do_plot=do_plot,
    print_every_n_iterations=print_every_n_iterations,
    do_fix_flux_vector=do_fix_flux_vector,
    do_fix_trans_vector=do_fix_trans_vector,
    do_fix_tau_vector=do_fix_tau_vector,
    do_fix_scale_vector=do_fix_scale_vector,
    fixed_flux=fixed_flux,
    fixed_trans=fixed_trans,
    fixed_tau=fixed_tau,
    fixed_scale=fixed_scale,
    init_with_flux_vector=init_with_flux_vector,
    init_with_tau_vector=init_with_tau_vector,
    init_with_trans_vector=init_with_trans_vector,
    init_with_scale_vector=init_with_scale_vector,)

# Save the results
tu.save_transit_model_results_to_fits(
    fits_load_dir="",
    label=fn_label,
    n_transit=n_transit,
    flux=flux,
    trans=trans,
    tau=tau,
    scale=scale,
    mask=mask,)

plt.close("all")

# Diagnostic plots
# Plot the component spectra (one pdf per transit).
for trans_i in range(n_transit):
    tm = transit_info["transit_num"] == trans_i

    tplt.plot_component_spectra(
        waves=waves,
        fluxes=flux,
        telluric_tau=tau[trans_i],
        planet_trans=trans,
        scale_vector=scale[tm],
        transit_num=trans_i,
        star_name=fn_label,)

tplt.plot_epoch_model_comp(
    waves=waves,
    obs_spec=obs_spec,
    model=model,
    fluxes=flux,
    telluric_tau=tau,
    planet_trans=trans,
    scale=scale,
    transit_info=transit_info,
    ref_fluxes=fixed_flux,
    ref_telluric_tau=fixed_tau,
    ref_planet_trans=fixed_trans,
    ref_scale=fixed_scale,)