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
# Load in settings file
# -----------------------------------------------------------------------------
yaml_settings_file = "scripts_transit/transit_model_settings.yml"
ms = tu.load_yaml_settings(yaml_settings_file)

# -----------------------------------------------------------------------------
# Read in pre-prepared fits file
# -----------------------------------------------------------------------------
# Load in prepared fits file for our transit
waves, fluxes_list, sigmas_list, det, orders, transit_info_list, syst_info = \
    tu.load_transit_info_from_fits(ms.save_path, ms.fn_label, ms.n_transit)

# Combine transits - stack observations allong phase dimension
obs_spec = np.vstack(fluxes_list)

# Add in transit number column to each dataframe, then stack
for t_i in range(ms.n_transit):
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
# HACK: Reset mu_wgt
# -----------------------------------------------------------------------------
# HACK to put mu_wgt in Nik's format. TODO discuss with Nik the correct
# formalism, specifically whether mu_wgt should be normalised or not.
R_SUN = const.R_sun.cgs.value
R_EARTH = const.R_earth.cgs.value

r_planet = (syst_info.loc["r_planet_rearth", "value"]*R_EARTH
    / (syst_info.loc["r_star_rsun", "value"]*R_SUN))

transit_info["planet_area_frac_mid"] = \
    transit_info["planet_area_frac_mid"].values * (np.pi * r_planet**2)

# -----------------------------------------------------------------------------
# Mask segments
# -----------------------------------------------------------------------------
# Create a mask if we're choosing to only run on a subset of orders. This mask
# gets applied to the input data, as well as the component spectra.
if ms.run_on_sub_slice:
    segment_mask = np.full(waves.shape[0], False)
    segment_mask[ms.segments_to_keep] = True
    waves = waves[segment_mask]
    obs_spec = obs_spec[:,segment_mask]

else:
    segment_mask = np.full(waves.shape[0], True)

# -----------------------------------------------------------------------------
# Initialise component vectors
# -----------------------------------------------------------------------------
# Load in component vectors for flux, tau, trans and scale.
# Note that the flux and trans vectors will have shape [n_spec, n_px], the tau
# vector will have shape [n_trans, n_spec, n_px] and the scale vector will be
# a list of length n_trans containing (likely unequal) vectors of n_phase.

# For simulated data, we know these vectors perfectly.
if ms.is_simulated_data:
    component_flux, component_tau, component_trans, component_scale = \
        tu.load_simulated_transit_components_from_fits(
            fits_load_dir=ms.save_path,
            label=ms.fn_label,
            n_transit=ms.n_transit,)
    
# Otherwise we're working with real data, and do not have perfect info of our
# respective components.
else:
    raise Exception("Not implemented!")

# Slice these so we're only considering the segments we're interested in, and
# concatenate the scale vector to be 1D.
fixed_flux = component_flux[segment_mask]
fixed_tau = component_tau[:, segment_mask]
fixed_trans = component_trans[segment_mask]
fixed_scale = np.concatenate(component_scale)

# -----------------------------------------------------------------------------
# Running modelling
# -----------------------------------------------------------------------------
flux, trans, tau, scale, model, mask = tmod.run_transit_model(
    waves=waves,
    obs_spec=obs_spec,
    transit_info=transit_info,
    syst_info=syst_info,
    lambda_treg_star=ms.lambda_treg_star,
    lambda_treg_tau=ms.lambda_treg_tau,
    lambda_treg_planet=ms.lambda_treg_planet,
    tau_nr_tolerance=ms.tau_nr_tolerance,
    model_converge_tolerance=ms.model_converge_tolerance,
    stellar_flux_limits=ms.stellar_flux_limits,
    telluric_trans_limits=ms.telluric_trans_limits,
    telluric_tau_limits=ms.telluric_tau_limits,
    planet_trans_limits=ms.planet_trans_limits,
    scale_limits=ms.scale_limits,
    model_limits=ms.model_limits,
    max_model_iter=ms.max_model_iter,
    max_tau_nr_iter=ms.max_tau_nr_iter,
    do_plot=ms.do_plot,
    print_every_n_iterations=ms.print_every_n_iterations,
    do_fix_flux_vector=ms.do_fix_flux_vector,
    do_fix_trans_vector=ms.do_fix_trans_vector,
    do_fix_tau_vector=ms.do_fix_tau_vector,
    do_fix_scale_vector=ms.do_fix_scale_vector,
    fixed_flux=fixed_flux,
    fixed_trans=fixed_trans,
    fixed_tau=fixed_tau,
    fixed_scale=fixed_scale,
    init_with_flux_vector=ms.init_with_flux_vector,
    init_with_tau_vector=ms.init_with_tau_vector,
    init_with_trans_vector=ms.init_with_trans_vector,
    init_with_scale_vector=ms.init_with_scale_vector,)

# Save the results
tu.save_transit_model_results_to_fits(
    fits_load_dir="",
    label=ms.fn_label,
    n_transit=ms.n_transit,
    model=model,
    flux=flux,
    trans=trans,
    tau=tau,
    scale=scale,
    mask=mask,)

plt.close("all")

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
# Plot the component spectra (one pdf per transit).
for trans_i in range(ms.n_transit):
    tm = transit_info["transit_num"] == trans_i

    tplt.plot_component_spectra(
        waves=waves,
        fluxes=flux,
        telluric_tau=tau[trans_i],
        planet_trans=trans,
        scale_vector=scale[tm],
        transit_num=trans_i,
        star_name=ms.fn_label,)

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