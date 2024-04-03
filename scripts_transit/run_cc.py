"""Script to cross-correlate a set of SYSREM residuals with a template 
exoplanet spectrum.
"""
import numpy as np
import transit.utils as tu
import transit.simulator as sim
import transit.sysrem as sr
import transit.plotting as tplt
import astropy.units as u
from astropy import constants as const
from PyAstronomy.pyasl import instrBroadGaussFast

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------
simulation_settings_file = "scripts_transit/sysrem_settings.yml"
ss = tu.load_yaml_settings(simulation_settings_file)

#------------------------------------------------------------------------------
# Import spectra, obs/system data, + SYSREM residuals
#------------------------------------------------------------------------------
# Import data
waves, fluxes_list, sigmas_list, det, orders, transit_info_list, syst_info = \
    tu.load_transit_info_from_fits(ss.save_path, ss.label, ss.n_transit)

# Import normalised fluxes
waves, fluxes_norm, sigmas_norm, bad_px = tu.load_normalised_spectra_from_fits(
    fits_load_dir=ss.save_path,
    label=ss.label,
    n_transit=ss.n_transit,
    transit_i=ss.transit_i,)        # TODO: this will eventually be in a loop

# Grab shape
(n_phase, n_spec, n_px) = fluxes_list[ss.transit_i].shape

# Import SYSREM residuals
resid_all = tu.load_sysrem_residuals_from_fits(
    ss.save_path, ss.label, ss.n_transit, ss.transit_i,)

# Mask out entire segments
for spec_i in ss.segments_to_mask_completely:
    bad_px[:,spec_i,:] = np.full((n_phase, n_px), True)

# DEBUGGING: Drop segments
if ss.do_drop_segments:
    # Slice arrays
    waves = waves[ss.segments_to_keep, :]
    fluxes_norm = fluxes_norm[:, ss.segments_to_keep,]
    sigmas_norm = sigmas_norm[:, ss.segments_to_keep, :]
    bad_px_mask_3D = bad_px[:,ss.segments_to_keep,]

    # Update dimensions
    (n_phase, n_spec, n_px) = fluxes_norm.shape

#------------------------------------------------------------------------------
# Import telluric spectra
#------------------------------------------------------------------------------
telluric_wave, telluric_tau, _ = sim.load_telluric_spectrum(
    molecfit_fits=ss.molecfit_fits[ss.transit_i],
    tau_fill_value=ss.tau_fill_value,)

# TODO: properly interpolate telluric vector

telluric_wave /= 10
telluric_trans = 10**-telluric_tau

#------------------------------------------------------------------------------
# Import planet spectra
#------------------------------------------------------------------------------
# Load in petitRADRTRANS datacube of templates
templ_wave, templ_spec_all, templ_info = \
    tu.load_transmission_templates_from_fits(fits_file=ss.template_fits)

# Clip edges to avoid edge effects introduced by interpolation
templ_spec_all = templ_spec_all[:,10:-10]
templ_wave = templ_wave[10:-10] / 10

molecules = templ_info.columns.values

# Pick a template
templ_i = 1     # H2O model

# The datacube spectra are at R~200,000, so we need to further downsample
trans_planet_instr = instrBroadGaussFast(
        wvl=templ_wave,
        flux=templ_spec_all[templ_i],
        resolution=100000,
        equid=True,)

# [Optional] For testing, we can use the telluric vector for cross correlation
if ss.cc_with_telluric:
    wave_template = telluric_wave
    spectrum_template = telluric_trans

# Otherwise run on a planet spectrum
else:
    wave_template = templ_wave
    spectrum_template = trans_planet_instr

#------------------------------------------------------------------------------
# Run Cross Correlation
#------------------------------------------------------------------------------
cc_rvs, cc_values = sr.cross_correlate_sysrem_resid(
    waves=waves,
    sysrem_resid=resid_all,
    sigma_spec=sigmas_norm,
    template_wave=wave_template,
    template_spec=spectrum_template,
    cc_rv_step=ss.cc_rv_step,
    cc_rv_lims=ss.cc_rv_lims,)


# ------
# Correct the per-segment cc
# ------
# Subtract minimum value for each [sysrem_i, phase_i, spec_i]
# HACK: Currently the cross-correlation peak isn't hugely significant over the
# 'background', meaning that to see anything we need to subtract the median
# value.....which isn't correct, so this will be removed once we figure out
# what is going on.
cc_values_median = np.broadcast_to(
    np.nanmedian(cc_values, axis=3)[:,:,:,None], cc_values.shape)
cc_vals_subbed = cc_values - cc_values_median

# ------
# Correct the total cc
# ------
cc_values_total = np.nansum(cc_values, axis=2)[:,:,None,:]
cc_values_total_median = np.broadcast_to(
    np.nanmedian(cc_values_total, axis=3)[:,:,:,None], cc_values_total.shape)
cc_vals_total_subbed = cc_values_total - cc_values_total_median

# Combine all segments into single mean CCF
#cc_values_mean = np.nanmean(cc_values, axis=2)
#cc_values_sum = np.nansum(cc_values, axis=2)

# Plot cross RV vs correlation value in a separate panel in for each SYSREM
# iteration. We colour code each line by the phase number.
#tplt.plot_sysrem_cc_1D(cc_rvs, cc_values_mean,)

# Instead plot the cross correlation as a 2D map (RV vs phase) where the colour
# bar is the cross correlation value.
planet_rvs = \
    transit_info_list[ss.transit_i]["delta"].values*const.c.cgs.to(u.km/u.s)

# First plot the cross-correlation one order at a time
tplt.plot_sysrem_cc_2D(
    cc_rvs=cc_rvs,
    cc_values=cc_vals_subbed,
    mean_spec_lambdas=np.mean(waves,axis=1),
    planet_rvs=planet_rvs,
    plot_label=ss.label,)

# Now plot the combined cross-correlation
tplt.plot_sysrem_cc_2D(
    cc_rvs=cc_rvs,
    cc_values=cc_vals_total_subbed,
    mean_spec_lambdas=None,
    planet_rvs=planet_rvs,
    fig_size=(6,6),
    plot_label="comb_{}".format(ss.label),)

# Plot the Kp vs Vsys map
Kp_steps, Kp_vsys_map = sr.compute_Kp_vsys_map(
    cc_rvs=cc_rvs,
    cc_values=cc_vals_total_subbed[:,:,0,:],
    transit_info=transit_info_list[ss.transit_i],
    syst_info=syst_info,
    Kp_lims=ss.Kp_lims,
    Kp_step=ss.Kp_step,)

# Plot Kp-Vsys map
tplt.plot_kp_vsys_map()