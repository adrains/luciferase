"""Script to cross-correlate a set of SYSREM residuals with a template 
exoplanet spectrum.
"""
import numpy as np
import transit.utils as tu
import transit.sysrem as sr
import transit.plotting as tplt
import astropy.units as u
import luciferase.utils as lu
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

# Grab shape
(n_phase, n_spec, n_px) = fluxes_list[ss.transit_i].shape

# Import SYSREM residuals
resid_all = tu.load_sysrem_residuals_from_fits(
    ss.save_path, ss.label, ss.n_transit, ss.transit_i,)

n_sysrem_iter = resid_all.shape[0]

planet_rvs = \
    transit_info_list[ss.transit_i]["delta"].values*const.c.cgs.to(u.km/u.s)

#------------------------------------------------------------------------------
# Import and prepare templates
#------------------------------------------------------------------------------
# Load in petitRADRTRANS datacube of templates. These templates will be in
# units of R_earth as a function of wavelength.
wave_p, spec_p_all, templ_info = tu.load_transmission_templates_from_fits(
    fits_file=ss.template_fits,
    min_wl_nm=16000,
    max_wl_nm=30000,)

# Clip edges to avoid edge effects introduced by interpolation
templ_spec_all = spec_p_all[:,10:-10]
wave_p = wave_p[10:-10] / 10

molecules = templ_info.columns.values

# Pick a template
templ_i = 1     # H2O model

# Convert to a transmission spectrum
r_e = const.R_earth.si.value
r_odot = const.R_sun.si.value

rp = syst_info.loc["r_planet_rearth", "value"] 
rs = syst_info.loc["r_star_rsun", "value"] * r_odot / r_e

trans_planet = 1 - ((templ_spec_all[templ_i] + rp)**2  / rs**2)

# Compute planet 'continuum' to normalise by
planet_cont = instrBroadGaussFast(
        wvl=wave_p,
        flux=trans_planet,
        resolution=300,
        equid=True,)

# [Optional] For testing, we can use the telluric vector for cross correlation
if ss.cc_with_telluric:
    print("Cross correlating with telluric template.")
    telluric_wave, _, _, telluric_trans = tu.load_telluric_spectrum(
        molecfit_fits=ss.molecfit_fits[0],
        tau_fill_value=ss.tau_fill_value,
        convert_to_angstrom=False,
        convert_to_nm=True,
        output_transmission=True,)

    wave_template = telluric_wave
    spectrum_template = telluric_trans

# [Optional] Or we can use a stellar spectrum
elif ss.cc_with_stellar:
    print("Cross correlating with stellar template.")
    wave_stellar, spec_stellar = lu.load_plumage_template_spectrum(
        template_fits=ss.stellar_template_fits,
        do_convert_air_to_vacuum_wl=False,)
    
    wave_template = wave_stellar
    spectrum_template = spec_stellar

# Otherwise run on a planet spectrum
else:
    print("Cross correlating with planet template.")
    wave_template = wave_p
    spectrum_template = trans_planet / planet_cont

#------------------------------------------------------------------------------
# Cross-correlation
#------------------------------------------------------------------------------
cc_rvs, ccv_per_spec, ccv_combined = sr.cross_correlate_sysrem_resid(
    waves=waves,
    sysrem_resid=resid_all,
    template_wave=wave_template,
    template_spec=spectrum_template,
    cc_rv_step=ss.cc_rv_step,
    cc_rv_lims=ss.cc_rv_lims,
    interpolation_method="cubic",)

# Plot cross-correlation
tplt.plot_sysrem_cc_2D(
    cc_rvs=cc_rvs,
    ccv_per_spec=ccv_per_spec,
    ccv_combined=ccv_combined,
    mean_spec_lambdas=np.mean(waves,axis=1),
    planet_rvs=planet_rvs,
    plot_label=ss.label,)

#------------------------------------------------------------------------------
# Kp-Vsys map
#------------------------------------------------------------------------------
# Create Kp-Vsys map
Kp_steps, Kp_vsys_map_per_spec, Kp_vsys_map_combined = sr.compute_Kp_vsys_map(
    cc_rvs=cc_rvs,
    ccv_per_spec=ccv_per_spec,
    ccv_combined=ccv_combined,
    transit_info=transit_info_list[ss.transit_i],
    syst_info=syst_info,
    Kp_lims=ss.Kp_lims,
    Kp_step=ss.Kp_step,)

# Plot Kp-Vsys map
tplt.plot_kp_vsys_map(
    cc_rvs=cc_rvs,
    Kp_steps=Kp_steps,
    Kp_vsys_map_per_spec=Kp_vsys_map_per_spec,
    Kp_vsys_map_combined=Kp_vsys_map_combined,
    mean_spec_lambdas=np.mean(waves,axis=1),
    plot_label=ss.label,)