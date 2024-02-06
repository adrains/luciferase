"""Script to run SYSREM on a datacube (either real or simulated) of
exoplanet transmission spectra.
"""
import numpy as np
import transit.utils as tu
import transit.simulator as sim
import luciferase.spectra as ls
import transit.sysrem as sr
import transit.plotting as tplt
import luciferase.utils as lu
import astropy.units as u
from astropy import constants as const
from PyAstronomy.pyasl import instrBroadGaussFast

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------
simulation_settings_file = "scripts_transit/sysrem_settings.yml"
ss = tu.load_yaml_settings(simulation_settings_file)

#------------------------------------------------------------------------------
# Import data + clean
#------------------------------------------------------------------------------
# Load data from fits file--this can either be real or simulated data
waves, fluxes_list, sigmas_list, det, orders, transit_info_list, syst_info = \
    tu.load_transit_info_from_fits(ss.save_path, ss.label, ss.n_transit)

# Grab dimensions of flux datacube
(n_phase, n_spec, n_px) = fluxes_list[ss.transit_i].shape

# Sort to be in proper wavelength order so nothing unexpected happens
wave_ii = np.argsort(np.median(waves, axis=1))
waves = waves[wave_ii]
waves_1d = np.reshape(waves, n_spec*n_px)

for night_i in range(len(fluxes_list)):
    fluxes_list[night_i] = fluxes_list[night_i][:, wave_ii, :]
    sigmas_list[night_i] = sigmas_list[night_i][:, wave_ii, :]

    # HACK: clean sigma=0 values
    is_zero = sigmas_list[night_i] == 0
    sigmas_list[night_i][is_zero] = 1E5

# Import telluric vector
telluric_wave, telluric_tau, _ = sim.load_telluric_spectrum(
    molecfit_fits=ss.molecfit_fits[ss.transit_i],
    tau_fill_value=ss.tau_fill_value,)

# TODO: properly interpolate telluric vector

telluric_wave /= 10
telluric_trans = 10**-telluric_tau

# Import stellar spectrum
wave_stellar, spec_stellar = lu.load_plumage_template_spectrum(
    template_fits=ss.stellar_template_fits,
    do_convert_air_to_vacuum_wl=False,)

# Continuum normalise
print("Continuum normalising spectra...")
fluxes_norm, sigmas_norm, poly_coeff = \
    ls.continuum_normalise_all_spectra_with_telluric_model(
        waves_sci=waves,
        fluxes_sci=fluxes_list[ss.transit_i],
        sigmas_sci=sigmas_list[ss.transit_i],
        wave_telluric=telluric_wave,
        trans_telluric=telluric_trans,
        wave_stellar=wave_stellar,
        spec_stellar=spec_stellar,
        bcors=transit_info_list[ss.transit_i]["bcor"].values,
        rv_star=syst_info.loc["rv_star", "value"],)

# Construct bad px mask from tellurics
print("Constructing bad px mask from tellurics...")
bad_px_mask_1D = telluric_trans < ss.telluric_trans_bad_px_threshold
bad_px_mask_3D = np.tile(
    bad_px_mask_1D, n_phase).reshape(n_phase, n_spec, n_px)

# Mask out entire segments
for spec_i in ss.segments_to_mask_completely:
    bad_px_mask_3D[:,spec_i,:] = np.full((n_phase, n_px), True)

# DEBUGGING: Drop segments
if ss.do_drop_segments:
    # Slice arrays
    waves = waves[ss.segments_to_keep, :]
    fluxes_norm = fluxes_norm[:, ss.segments_to_keep,]
    sigmas_norm = sigmas_norm[:, ss.segments_to_keep, :]
    bad_px_mask_3D = bad_px_mask_3D[:,ss.segments_to_keep,]

    # Update dimensions
    (n_phase, n_spec, n_px) = fluxes_norm.shape

#------------------------------------------------------------------------------
# Import planet spectra
#------------------------------------------------------------------------------
# Load in Fabio's grid of planet models
# TODO: add an option to convert to nm inside the function
yaml_settings_file = "scripts_transit/transit_model_settings.yml"
ms = tu.load_yaml_settings(yaml_settings_file)

templ_wave, templ_spec_all, templ_info = \
    tu.load_transmission_templates_from_fits(fits_file=ms.template_fits)

# Convert to nm
templ_wave /= 10

# Clip edges
templ_spec_all = templ_spec_all[:,10:-10]
templ_wave = templ_wave[10:-10]

molecules = templ_info.columns.values

# Pick a template
templ_i = 6     # H2O model
templ_spec = templ_spec_all[templ_i]

# -------------
# Load in Ansgar's planet spectrum
wave_planet, trans_planet = sim.load_planet_spectrum(
    wave_file=ss.planet_wave_fits,
    spec_file=ss.planet_spec_fits,)

# Convert to nm
wave_planet /= 10

trans_planet_instr = instrBroadGaussFast(
        wvl=wave_planet,
        flux=trans_planet,
        resolution=100000,
        equid=True,)

# ------------------
# For testing, we can use the telluric vector to cross correlate against
if ss.cc_with_telluric:
    wave_template = telluric_wave
    spectrum_template = telluric_trans

# Otherwise run on a planet spectrum
else:
    wave_template = wave_planet
    spectrum_template = trans_planet_instr

#------------------------------------------------------------------------------
# SYSREM
#------------------------------------------------------------------------------
print("Running SYSREM...")
resid_all = np.full((ss.n_sysrem_iter+1, n_phase, n_spec, n_px), np.nan)

# Run SYSREM on one spectral segment at a time
for spec_i in range(n_spec):
    fmt_txt = "\nSpectral segment {:0.0f}, λ~{:0.0f} nm\n".format(
            spec_i, np.mean(waves[spec_i]))
    print("-"*80, fmt_txt, "-"*80,sep="")
    
    # Skip entirely nan segments
    if spec_i in ss.segments_to_mask_completely and not ss.do_drop_segments:
        print("\tSkipping...\n")
        continue

    resid = sr.run_sysrem(
        spectra=fluxes_norm[:,spec_i,:],
        e_spectra=sigmas_norm[:,spec_i,:],
        bad_px_mask=bad_px_mask_3D[:,spec_i,:],
        n_iter=ss.n_sysrem_iter,
        mjds=transit_info_list[ss.transit_i]["mjd_mid"].values,
        tolerance=ss.sysrem_convergence_tol,
        max_converge_iter=ss.sysrem_max_convergence_iter,
        diff_method=ss.sysrem_diff_method,
        sigma_threshold_phase=ss.sigma_threshold_phase,
        sigma_threshold_spectral=ss.sigma_threshold_spectral,)
    
    resid_all[:,:,spec_i,:] = resid

tplt.plot_sysrem_residuals(waves, resid_all)

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

# Subtract minimum value for each [sysrem_i, phase_i, spec_i]
# HACK: Currently the cross-correlation peak isn't hugely significant over the
# 'background', meaning that to see anything we need to subtract the median
# value.....which isn't correct, so this will be removed once we figure out
# what is going on.
cc_values_median = np.broadcast_to(
    np.nanmedian(cc_values, axis=3)[:,:,:,None], cc_values.shape)
cc_vals_subbed = cc_values - cc_values_median

# Combine all segments into single mean CCF
cc_values_mean = np.nanmean(cc_values, axis=2)
cc_values_sum = np.nansum(cc_values, axis=2)

# Plot cross RV vs correlation value in a separate panel in for each SYSREM
# iteration. We colour code each line by the phase number.
tplt.plot_sysrem_cc_1D(cc_rvs, cc_values_mean,)

# Instead plot the cross correlation as a 2D map (RV vs phase) where the colour
# bar is the cross correlation value.
planet_rvs = \
    transit_info_list[ss.transit_i]["delta"].values*const.c.cgs.to(u.km/u.s)
tplt.plot_sysrem_cc_2D(
    cc_rvs,
    cc_vals_subbed,
    np.mean(waves,axis=1),
    planet_rvs=planet_rvs,)

# Plot the Kp vs Vsys map
Kp_steps, Kp_vsys_map = sr.compute_Kp_vsys_map(
    cc_rvs=cc_rvs,
    cc_values=cc_values_mean,
    transit_info=transit_info_list[ss.transit_i],
    syst_info=syst_info,
    Kp_lims=ss.Kp_lims,
    Kp_step=ss.Kp_step,)

# Plot Kp-Vsys map
pass