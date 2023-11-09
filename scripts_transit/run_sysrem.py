"""Script to run SYSREM on a datacube (either real or simulated) of
exoplanet transmission spectra.
"""
import numpy as np
import transit.utils as tu
import transit.simulator as sim
import transit.sysrem as sr
import matplotlib.pyplot as plt
import transit.plotting as tplt

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------
simulation_settings_file = "scripts_transit/simulation_settings.yml"
ss = tu.load_yaml_settings(simulation_settings_file)

transit_i = 0

n_sysrem_iter = 5

label = "simulated_R100000_wasp107_trans_boost_x100_SNR_inf"
#label = "simulated_R100000_wasp107_trans_boost_x10_SNR_200"
#label = "transit_data_wasp107_n2.fits"
label = "wasp107"

#------------------------------------------------------------------------------
# Import data + clean
#------------------------------------------------------------------------------
# Load data from fits file--this can either be real or simulated data
waves, fluxes_list, sigmas_list, det, orders, transit_info_list, syst_info = \
    tu.load_transit_info_from_fits(ss.save_path, label, ss.n_transit)

# Grab dimensions of flux datacube
(n_phase, n_spec, n_px) = fluxes_list[transit_i].shape

# Sort
wave_ii = np.argsort(np.median(waves, axis=1))
waves = waves[wave_ii]
waves_1d = np.reshape(waves, n_spec*n_px)

for night_i in range(len(fluxes_list)):
    fluxes_list[night_i] = fluxes_list[night_i][:, wave_ii, :]
    sigmas_list[night_i] = sigmas_list[night_i][:, wave_ii, :]

# From here we want to normalise the data and construct a single bad px mask.
# This involves the following steps:
#   1) Continuum normalise each spectrum
#   2) Sigma clip in the spectral dimension, produce bad_px_mask_spec
#   3) Sigma clip in the phase dimension, produce bad_px_mask_phase
#   4) Produce bad_px_mask_telluric from regions with excessive telluric 
#      absorption
#   5) Create master bad_px_mask
#   6) Interpolate missing values for each px along phase dimension

# Continuum normalise
print("Continuum normalising spectra...")
fluxes_norm, sigmas_norm = tu.continuum_normalise_spectra(
    waves=waves,
    fluxes=fluxes_list[transit_i],
    sigmas=sigmas_list[transit_i],
    detectors=det,
    orders=orders,
    continuum_poly_coeff_path=\
        "data_reduction/WASP107/220310_WASP107/continuum_poly_coeff_2022-03-11_WASP-107.txt",)

# Construct bad px mask from tellurics
print("Constructing bad px mask from tellurics...")
telluric_wave, telluric_tau, _ = sim.load_telluric_spectrum(
    molecfit_fits=ss.molecfit_fits[transit_i],
    tau_fill_value=ss.tau_fill_value,)

# TODO: properly interpolate telluric vector

telluric_wave /= 10
telluric_trans = 10**-telluric_tau

bad_px_mask_1D = telluric_trans < 0.5
bad_px_mask_3D = np.tile(
    bad_px_mask_1D, n_phase).reshape(n_phase, n_spec, n_px)


# If running SYSREM on entire spectral range at once
#bad_px_mask_2D = np.tile(
#    bad_px_mask_1D, n_phase).reshape(n_phase, len(bad_px_mask_1D))

#fluxes = fluxes_norm.reshape(n_phase, len(bad_px_mask_1D))
#sigmas = sigmas_norm.reshape(n_phase, len(bad_px_mask_1D))

# If setting default uncertainties
#sigmas = np.full_like(fluxes, 0.01)
# sigmas_list[transit_i].reshape(n_phase, len(bad_px_mask_1D))

#------------------------------------------------------------------------------
# SYSREM
#------------------------------------------------------------------------------
print("Running SYSREM...")
resid_all = np.zeros((n_sysrem_iter+1, n_phase, n_spec, n_px))

# Run SYSREM on one spectral segment at a time
for spec_i in range(n_spec):
    resid = sr.run_sysrem(
        spectra=fluxes_norm[:,spec_i,:],
        e_spectra=sigmas_norm[:,spec_i,:],
        bad_px_mask=bad_px_mask_3D[:,spec_i,:],
        n_iter=n_sysrem_iter,
        tolerance=1E-6,
        max_converge_iter=100,
        diff_method="median",)
    
    resid_all[:,:,spec_i,:] = resid

#tplt.plot_sysrem_residuals(waves, resid)

#------------------------------------------------------------------------------
# Run Cross Correlation
#------------------------------------------------------------------------------
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

cc_rvs, cc_values = sr.cross_correlate_sysrem_resid(
    waves=waves,
    sysrem_resid=resid_all,
    template_wave=telluric_wave,#templ_wave,
    template_spec=telluric_trans,#templ_spec,
    cc_rv_step=0.5,
    cc_rv_lims=(-100,100),)

# Combine all segments into single mean CCF
cc_values_mean = np.nanmean(cc_values, axis=2)

tplt.plot_sysrem_cc_1D(cc_rvs, cc_values_mean,)
#tplt.plot_sysrem_cc_2D(cc_rvs, cc_values,)


Kp_steps, Kp_vsys_map = sr.compute_Kp_vsys_map(
    cc_rvs=cc_rvs,
    cc_values=cc_values_mean,
    transit_info=transit_info_list[transit_i],
    syst_info=syst_info,
    Kp_lims=(0,400),
    Kp_step=1.0,)

#------------------------------------------------------------------------------
# Diagnostics
#------------------------------------------------------------------------------
#plt.close("all")
#fig, axes = plt.subplots(6, sharex=True)
#for i in range(6):
#    im = axes[i].imshow(resid[i], aspect="auto", interpolation="none")
#    cb = fig.colorbar(im, ax=axes[i])
