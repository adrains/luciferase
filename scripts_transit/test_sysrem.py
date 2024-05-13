"""Minimalistic implementation of SYSREM for debugging.
"""
import numpy as np
import transit.utils as tu
import transit.sysrem as sr
from tqdm import tqdm
import astropy.units as u
import transit.plotting as tplt
from astropy import constants as const

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------
simulation_settings_file = "scripts_transit/sysrem_settings.yml"
ss = tu.load_yaml_settings(simulation_settings_file)

# Load data from fits file--this can either be real or simulated data
waves, fluxes_list, sigmas_list, det, orders, transit_info_list, syst_info = \
    tu.load_transit_info_from_fits(ss.save_path, ss.label, ss.n_transit)

planet_rvs = \
    transit_info_list[ss.transit_i]["delta"].values*const.c.cgs.to(u.km/u.s)

#------------------------------------------------------------------------------
# SYSREM
#------------------------------------------------------------------------------
# Run Nik's SYSREM
transit_i = 0
spectra = fluxes_list[transit_i]
(n_phase, n_spec, n_px) = spectra.shape

mm = np.isnan(spectra)
spectra[mm] = np.nanmean(spectra)

resid_all = sr.sysrem_piskunov(
    spectra=spectra.copy(),
    n_iter=ss.n_sysrem_iter,
    sigma_threshold=3.0,)

#------------------------------------------------------------------------------
# Cross-correlation
#------------------------------------------------------------------------------
# Import planet template
template = np.loadtxt("wasp107b_H2O_transmission.txt", delimiter=",")
temp_wave = template[:,0]
temp_spec = template[:,1]

cc_rvs, ccv_per_spec, ccv_global = sr.cross_correlate_sysrem_resid(
    waves=waves,
    sysrem_resid=resid_all,
    template_wave=temp_wave,
    template_spec=temp_spec,
    cc_rv_step=ss.cc_rv_step,
    cc_rv_lims=ss.cc_rv_lims,
    interpolation_method="cubic",)

# (n_phase, n_spec, n_cc)
ccv_norm = ccv_per_spec.copy()
ccvg = ccv_global.copy()
(_, _, _, n_cc) = ccv_norm.shape

# Normalise
# TODO: move inside function
for iter_i in range(ss.n_sysrem_iter+1):
    for spec_i in tqdm(range(n_spec), desc="Plotting", leave=False):
        # Sum along the CC direction
        norm_1D = np.nansum(ccv_norm[iter_i, :,spec_i,:], axis=1)
        norm_2D = np.broadcast_to(norm_1D[:,None], (n_phase, n_cc))

        ccv_norm[iter_i, :,spec_i,:] /= norm_2D

# Plot cross-correlation
tplt.plot_sysrem_cc_2D(
    cc_rvs=cc_rvs,
    cc_values=ccv_norm,
    mean_spec_lambdas=np.mean(waves,axis=1),
    planet_rvs=planet_rvs,
    plot_label=ss.label,)

#------------------------------------------------------------------------------
# Kp-Vsys plot
#------------------------------------------------------------------------------
Kp_steps, Kp_vsys_map = sr.compute_Kp_vsys_map(
    cc_rvs=cc_rvs,
    cc_values=ccv_norm,
    transit_info=transit_info_list[ss.transit_i],
    syst_info=syst_info,
    Kp_lims=ss.Kp_lims,
    Kp_step=ss.Kp_step,)

tplt.plot_kp_vsys_map(
    cc_rvs=cc_rvs,
    Kp_steps=Kp_steps,
    Kp_vsys_map=Kp_vsys_map,
    mean_spec_lambdas=np.mean(waves,axis=1),
    plot_label=ss.label,)