"""Minimalistic implementation of SYSREM for debugging.
"""
import numpy as np
import transit.utils as tu
import transit.sysrem as sr
import matplotlib.pyplot as plt
from tqdm import tqdm

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------
simulation_settings_file = "scripts_transit/sysrem_settings.yml"
ss = tu.load_yaml_settings(simulation_settings_file)

# Load data from fits file--this can either be real or simulated data
waves, fluxes_list, sigmas_list, det, orders, transit_info_list, syst_info = \
    tu.load_transit_info_from_fits(ss.save_path, ss.label, ss.n_transit)

#------------------------------------------------------------------------------
# SYSREM
#------------------------------------------------------------------------------
# Run Nik's SYSREM
transit_i = 0
spectra = fluxes_list[transit_i]
(n_phase, n_spec, n_px) = spectra.shape

mm = np.isnan(spectra)
spectra[mm] = np.nanmean(spectra)

resid = sr.sysrem_piskunov(spectra.copy())

#------------------------------------------------------------------------------
# Cross-correlation
#------------------------------------------------------------------------------
# Import planet template
template = np.loadtxt("wasp107b_H2O_transmission.txt", delimiter=",")
temp_wave = template[:,0]
temp_spec = template[:,1]

cc_rvs, cc_values = sr.cross_correlate_sysrem_resid(
    waves=waves,
    sysrem_resid=resid[None,:,:,:],
    template_wave=temp_wave,
    template_spec=temp_spec,
    cc_rv_step=ss.cc_rv_step,
    cc_rv_lims=ss.cc_rv_lims,
    interpolation_method="cubic",)

# (n_phase, n_spec, n_cc)
ccv = cc_values[0].copy()
(_, _, n_cc) = ccv.shape

#------------------------------------------------------------------------------
# Diagnostics
#------------------------------------------------------------------------------
plt.close("all")
fig, axes = plt.subplots(1,18, sharey=True, figsize=(20, 2))
for spec_i in tqdm(range(n_spec), desc="Plotting", leave=False):
  # Sum along the CC direction
  norm_1D = np.nanmedian(ccv[:,spec_i,:], axis=1)
  norm_2D = np.broadcast_to(norm_1D[:,None], (n_phase, n_cc))

  ccv[:,spec_i,:] /= norm_2D
  axes[spec_i].imshow(ccv[:,spec_i,:], interpolation="none", aspect="auto")
