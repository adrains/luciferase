"""Perturbs the wavelength scale of a time-series set of CRIRES+ observations
or simulations to simulate uncertainies. Uncertainties can be added either by
treating each phase as coherent (n_phase rv shifts) or by treating the spectral
segments as separate (n_phase x n_spec rv_shifts). Uncertainties of +
/- sigma_wave km/s are added by interpolating the fluxes and sigmas onto a new
scale, but saving using the old one. The results are then saved to a duplicate
fits file with a modified label.
"""
import os
import shutil
import numpy as np
import transit.utils as tu
from tqdm import tqdm
import astropy.constants as const
from scipy.interpolate import interp1d

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------
# Original file to import
save_path = "simulations"
n_transit = 2
label = "simulated_R100000_WASP107b_variable_H2O_260203_stellar_1_telluric_1_planet_1_H2O_R_boost_1_slit_loss_1_vsys_offset_0_SNR_300"

# Whether to treat a single phase of n_spec as rigid in RV or not.
do_per_segment_pertubation = False

# Wavelength perturbation in km/s
sigma_kms = 0.1

#------------------------------------------------------------------------------
# Import
#------------------------------------------------------------------------------
# Load in
waves, fluxes_list, sigmas_list, det, orders, transit_info_list, syst_info = \
    tu.load_transit_info_from_fits(save_path, label, n_transit)

#------------------------------------------------------------------------------
# Doppler shift and interpolate fluxes and uncertainties
#------------------------------------------------------------------------------
fluxes_list_new = []
sigmas_list_new = []

for transit_i in range(n_transit):
    fluxes_new = np.zeros_like(fluxes_list[transit_i])
    sigmas_new = np.zeros_like(sigmas_list[transit_i])

    (n_phase, n_spec, n_px) = fluxes_new.shape

    # Draw n_phase x n_spec wavelength RV shifts
    if do_per_segment_pertubation:
        e_kms = np.random.normal(
            loc=0, scale=sigma_kms, size=(n_phase, n_spec))
        
    # Or just draw n_phase wavelength RV shifts and broadcast
    else:
         e_kms = np.random.normal(loc=0, scale=sigma_kms, size=(n_phase))
         e_kms = np.broadcast_to(e_kms[:,None], (n_phase, n_spec))

    desc = "Perturbing phases for night #{}".format(transit_i+1)

    # Loop over all segments
    for phase_i in tqdm(range(n_phase), leave=False, desc=desc):
        for spec_i in range(n_spec):
            wave = waves[spec_i].copy()
            spec = fluxes_list[transit_i][phase_i, spec_i].copy()
            sigma = sigmas_list[transit_i][phase_i, spec_i].copy()

            # Doppler shift wavelength scale
            wave_ds = wave * (1-e_kms[phase_i, spec_i]/(const.c.si.value/1000))
        
            # Interpolate fluxes and uncertainties here
            calc_flux = interp1d(
                x=wave,
                y=spec,
                kind="cubic",
                bounds_error=False,
                assume_sorted=True,)
            
            calc_sigma = interp1d(
                x=wave,
                y=sigma,
                kind="cubic",
                bounds_error=False,
                assume_sorted=True,)

            # Store these back to array
            fluxes_new[phase_i, spec_i] = calc_flux(wave_ds)
            sigmas_new[phase_i, spec_i] = calc_sigma(wave_ds)

    # Done with this night
    fluxes_list_new.append(fluxes_new)
    sigmas_list_new.append(sigmas_new)

#------------------------------------------------------------------------------
# Save to new file
#------------------------------------------------------------------------------
sigma_mps = sigma_kms * 1000
label_new = "{}_lambda_perturbed_{:0.0f}mps".format(label, sigma_mps)

fn_existing = os.path.join(
    save_path, "transit_data_{}_n{}.fits".format(label, n_transit))

fn_new = os.path.join(
    save_path, "transit_data_{}_n{}.fits".format(label_new, n_transit))

# Duplicate existing file
shutil.copyfile(fn_existing, fn_new)

# Save fluxes and uncertainties back, one pair per transit
for transit_i in range(n_transit):
    tu.save_fits_image_hdu(
        data=fluxes_list_new[transit_i],
        extension="spec",
        label=label_new,
        n_transit=n_transit,
        transit_i=transit_i,
        path=save_path)
    
    tu.save_fits_image_hdu(
        data=sigmas_list_new[transit_i],
        extension="sigma",
        label=label_new,
        n_transit=n_transit,
        transit_i=transit_i,
        path=save_path)