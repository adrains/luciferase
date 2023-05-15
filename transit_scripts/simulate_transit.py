"""Simulates multiple transits.

For simplicity, to begin with we'll just duplicate an already observed transit
which saves the trouble of having to predict observational factors like
airmass, barycentric correction, etc. To do this, we require that we've already
run prepare_transit_model.py and have a fits file. We'll simply load in this
fits file and swap out the fluxes. Eventually we'll move to simulating an
arbitrary transit.
"""
import numpy as np
import transit.simulator as sim
import transit.utils as tu
import transit.plotting as tplt

# -----------------------------------------------------------------------------
# Setup and Options
# -----------------------------------------------------------------------------
# Save path
save_path = ""
star_name = "wasp107"
label = "simulated_{}".format(star_name)

N_TRANSIT = 2

# Files
fits_load_dir = "transit_data_wasp107.fits"
marcs_fits = "synth/marcs_ltt1445/3340g5.0z-0.34m0.3t0.itf"
planet_wave_fits = "data/W107b_spec_H2O_main_iso_CO_all_iso_clear_wave_R200000.fits"
planet_spec_fits = "data/W107b_spec_H2O_main_iso_CO_all_iso_clear_flux_R200000.fits"
molecfit_fits = "data_reduction/molecfit_wasp107_continuum_norm/MODEL/BEST_FIT_MODEL.fits"
throughput_json_path = "data/crires_etc_throughput_K2148.json"

waves, fluxes_list, sigmas_list, det, orders, transit_info_list, syst_info = \
    tu.load_transit_info_from_fits("", star_name, n_transit=N_TRANSIT,)

# Params
wl_min = 19000
wl_max = 25000
instr_resolving_power = 100000

# VLT UT M1 mirror and central obstruction radius
# https://www.eso.org/sci/facilities/paranal/telescopes/ut/m1unit.html
r_tel_prim = 8.2 / 2
r_tel_cen_ob = 1 / 2

do_equidistant_lambda_sampling_before_broadening = True
fill_throughput_value = 1

tau_fill_value = 0

apply_blaze_function = False

# -----------------------------------------------------------------------------
# Test/troubleshooting settings
# -----------------------------------------------------------------------------
# Set these to true to disable the respective component in the modelling
do_use_uniform_stellar_spec = False
do_use_uniform_telluric_spec = False
do_use_uniform_planet_spec = False

# Set this to values > 1 to increase the planetary absorption
planet_transmission_boost_fac = 1000

# -----------------------------------------------------------------------------
# Running things
# -----------------------------------------------------------------------------
model_flux_list = []
snr_model_list = []
sigmas_model_list = []

line = "-"*80

# Run separately for each transit
for transit_i in range(N_TRANSIT):
    print(line, "\nModelling transit #{}\n".format(transit_i), line, sep="")
    fluxes_model_all, snr_model_all = sim.simulate_transit_multiple_epochs(
        wave_observed=waves*10,
        syst_info=syst_info,
        transit_info=transit_info_list[transit_i],
        marcs_fits=marcs_fits,
        planet_wave_fits=planet_wave_fits,
        planet_spec_fits=planet_spec_fits,
        molecfit_fits=molecfit_fits,
        throughput_json_path=throughput_json_path,
        wl_min=wl_min,
        wl_max=wl_max,
        instr_resolving_power=instr_resolving_power,
        r_tel_prim=r_tel_prim,
        r_tel_cen_ob=r_tel_cen_ob,
        do_equidistant_lambda_sampling_before_broadening=\
            do_equidistant_lambda_sampling_before_broadening,
        fill_throughput_value=fill_throughput_value,
        tau_fill_value=tau_fill_value,
        planet_transmission_boost_fac=planet_transmission_boost_fac,
        do_use_uniform_stellar_spec=do_use_uniform_stellar_spec,
        do_use_uniform_telluric_spec=do_use_uniform_telluric_spec,
        do_use_uniform_planet_spec=do_use_uniform_planet_spec,
        apply_blaze_function=apply_blaze_function,)
    
    model_flux_list.append(fluxes_model_all)
    snr_model_list.append(snr_model_all)

    # Save our simulated fluxes in the same format as we would real data
    sigmas_model_list.append(np.zeros_like(fluxes_model_all))

# Plot
tplt.plot_all_input_spectra(
    waves=waves,
    fluxes_all_list=model_flux_list,
    transit_info_list=transit_info_list,
    n_transits=N_TRANSIT,)

# Save results
tu.save_transit_info_to_fits(
    waves=waves,
    obs_spec_list=model_flux_list,
    sigmas_list=sigmas_model_list,
    n_transits=N_TRANSIT,
    detectors=det,
    orders=orders,
    transit_info_list=transit_info_list,
    syst_info=syst_info,
    fits_save_dir=save_path,
    label=label,)