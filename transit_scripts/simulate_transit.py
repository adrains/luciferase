"""Simulates a given planet transit.

For simplicity, to begin with we'll just duplicate an already observed transit
which saves the trouble of having to predict observational factors like
airmass, barycentric correction, etc. To do this, we require that we've already
run prepare_transit_model.py and have a fits file. We'll simply load in this
fits file and swap out the fluxes. Eventually we'll move to simulating an
arbitrary transit.
"""
import transit.simulator as sim
import transit.utils as tu

# -----------------------------------------------------------------------------
# Setup and Options
# -----------------------------------------------------------------------------
# Save path
save_path = ""
star_name = "wasp107"

# Location of the 
planet_properties_file = "transit_scripts/planet_data_wasp107.csv"

# Data directory
planet_root_dir = "/Users/arains/data/220310_WASP107/"

# Files
fits_load_dir = "transit_data_wasp107.fits"
marcs_fits = "synth/marcs_ltt1445/3340g5.0z-0.34m0.3t0.itf"
planet_wave_fits = "data/W107b_spec_H2O_main_iso_CO_all_iso_clear_wave_R200000.fits"
planet_spec_fits = "data/W107b_spec_H2O_main_iso_CO_all_iso_clear_flux_R200000.fits"
molecfit_fits = "data_reduction/molecfit_wasp107_continuum_norm/MODEL/BEST_FIT_MODEL.fits"
throughput_json_path = "data/crires_etc_throughput_K2148.json"

waves, obs_spec, sigmas, detectors, orders, transit_info, syst_info = \
    tu.load_transit_info_from_fits("", star_name,)

# Params
wl_min = 20000
wl_max = 25000
instr_resolving_power = 100000

# VLT UT M1 mirror and central obstruction radius
# https://www.eso.org/sci/facilities/paranal/telescopes/ut/m1unit.html
r_tel_prim = 8.2 / 2
r_tel_cen_ob = 1 / 2

do_equidistant_lambda_sampling_before_broadening = True
fill_throughput_value = 1

planet_transmission_boost_fac = 1

tau_fill_value = 0

# -----------------------------------------------------------------------------
# Running things
# -----------------------------------------------------------------------------
fluxes_model_all, snr_model_all = sim.simulate_transit_multiple_epochs(
    wave_observed=waves*10,
    syst_info=syst_info,
    transit_info=transit_info,
    marcs_fits=marcs_fits,
    planet_wave_fits=planet_wave_fits,
    planet_spec_fits=planet_spec_fits,
    molecfit_fits=molecfit_fits,
    wl_min=wl_min,
    wl_max=wl_max,
    instr_resolving_power=instr_resolving_power,
    r_tel_prim=r_tel_prim,
    r_tel_cen_ob=r_tel_cen_ob,
    do_equidistant_lambda_sampling_before_broadening=\
        do_equidistant_lambda_sampling_before_broadening,
    throughput_json_path=throughput_json_path,
    fill_throughput_value=fill_throughput_value,
    planet_transmission_boost_fac=planet_transmission_boost_fac,
    tau_fill_value=tau_fill_value,)