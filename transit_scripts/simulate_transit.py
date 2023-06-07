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

# For simulating a transit we currently just swap out the fluxes from an
# actually observed data cube created by prepare_transit_model.fits
fits_load_dir = "transit_data_wasp107.fits"

# Model stellar spectrum
marcs_fits = "synth/marcs_wasp107_4420g4.61z+0.00m0.683t0.plt"

# Model planet spectrum
planet_wave_fits = \
    "data/W107b_spec_H2O_main_iso_CO_all_iso_clear_wave_R200000.fits"
planet_spec_fits = \
    "data/W107b_spec_H2O_main_iso_CO_all_iso_clear_flux_R200000.fits"

# Molecfit modelled telluric spectra--one per night to be simulated
molecfit_fits = [
    "data_reduction/molecfit_wasp107_continuum_norm/MODEL/BEST_FIT_MODEL.fits",
    "data_reduction/wasp_107_n2/BEST_FIT_MODEL_opt.fits"]

# CRIRES+ throughput data from the ETC
throughput_json_path = "data/crires_etc_throughput_K2148.json"

# Wavelength limits and spectroscopic resolution
wl_min = 19000
wl_max = 25000
instr_resolving_power = 100000

# VLT UT M1 mirror and central obstruction radius
# https://www.eso.org/sci/facilities/paranal/telescopes/ut/m1unit.html
r_tel_prim = 8.2 / 2
r_tel_cen_ob = 1 / 2

do_equid_lambda_resample = True

# Fill value for missing values of CRIRES throughput
fill_throughput_value = 1

# Fill value for missing telluric tau values
tau_fill_value = 0

# Whether to apply the blaze/grating efficiency/throughput term
apply_blaze_function = False

# These three parameters determine how we simulate our scale vector which
# represents slit losses. To set this to unity, set scale_vector_method to 
# 'constant_unity'. Alternatively, set it to 'smoothed_random' where we
# generate n_phase random points between 0 and 2, and use a Savitzky–Golay to
# smooth this using a window size of savgol_window_frac_size * n_phase and a
# polynomial order of savgol_poly_order.
scale_vector_method = "smoothed_random"
savgol_window_frac_size = 0.5
savgol_poly_order = 3

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
# Load in pre-prepared fits file summarising real CRIRES observations. At the
# moment we simulate real transits using real airmass/phase/velocity info and
# simply duplicate the file and swap out the fluxes with simulated equivalents.
waves, _, _, det, orders, transit_info_list, syst_info = \
    tu.load_transit_info_from_fits("", star_name, n_transit=N_TRANSIT,)

# Initialise arrays for keeping track of simulated spectra
model_flux_list = []
snr_model_list = []
sigmas_model_list = []

# Keep track of components used to create said simulated spectra
flux_components = []
telluric_components = []
planet_components = []
scale_components = []

line = "-"*80

# Run separately for each transit
for transit_i in range(N_TRANSIT):
    print(line, "\nModelling transit #{}\n".format(transit_i), line, sep="")
    fluxes_model_all, snr_model_all, component_vectors = \
        sim.simulate_transit_multiple_epochs(
            wave_observed=waves*10,
            syst_info=syst_info,
            transit_info=transit_info_list[transit_i],
            marcs_fits=marcs_fits,
            planet_wave_fits=planet_wave_fits,
            planet_spec_fits=planet_spec_fits,
            molecfit_fits=molecfit_fits[transit_i],
            throughput_json_path=throughput_json_path,
            wl_min=wl_min,
            wl_max=wl_max,
            instr_resolving_power=instr_resolving_power,
            r_tel_prim=r_tel_prim,
            r_tel_cen_ob=r_tel_cen_ob,
            do_equid_lambda_resample=do_equid_lambda_resample,
            fill_throughput_value=fill_throughput_value,
            tau_fill_value=tau_fill_value,
            planet_transmission_boost_fac=planet_transmission_boost_fac,
            do_use_uniform_stellar_spec=do_use_uniform_stellar_spec,
            do_use_uniform_telluric_spec=do_use_uniform_telluric_spec,
            do_use_uniform_planet_spec=do_use_uniform_planet_spec,
            apply_blaze_function=apply_blaze_function,
            scale_vector_method=scale_vector_method,
            savgol_window_frac_size=savgol_window_frac_size,
            savgol_poly_order=savgol_poly_order,)
    
    model_flux_list.append(fluxes_model_all)
    snr_model_list.append(snr_model_all)

    # Keep track of the components that change from night to night
    # TODO: keep track of noise once we implement it
    flux_components.append(component_vectors["stellar_flux"])
    telluric_components.append(component_vectors["telluric_tau"])
    planet_components.append(component_vectors["planet_trans"])
    scale_components.append(component_vectors["scale_vector"])

    # Save our simulated fluxes in the same format as we would real data
    sigmas_model_list.append(np.zeros_like(fluxes_model_all))

# Convert to numpy arrays (not scale as n_phase isn't equal across transits)
flux_components = np.array(flux_components)
telluric_components = np.array(telluric_components)
planet_components = np.array(planet_components)

# Plot the simulated 'observed' spectra for all epochs
tplt.plot_all_input_spectra(
    waves=waves,
    fluxes_all_list=model_flux_list,
    transit_info_list=transit_info_list,
    n_transits=N_TRANSIT,)

# Plot the component spectra (one pdf per transit).
for trans_i in range(N_TRANSIT):
    tplt.plot_component_spectra(
        waves=waves,
        fluxes=flux_components[trans_i],
        telluric_tau=telluric_components[trans_i],
        planet_trans=planet_components[trans_i],
        scale_vector=scale_components[trans_i],
        transit_num=trans_i,)

# Save results to new fits file
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

# Add the component spectra.  Note that we assume that flux and planet trans
# is constant across transits, but that our tau  and scale vectors vary.
tu.save_simulated_transit_components_to_fits(
    fits_load_dir=save_path,
    label=label,
    n_transit=N_TRANSIT,
    flux=flux_components[0],
    tau=telluric_components,
    trans=planet_components[0],
    scale=scale_components,)