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
import luciferase.spectra as lsp
import luciferase.utils as lu
import datetime

# -----------------------------------------------------------------------------
# Setup and Options
# -----------------------------------------------------------------------------
# Import our simulation settings from a separate YAML file
simulation_settings_file = "scripts_transit/simulation_settings.yml"
ss = tu.load_yaml_settings(simulation_settings_file)

# -----------------------------------------------------------------------------
# Running the simualator
# -----------------------------------------------------------------------------
# Load in pre-prepared fits file summarising real CRIRES observations. At the
# moment we simulate real transits using real airmass/phase/velocity info and
# simply duplicate the file and swap out the fluxes with simulated equivalents.
waves, _, _, det, orders, transit_info_list, syst_info = \
    tu.load_transit_info_from_fits(
        ss.base_fits_path,
        ss.base_fits_label,
        n_transit=ss.n_transit,)

# Initialise arrays for keeping track of simulated spectra
model_flux_list = []
model_sigma_list = []

# Keep track of components used to create said simulated spectra
flux_components = []
telluric_components = []
planet_components = []
scale_components = []

line = "-"*80

# Print summary
print(line, "\nSimulation Settings\n", line, sep="")
if ss.base_fits_path == "":
    print("\tBase fits\t\ttransit_data_{}_n{}.fits".format(
        ss.base_fits_label, ss.n_transit))
else:
    print("\tBase fits\t\t{}/transit_data_{}_n{}.fits".format(
        ss.base_fits_path, ss.base_fits_label, ss.n_transit))
if ss.target_snr is None:
    print("\tSNR\t\t\tinf")
else:
    print("\tSNR\t\t\t{:0.0f}".format(ss.target_snr))
print("\tSpecies\t\t\t{}".format(", ".join(ss.species_to_model)))
print("\tBoost Fac\t\t{:0.0f}x".format(ss.planet_transmission_boost_fac))
print("\tBlaze Corr\t\t{}".format(ss.correct_for_blaze))
print("\tScale Vec\t\t{}".format(ss.scale_vector_method))
print("\tUniform stellar\t\t{}".format(ss.do_use_uniform_stellar_spec))
print("\tUniform telluric\t{}".format(ss.do_use_uniform_telluric_spec))
print("\tUniform planet\t\t{}\n".format(ss.do_use_uniform_planet_spec))

# Run separately for each transit
for transit_i in range(ss.n_transit):
    print(line, "\nModelling transit #{}\n".format(transit_i+1), line, sep="")
    model_flux, model_sigma, component_vectors = \
        sim.simulate_transit_multiple_epochs(
            wave_observed=waves*10,                 # Convert to Ångström
            syst_info=syst_info,
            transit_info=transit_info_list[transit_i],
            marcs_fits=ss.marcs_fits,
            planet_fits=ss.planet_fits,
            planet_species_to_model=ss.species_to_model,
            molecfit_fits=ss.molecfit_fits[transit_i],
            throughput_json_path=ss.throughput_json_path,
            target_snr=ss.target_snr,
            wl_min=ss.wl_min,
            wl_max=ss.wl_max,
            instr_resolving_power=ss.instr_resolving_power,
            r_tel_prim=ss.r_tel_prim,
            r_tel_cen_ob=ss.r_tel_cen_ob,
            do_equid_lambda_resample=ss.do_equid_lambda_resample,
            fill_throughput_value=ss.fill_throughput_value,
            tau_fill_value=ss.tau_fill_value,
            planet_transmission_boost_fac=ss.planet_transmission_boost_fac,
            do_use_uniform_stellar_spec=ss.do_use_uniform_stellar_spec,
            do_use_uniform_telluric_spec=ss.do_use_uniform_telluric_spec,
            do_use_uniform_planet_spec=ss.do_use_uniform_planet_spec,
            correct_for_blaze=ss.correct_for_blaze,
            scale_vector_method=ss.scale_vector_method,
            savgol_window_frac_size=ss.savgol_window_frac_size,
            savgol_poly_order=ss.savgol_poly_order,)
    
    model_flux_list.append(model_flux)
    model_sigma_list.append(model_sigma)

    # Keep track of the components that change from night to night
    # TODO: keep track of noise to compare with later
    flux_components.append(component_vectors["stellar_flux"])
    telluric_components.append(component_vectors["telluric_tau"])
    planet_components.append(component_vectors["planet_trans"])
    scale_components.append(component_vectors["scale_vector"])

# Convert to numpy arrays (not scale as n_phase isn't equal across transits)
flux_components = np.array(flux_components)
telluric_components = np.array(telluric_components)
planet_components = np.array(planet_components)

# -----------------------------------------------------------------------------
# Save results to fits
# -----------------------------------------------------------------------------
# Construct save file name
if type(ss.target_snr) == float or type(ss.target_snr) == int:
    target_snr = ss.target_snr

elif ss.target_snr is None or ss.target_snr.upper() == "NONE":
    target_snr = np.inf

else:
    raise Exception("Something went wrong")

# Construct file name with alphabetical list of species + other important info
species = ss.species_to_model
species.sort()
species_str = "_".join(species)

# Also grab the date
date = datetime.datetime.now()
date_str = date.strftime("%y%m%d")

model_slit_losses = 1 if ss.scale_vector_method == "smoothed_random" else 0

fn_label = "_".join([
    "{}".format(ss.label),
    date_str,
    "{}".format(species_str),
    "stellar_{:0.0f}".format(int(not ss.do_use_uniform_stellar_spec)),
    "telluric_{:0.0f}".format(int(not ss.do_use_uniform_telluric_spec)),
    "planet_{:0.0f}".format(int(not ss.do_use_uniform_planet_spec)),
    "boost_{:0.0f}".format(ss.planet_transmission_boost_fac),
    "slit_loss_{:0.0f}".format(model_slit_losses),
    "SNR_{:0.0f}".format(target_snr),])

# Construct fits table of simulation information
sim_info = sim.make_sim_info_df(ss)

# Save results to new fits file
tu.save_transit_info_to_fits(
    waves=waves,
    obs_spec_list=model_flux_list,
    sigmas_list=model_sigma_list,
    n_transits=ss.n_transit,
    detectors=det,
    orders=orders,
    transit_info_list=transit_info_list,
    syst_info=syst_info,
    fits_save_dir=ss.save_path,
    label=fn_label,
    sim_info=sim_info,)

# Add the component spectra.  Note that we assume that flux and planet trans
# is constant across transits, but that our tau  and scale vectors vary.
tu.save_simulated_transit_components_to_fits(
    fits_load_dir=ss.save_path,
    label=fn_label,
    n_transit=ss.n_transit,
    flux=flux_components[0],
    tau=telluric_components,
    trans=planet_components[0],
    scale=scale_components,)

# -----------------------------------------------------------------------------
# Diagnostic plots
# -----------------------------------------------------------------------------
# Plot the simulated 'observed' spectra for all epochs
tplt.plot_all_input_spectra(
    waves=waves,
    fluxes_all_list=model_flux_list,
    transit_info_list=transit_info_list,
    n_transits=ss.n_transit,)

# Plot the component spectra (one pdf per transit).
for trans_i in range(ss.n_transit):
    tplt.plot_component_spectra(
        waves=waves,
        fluxes=flux_components[trans_i],
        telluric_tau=telluric_components[trans_i],
        planet_trans=planet_components[trans_i],
        scale_vector=scale_components[trans_i],
        transit_num=trans_i,
        star_name=ss.star_name,)