"""
Plot to visualise transit components. Plots the following spectra (offset):
 - combined spectra on each night at central phase
 - telluric component for night 1
 - stellar component for night 1 at central phase
 - exoplanet blocking spectrum for night 1 at ingress and egress
"""
import numpy as np
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
import transit.utils as tu
import transit.simulator as sim
import matplotlib.ticker as plticker

# -----------------------------------------------------------------------------
# Setup and Options
# -----------------------------------------------------------------------------
# Import our simulation settings from a separate YAML file
simulation_settings_file = "scripts_transit/simulation_settings.yml"
ss = tu.load_yaml_settings(simulation_settings_file)

# Simulation to visualise.
label = "simulated_R100000_wasp107_240805_C2H2_CH4_CO_CO2_H2O_H2S_HCN_NH3_PH3_SiO_stellar_1_telluric_1_planet_1_boost_1_slit_loss_0_SNR_inf"

# Select the spectral segment we want to plot. Set to None to plot all spectra.
spec_i_selected = 13

# Import simulation
waves, fluxes_list, sigmas_list, det, orders, transit_info_list, syst_info = \
    tu.load_transit_info_from_fits(
        "simulations",
        label,
        n_transit=ss.n_transit,)

# Grab central phases for each night
phase_mid_is = [fluxes_list[ni].shape[0]//2 for ni in range(ss.n_transit)]

# Grab observed fluxes and gammas for central phases
obs_fluxes = []
obs_gammas = []

for night_i, phase_mid_i in zip(range(ss.n_transit), phase_mid_is):
    obs_fluxes.append(fluxes_list[night_i][phase_mid_i])
    obs_gammas.append(transit_info_list[night_i]["gamma"].values[phase_mid_i])

obs_fluxes = np.stack(obs_fluxes)

# Grab phases at ingress and egress for planet on night 1
transit_info = transit_info_list[0]
in_transit = transit_info["is_in_transit_mid"].values
transit_min_i = transit_info[in_transit].index.values[0]
transit_max_i = transit_info[in_transit].index.values[-1]

phases = [transit_min_i, transit_max_i]

# -----------------------------------------------------------------------------
# Telluric template
# -----------------------------------------------------------------------------
# Load in telluric templates for each night
telluric_trans_2D_all = []

for night_i in range(ss.n_transit):
    telluric_wave, _, _, telluric_trans = tu.load_telluric_spectrum(
        molecfit_fits=ss.molecfit_fits[night_i],
        tau_fill_value=ss.tau_fill_value,
        convert_to_angstrom=False,
        convert_to_nm=True,
        output_transmission=True,)

    (n_spec, n_px) = waves.shape
    telluric_trans_2D_all.append(telluric_trans.reshape(n_spec, n_px))

telluric_trans_2D_all = np.stack(telluric_trans_2D_all)

# -----------------------------------------------------------------------------
# Stellar template
# -----------------------------------------------------------------------------
# Import stellar spectral component
wave_marcs, fluxes_marcs, mus_marcs = sim.read_marcs_spectrum(
    filepath=ss.marcs_fits,
    wave_min=ss.wl_min,
    wave_max=ss.wl_max,
    a_limb=None,
    n_mu_samples=None,)

wave_marcs /= 10

# Disc integrate stellar spectrum
delta_lambda = np.median(np.diff(wave_marcs))
med_lambda = np.median(wave_marcs)
delta_v = const.c.to(u.km / u.second).value * delta_lambda / med_lambda

fluxes_disk = sim.disk_integration(
    mus=mus_marcs[0],
    inten=fluxes_marcs,
    deltav=delta_v,
    vsini_in=0,
    vrt_in=0,
    osamp=1)

# Interpolate and RV shift
stellar_fluxes = []

for night_i in range(ss.n_transit):
    stellar_flux_obs = sim.shift_spec_and_convert_to_instrumental_scale(
        wave_old=wave_marcs,
        flux_old=fluxes_disk,
        wave_obs=waves,
        doppler_shift=obs_gammas[night_i],
        instr_resolving_power=ss.instr_resolving_power,
        do_equid_lambda_resample=ss.do_equid_lambda_resample,)
    
    stellar_fluxes.append(stellar_flux_obs)

stellar_fluxes = np.stack(stellar_fluxes)

# -----------------------------------------------------------------------------
# Planet template
# -----------------------------------------------------------------------------
# Import planet wavelengths, fluxes, and template (i.e. species) info
wave_planet, Rp_Re_vs_lambda_planet_all, templ_info = \
    tu.load_transmission_templates_from_fits(
        fits_file=ss.planet_fits,
        convert_rp_rj_to_re=True,)

wave_planet /= 10

# Now select the appropriate template from trans_planet_all simulating the
# appropriate set of molecules. Raise an exception if we don't have a
# template with that combination of molecules.
molecule_cols = templ_info.columns.values
has_molecule = np.full_like(molecule_cols, False)

for mol_i, molecule in enumerate(molecule_cols):
    if molecule in ss.species_to_model:
        has_molecule[mol_i] = True

match_i = np.argwhere(np.all(has_molecule == templ_info.values, axis=1))

if len(match_i) == 0:
    raise ValueError("Invalid molecule combination!")
else:
    Rp_Re_vs_lambda_planet = Rp_Re_vs_lambda_planet_all[int(match_i)]

# Interpolate and RV shift
planet_spectra = []

for phase_i in phases:
    planet_spec = sim.shift_spec_and_convert_to_instrumental_scale(
        wave_old=wave_planet,
        flux_old=Rp_Re_vs_lambda_planet,
        wave_obs=waves,
        doppler_shift=transit_info["delta"][phase_i],
        instr_resolving_power=ss.instr_resolving_power,
        do_equid_lambda_resample=ss.do_equid_lambda_resample,)
    
    planet_spectra.append(planet_spec)

planet_spectra = np.stack(planet_spectra)

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
plt.close("all")

fig, axis = plt.subplots(nrows=1, figsize=(20,10), sharex=True)

night_colours = ["forestgreen", "darkorange"]
planet_colours = ["b", "r"]
planet_labels = ["WASP-107 b (Ingress)", "WASP-107 b (Egress)"]

# Plot each spectrum suitably offset and scaled
for phase_i in range(2):
    for spec_i in range(n_spec):
        # only plot spectral segment of interest
        if spec_i_selected is not None and spec_i_selected != spec_i:
            continue
        
        # Grab shorter handles for convenience
        p_spec = planet_spectra[phase_i, spec_i]
        s_spec = stellar_fluxes[phase_i, spec_i]
        t_spec = telluric_trans_2D_all[phase_i, spec_i]
        o_spec = obs_fluxes[phase_i, spec_i]

        # planet
        axis.plot(
            waves[spec_i],
            10 * p_spec / np.nanmedian(p_spec) - 10,
            linewidth=1.0,
            color=planet_colours[phase_i],
            alpha=0.9,
            label=planet_labels[phase_i])
        
        # Just plot first night for stellar and telluric component
        if phase_i == 0:
            # stellar
            axis.plot(
                waves[spec_i],
                s_spec / np.nanmedian(s_spec),
                linewidth=1.0,
                color="cadetblue",
                label="Stellar")
            
            # telluric
            axis.plot(
                waves[spec_i],
                1 + t_spec,
                linewidth=1.0,
                color="darkmagenta",
                label="Telluric",)
            
        # observed
        axis.plot(
            waves[spec_i],
            2 + o_spec / np.nanmedian(o_spec),
            linewidth=1.0,
            color=night_colours[phase_i],
            alpha=0.9,
            label="Observed (Night {})".format(phase_i+1))

x = waves[spec_i_selected][10]
axis.text(x=x, y=0.5, s="Planet", color="r", fontsize="large")
axis.text(x=x, y=1.1, s="Stellar", color="cadetblue", fontsize="large")
axis.text(x=x, y=2.0, s="Telluric", color="darkmagenta", fontsize="large")
axis.text(x=x, y=3.15, s="Observed", color="forestgreen", fontsize="large")

axis.tick_params(axis='both', which='major', labelsize="large")
axis.xaxis.set_major_locator(plticker.MultipleLocator(base=1))
axis.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.5))
axis.set_yticks([])
axis.set_xlabel("Wavelength (nm)", fontsize="x-large")

# Plot legend, but re-order and make lines thicker
handles, labels = plt.gca().get_legend_handles_labels()
order = [3,5,2,1,0,4]

leg = plt.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    ncol=6,
    loc="upper center",
    fontsize="x-large",)

for legobj in leg.legendHandles:
    legobj.set_linewidth(1.5)

axis.set_ylim(-0.25,3.5)
axis.set_xlim(waves[spec_i_selected][0], waves[spec_i_selected][-1])

plt.tight_layout()
plt.savefig("plots/sim_component_overview_pptx.pdf")