"""Script to cross-correlate a set of SYSREM residuals with a template 
exoplanet spectrum.
"""
import os
import numpy as np
import transit.utils as tu
import transit.sysrem as sr
import transit.plotting as tplt
import astropy.units as u
import luciferase.utils as lu
from astropy import constants as const

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------
simulation_settings_file = "scripts_transit/sysrem_settings.yml"
ss = tu.load_yaml_settings(simulation_settings_file)

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------
# Import data
waves, fluxes_list, sigmas_list, det, orders, transit_info_list, syst_info = \
    tu.load_transit_info_from_fits(ss.save_path, ss.label, ss.n_transit)

# Grab the number of nights
n_transit = len(fluxes_list)

# Import template for CC
wave_template, spectrum_template = tu.prepare_cc_template(
    cc_settings=ss,
    syst_info=syst_info,
    templ_wl_nm_bounds=(16000,30000),
    continuum_resolving_power=300,)

#------------------------------------------------------------------------------
# Do *nightly* cross correlation and Kp-Vsys maps
#------------------------------------------------------------------------------
ccv_per_spec_all = []
ccv_combined_all = []
Kp_vsys_map_per_spec_all = []
Kp_vsys_map_combined_all = []

rv_star = syst_info.loc["rv_star", "value"]

for transit_i in range(n_transit):
    #--------------------------------------------------------------------------
    # Imports and setup
    #--------------------------------------------------------------------------
    print("\nRunning on night {}/{}".format(transit_i+1, n_transit))

    # Import SYSREM residuals
    resid_all = tu.load_sysrem_residuals_from_fits(
        ss.save_path, ss.label, ss.n_transit, transit_i,)

    n_sysrem_iter = resid_all.shape[0]

    # Grab planet RVs for overplotting on CC plot
    planet_rvs = \
        transit_info_list[transit_i]["delta"].values*const.c.cgs.to(u.km/u.s)

    #--------------------------------------------------------------------------
    # Cross-correlation
    #--------------------------------------------------------------------------
    # Grab the barycentric velocities. We'll take these into account when doing
    # the cross correlation such that we're cross correlating around the *star*
    # rather than around the telescope frame of reference.
    rv_bcors =  -1 * transit_info_list[transit_i]["bcor"].values + rv_star

    cc_rvs, ccv_per_spec, ccv_combined = sr.cross_correlate_sysrem_resid(
        waves=waves,
        sysrem_resid=resid_all,
        template_wave=wave_template,
        template_spec=spectrum_template,
        bcors=rv_bcors,
        cc_rv_step=ss.cc_rv_step,
        cc_rv_lims=ss.cc_rv_lims,
        interpolation_method="cubic",)

    # Store
    ccv_per_spec_all.append(ccv_per_spec)
    ccv_combined_all.append(ccv_combined)

    # Plot cross-correlation
    tplt.plot_sysrem_cc_2D(
        cc_rvs=cc_rvs,
        ccv_per_spec=ccv_per_spec,
        ccv_combined=ccv_combined,
        mean_spec_lambdas=np.mean(waves,axis=1),
        planet_rvs=planet_rvs,
        plot_label="{}_n{}".format(ss.label, transit_i+1),)

    #--------------------------------------------------------------------------
    # Kp-Vsys map
    #--------------------------------------------------------------------------
    # Create Kp-Vsys map *per spectral segment*
    Kp_steps, Kp_vsys_map_per_spec, Kp_vsys_map_combined = \
        sr.compute_Kp_vsys_map(
            cc_rvs=cc_rvs,
            ccv_per_spec=ccv_per_spec,
            ccv_combined=ccv_combined,
            transit_info=transit_info_list[transit_i],
            Kp_lims=ss.Kp_lims,
            Kp_step=ss.Kp_step,)

    # Store
    Kp_vsys_map_per_spec_all.append(Kp_vsys_map_per_spec)
    Kp_vsys_map_combined_all.append(Kp_vsys_map_combined)

    # Plot overview Kp-Vsys map
    tplt.plot_kp_vsys_map(
        cc_rvs=cc_rvs,
        Kp_steps=Kp_steps,
        Kp_vsys_map_per_spec=Kp_vsys_map_per_spec,
        Kp_vsys_map_combined=Kp_vsys_map_combined,
        mean_spec_lambdas=np.mean(waves,axis=1),
        plot_label="{}_n{}".format(ss.label, transit_i+1),)
    
    # Combined Kp-Vsys map for this night after merging all spectral segments
    tplt.plot_combined_kp_vsys_map_as_snr(
        cc_rvs=cc_rvs,
        Kp_steps=Kp_steps,
        Kp_vsys_maps=Kp_vsys_map_combined,
        plot_title="Night #{}".format(transit_i+1),
        plot_label="{}_n{}".format(ss.label, transit_i+1),)
    
#------------------------------------------------------------------------------
# Now *combine* each nightly Kp-Vsys map
#------------------------------------------------------------------------------
# Combine Kp-Vsys maps
# TODO: Currently we just mask negative values and add in quadrature.
map_per_spec_all_nights = sr.combine_kp_vsys_map(Kp_vsys_map_per_spec_all)
map_combined_all_nights = sr.combine_kp_vsys_map(Kp_vsys_map_combined_all)

# Plot overview Kp-Vsys map with all spectral segments
tplt.plot_kp_vsys_map(
    cc_rvs=cc_rvs,
    Kp_steps=Kp_steps,
    Kp_vsys_map_per_spec=map_per_spec_all_nights,
    Kp_vsys_map_combined=map_combined_all_nights,
    mean_spec_lambdas=np.mean(waves,axis=1),
    plot_label="{}_all_nights".format(ss.label),)

# Combined Kp-Vsys Map after merging all spectral segments
tplt.plot_combined_kp_vsys_map_as_snr(
    cc_rvs=cc_rvs,
    Kp_steps=Kp_steps,
    Kp_vsys_maps=map_combined_all_nights,
    plot_title="Combined Nights",
    plot_label="{}_all_nights".format(ss.label),)

#------------------------------------------------------------------------------
# Dump arrays to disk
#------------------------------------------------------------------------------
# Dump the results of CC and the resulting Kp-Vsys map to a pickle, with the
# label, number of transits, template info, and the SYSREM settings obj as a
# whole in the filename.\
species = "_".join(ss.species_to_cc)
fn_label = "_".join(
    ["cc_results", ss.label, str(ss.n_transit), str("template"), species])
fn = os.path.join(ss.save_path, "{}.pkl".format(fn_label))

tu.dump_cc_results(
    filename=fn,
    cc_rvs=cc_rvs,
    ccv_ps=ccv_per_spec_all,
    ccv_comb=ccv_combined_all,
    Kp_steps=Kp_steps,
    Kp_vsys_map_ps=Kp_vsys_map_per_spec_all,
    Kp_vsys_map_comb=Kp_vsys_map_combined_all,
    Kp_vsys_map_ps_all_nights=map_per_spec_all_nights,
    Kp_vsys_map_comb_all_nights=map_combined_all_nights,
    sysrem_settings=ss,)