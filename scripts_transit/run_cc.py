"""Script to cross-correlate a set of SYSREM residuals with a template 
exoplanet spectrum.
"""
import os
import numpy as np
import transit.utils as tu
import transit.sysrem as sr
import transit.plotting as tplt
import astropy.units as u
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
# Print summary
#------------------------------------------------------------------------------
print("-"*80, "\nCross Correlation Settings\n", "-"*80, sep="")
if ss.save_path == "":
    print("\tSpectra fits\t\ttransit_data_{}_n{}.fits".format(
        ss.label, ss.n_transit))
else:
    print("\tSpectra fits\t\t{}/transit_data_{}_n{}.fits".format(
        ss.save_path, ss.label, ss.n_transit))

if ss.cc_with_telluric:
    print("\tTelluric template\t{}".format(ss.molecfit_fits[0]))
elif ss.cc_with_stellar:
    print("\tStellar template\t{}".format(ss.stellar_template_fits))
else:
    print("\tPlanet template\t\t{}".format(ss.planet_fits))
    print("\tSpecies\t\t\t{}".format(", ".join(ss.species_to_cc)))
print("\tSplit A/B sequences\t{}".format(ss.split_AB_sequences))
print("\tCC RV step\t\t{:0.2f} km/s".format(ss.cc_rv_step))
print("\tCC RV limits\t\t{:0.0f} - {:0.0f} km/s".format(*ss.cc_rv_lims))
print("\tKp RV step\t\t{:0.2f} km/s".format(ss.Kp_step))
print("\tKp RV limits\t\t{:0.0f} - {:0.0f} km/s".format(*ss.Kp_lims))
print("\tVsys RV limits\t\t{:0.0f} - {:0.0f} km/s".format(*ss.kp_vsys_x_range))
print("-"*80)

#------------------------------------------------------------------------------
# Do *nightly* cross correlation and Kp-Vsys maps
#------------------------------------------------------------------------------
ccv_per_spec_all = []
ccv_combined_all = []
Kp_vsys_map_per_spec_all = []
Kp_vsys_map_combined_all = []

rv_star = syst_info.loc["rv_star", "value"]

species = "_".join(ss.species_to_cc)

for transit_i in range(n_transit):
    #--------------------------------------------------------------------------
    # [Optional] Split A/B sequences
    #--------------------------------------------------------------------------
    # Grab shape for convenience
    n_phase, n_spec, n_px = fluxes_list[transit_i].shape

    # Split A/B frames into separate sequences
    if ss.split_AB_sequences:
        nod_pos = transit_info_list[transit_i]["nod_pos"].values
        sequence_masks = [nod_pos == "A", nod_pos == "B"]
        sequences = ["A", "B"]

    # OR run with interleaved A/B sequences
    else:
        sequence_masks = [np.full(n_phase, True)]
        sequences = ["AB"]
    
    #--------------------------------------------------------------------------
    # Loop over each sequence for this night
    #--------------------------------------------------------------------------
    for seq, seq_mask in zip(sequences, sequence_masks):
        print("\nRunning on night {}/{}, seq {}".format(
            transit_i+1, n_transit, seq))

        # Import SYSREM residuals
        resid_all = tu.load_sysrem_residuals_from_fits(
            ss.save_path, ss.label, ss.n_transit, transit_i, seq)

        n_sysrem_iter = resid_all.shape[0]

        # Grab planet RVs for overplotting on CC plot
        planet_rvs = transit_info_list[transit_i]["delta"].values[seq_mask]
        planet_rvs = planet_rvs * const.c.cgs.to(u.km/u.s).value

        #----------------------------------------------------------------------
        # Cross-correlation
        #----------------------------------------------------------------------
        print("Cross correlating...")
        # Grab the barycentric velocities. We'll take these into account when
        # doing the cross correlation such that we're cross correlating around
        # the *star* rather than around the telescope frame of reference.
        rv_bcors =  \
            -1*transit_info_list[transit_i]["bcor"].values[seq_mask] + rv_star

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
            plot_label="{}_n{}_{}_{}".format(
                ss.label, transit_i+1, seq, species),)

        #----------------------------------------------------------------------
        # Kp-Vsys map
        #----------------------------------------------------------------------
        print("Computing Kp-Vsys map...")

        # Create Kp-Vsys map *per spectral segment*
        cc_rvs_subset, Kp_steps, Kp_vsys_map_per_spec, Kp_vsys_map_combined = \
            sr.compute_Kp_vsys_map(
                cc_rvs=cc_rvs,
                ccv_per_spec=ccv_per_spec,
                ccv_combined=ccv_combined,
                transit_info=transit_info_list[transit_i],
                Kp_lims=ss.Kp_lims,
                Kp_step=ss.Kp_step,
                vsys_lims=ss.kp_vsys_x_range,)

        # Store
        Kp_vsys_map_per_spec_all.append(Kp_vsys_map_per_spec)
        Kp_vsys_map_combined_all.append(Kp_vsys_map_combined)

        # Plot overview Kp-Vsys map
        tplt.plot_kp_vsys_map(
            cc_rvs=cc_rvs_subset,
            Kp_steps=Kp_steps,
            Kp_vsys_map_per_spec=Kp_vsys_map_per_spec,
            Kp_vsys_map_combined=Kp_vsys_map_combined,
            mean_spec_lambdas=np.mean(waves,axis=1),
            plot_label="{}_n{}_{}_{}".format(
                ss.label, transit_i+1, seq, species),)
        
        # Combined Kp-Vsys map for this seq after merging all spectral segments
        tplt.plot_combined_kp_vsys_map_as_snr(
            cc_rvs=cc_rvs_subset,
            Kp_steps=Kp_steps,
            Kp_vsys_maps=Kp_vsys_map_combined,
            plot_title="Night #{} ({})".format(transit_i+1, seq),
            plot_label="{}_n{}_{}_{}".format(
                ss.label, transit_i+1, seq, species),)

Kp_vsys_map_per_spec_all = np.array(Kp_vsys_map_per_spec_all)
Kp_vsys_map_combined_all = np.array(Kp_vsys_map_combined_all)

#------------------------------------------------------------------------------
# Now merge for nightly and combined Kp-Vsys maps
#------------------------------------------------------------------------------
# If we have split the A/B sequences, we need to make 2 sets of summary plots:
# one for each night separately, and one for all nights combined. If we have
# not split the sequences, we only need to make one set of summary plots.
if ss.split_AB_sequences:
    # Setup plot labels and titles
    plot_labels = ["{}_n{}_AB_{}".format(ss.label, ti+1, species)
                   for ti in range(n_transit)]
    plot_labels.append("{}_all_nights_{}".format(ss.label, species))

    plot_titles = ["Night #{} (AB): {}".format(ti+1, species) 
                   for ti in range(n_transit)]
    plot_titles.append("Combined Nights: {}".format(species))

    # Setup a mask such to enable combining the A/B sequences within a given
    # night. e.g. [1, 1, 0, 0] for night 1 when there are two nights.
    map_masks = []
    for ti in range(n_transit):
        map_masks.append([True if (seq_i == ti*2 or seq_i == ti*2+1) else False 
                         for seq_i in range(n_transit*2)])
    # Add in a mask selecting *everything* to combine all nights together
    map_masks.append(np.full(n_transit*2, True))

# If we haven't split up the sequences, set things up to combine all nights
else:
    plot_labels = ["{}_all_nights_{}".format(ss.label, species)]
    plot_titles = ["Combined Nights: {}".format(species)]
    map_masks = [np.full(n_transit*2, True)]

# Now combine all sequences and nights by looping over labels, titles, & masks
for label, title, map_mask in zip(plot_labels, plot_titles, map_masks):
    # Combine Kp-Vsys maps
    # TODO: Currently we just mask negative values and add in quadrature.
    map_per_spec_all_nights = \
        sr.combine_kp_vsys_map(Kp_vsys_map_per_spec_all[map_mask])
    map_combined_all_nights = \
        sr.combine_kp_vsys_map(Kp_vsys_map_combined_all[map_mask])

    # Plot overview Kp-Vsys map with all spectral segments
    tplt.plot_kp_vsys_map(
        cc_rvs=cc_rvs,
        Kp_steps=Kp_steps,
        Kp_vsys_map_per_spec=map_per_spec_all_nights,
        Kp_vsys_map_combined=map_combined_all_nights,
        mean_spec_lambdas=np.mean(waves,axis=1),
        plot_label=label,)

    # Combined Kp-Vsys Map after merging all spectral segments
    tplt.plot_combined_kp_vsys_map_as_snr(
        cc_rvs=cc_rvs,
        Kp_steps=Kp_steps,
        Kp_vsys_maps=map_combined_all_nights,
        plot_title=title,
        plot_label=label,)

#------------------------------------------------------------------------------
# Dump arrays to disk
#------------------------------------------------------------------------------
# Dump the results of CC and the resulting Kp-Vsys map to a pickle, with the
# label, number of transits, template info, and the SYSREM settings obj as a
# whole in the filename.\
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