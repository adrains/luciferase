"""Script to cross-correlate a set of SYSREM residuals with a template 
exoplanet spectrum and produce Kp-Vsys maps. Each night can be run as two
separate time series sequences (once for the A and B nodding frames
separately), or a single time series (A/B interleaved).

The following plots are generated:
 1) Per-sequence cross correlation trace
 2) Per-sequence Kp-Vsys maps for all spectral segments separately
 3) Per-sequence in- vs out-of-transit residual histograms
 4) Per-sequence Kp-Vsys map for all spectral segments
 5) Template auto-correlation per spectral segment
 6) Combined Kp-Vsys map for all spectral segments
 7) Combined CCF plots for all sequences
 8) Combined Kp-Vsys map for all nights + combined nights
"""
import os
import warnings
import numpy as np
import transit.utils as tu
import transit.sysrem as sr
import transit.plotting as tplt

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

# If masking orders during analysis, select subset of wavelength scale
if ss.do_mask_orders_for_analysis:
    n_spec = len(ss.selected_segments)

    waves_full = waves
    waves = waves[ss.selected_segments,:]

# Grab the number of nights
n_transit = len(fluxes_list)

# Import template/s for CC. We have the option to have different templates for
# each night (e.g. in case each night has super-resolution), so we can provide
# multiple planet templates. However, if only a single template is provided, we
# use it for all transits.
wave_templates = []
spectra_templates = []

# If there is only a single template given in the settings file, then use that
# for all nights.
if len(ss.planet_fits) == 1:
    wave_template, spectrum_template = tu.prepare_cc_template(
        cc_settings=ss,
        syst_info=syst_info,
        planet_fits_i=0,
        templ_wl_nm_bounds=(16000,30000),
        continuum_resolving_power=300,)
    
    wave_templates = [wave_template] * n_transit
    spectra_templates = [spectrum_template] * n_transit

# 1:1 match between templates and transits
elif len(ss.planet_fits) == n_transit:
    for transit_i in range(n_transit):
        wave_template, spectrum_template = tu.prepare_cc_template(
            cc_settings=ss,
            syst_info=syst_info,
            planet_fits_i=transit_i,
            templ_wl_nm_bounds=(16000,30000),
            continuum_resolving_power=300,)
        
        wave_templates.append(wave_template)
        spectra_templates.append(spectrum_template)

# Otherwise throw an exception just to be safe
else:
    raise ValueError("# planet templates =/= n_transit")

# Set the RV frame that we'll be running in
rv_frame = "stellar" if ss.run_sysrem_in_stellar_frame else "telluric"

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
if ss.do_mask_orders_for_analysis:
    print("\tSpectral segments\t{}".format(ss.selected_segments))
else:
    print("\tSpectral segments\tAll")
print("\tSplit A/B sequences\t{}".format(ss.split_AB_sequences))
print("\tPer-px resid weighting\t{}".format(
    ss.normalise_resid_by_per_phase_std))
print("\tRV frame\t\t{}".format(rv_frame))
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
planet_rvs_per_seq = []
rv_bcor_median_per_seq = []
in_transit_mask_per_seq = []
Kp_vsys_map_per_spec_all = []
Kp_vsys_map_combined_all = []

rv_star = syst_info.loc["rv_star", "value"]

species_label = "_".join(ss.species_to_cc)
species_list = ", ".join(ss.species_to_cc)

# Setup plotting subfolder
plot_folder = "plots/{}_{}_{}/".format(rv_frame, ss.label, species_label)

# This holds the combined map for each seq/night, plus the final combined map
combined_maps = []
combined_map_titles = []

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
    # Loop over each sequence (A, B, or AB) for this night
    #--------------------------------------------------------------------------
    for seq, seq_mask in zip(sequences, sequence_masks):
        print("\nRunning on night {}/{}, seq {}".format(
            transit_i+1, n_transit, seq))

        # Grab transit_info for this sequence + transit
        transit_info_seq = transit_info_list[transit_i][seq_mask]

        # Grab a mask to discriminate in/out of transit phases
        in_transit = transit_info_seq["is_in_transit_mid"].values

        # Import SYSREM residuals
        resid_all = tu.load_sysrem_residuals_from_fits(
            ss.save_path, ss.label, ss.n_transit, transit_i, seq, rv_frame,)

        n_sysrem_iter = resid_all.shape[0]
        
        # If we're masking out orders for analysis (e.g. because a given
        # species has no lines in certain spectral segments) do so here.
        if ss.do_mask_orders_for_analysis:
            resid_all_full = resid_all
            resid_all = resid_all[:,:,ss.selected_segments,:]

        # Grab planet RVs for overplotting on CC plot. Note that we have to use
        # v_y_mid rather than delta, since delta contains the barycentric and 
        # systemic velocities that we've already corrected for.
        planet_rvs = transit_info_seq["v_y_mid"].values

        #----------------------------------------------------------------------
        # Cross-correlation
        #----------------------------------------------------------------------
        print("Cross correlating...")
        # Grab the barycentric velocities. We'll take these into account when
        # doing the cross correlation such that we're cross correlating around
        # the *star* rather than around the telescope frame of reference. If
        # we're running in the standard telluric frame, we just take these as
        # is. However, if running in the stellar frame we need to account for
        # the fact the data has already been partially shifted.
        bcors = transit_info_seq["bcor"].values

        # There might be unexplained RV offsets from night-to-night, which are
        # accounted for here.
        rv_offset = ss.nightly_rv_offsets[transit_i]

        # If SYSREM was run in the stellar frame, we've already corrected for 
        # the *change* in bcor, but not bcor itself or the stellar velocity.
        if ss.run_sysrem_in_stellar_frame:
            rv_bcors = np.full_like(bcors, -1*bcors[0]) + rv_star + rv_offset

        # Otherwise we ran SYSREM in the telluric frame and there has been no
        # doppler shifting yet, so we need to do it all here.
        else: 
            rv_bcors = -1*bcors + rv_star + rv_offset

        # Bookkeeping for CCF plotting later
        rv_bcor_median_per_seq.append(np.median(-1*bcors) + rv_star)

        # Normalise (i.e. weight) residuals by per-px std computed over all
        # phases (within a given night). This downweights time-varying pixels,
        # specifically those associated with tellurics.
        if ss.normalise_resid_by_per_phase_std:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                std_3D = np.nanstd(resid_all, axis=1)
                std_4D = np.broadcast_to(std_3D[:,None,:,:], resid_all.shape)
                resid_all /= std_4D

        cc_rvs, ccv_per_spec, ccv_combined = sr.cross_correlate_sysrem_resid(
            waves=waves,
            sysrem_resid=resid_all,
            template_wave=wave_templates[transit_i],
            template_spec=spectra_templates[transit_i],
            bcors=rv_bcors,
            cc_rv_step=ss.cc_rv_step,
            cc_rv_lims=ss.cc_rv_lims,
            interpolation_method="linear",)

        # Store
        ccv_per_spec_all.append(ccv_per_spec)
        ccv_combined_all.append(ccv_combined)
        planet_rvs_per_seq.append(planet_rvs)
        in_transit_mask_per_seq.append(in_transit)

        # Plot *per-segment* cross-correlation
        tplt.plot_2D_CCF_per_spectral_segment(
            cc_rvs=cc_rvs,
            ccv_per_spec=ccv_per_spec,
            ccv_combined=ccv_combined,
            mean_spec_lambdas=np.mean(waves,axis=1),
            planet_rvs=planet_rvs,
            in_transit_mask=in_transit,
            plot_label="{}_n{}_{}_{}".format(
                ss.label, transit_i+1, seq, species_label),
            plot_folder=plot_folder,)

        #----------------------------------------------------------------------
        # Cross-correlation histograms
        #----------------------------------------------------------------------
        # Plot histograms comparing the in-transit and out-of-transit phases of
        # the cross-correlation function. TODO: just plot for the velocities
        # where we expect the planet to be.
        hist_title = "{}, Night #{} ({}): {}".format(
            ss.label, transit_i+1, seq, species_list)

        tplt.plot_in_and_out_of_transit_histograms(
            ccv_per_spec=None,
            ccv_combined=ccv_combined,
            in_transit=in_transit,
            mean_spec_lambdas=np.mean(waves,axis=1),
            plot_label="{}_n{}_{}_{}".format(
                ss.label, transit_i+1, seq, species_label),
            plot_title=hist_title,
            plot_folder=plot_folder,)

        #----------------------------------------------------------------------
        # Kp-Vsys map
        #----------------------------------------------------------------------
        # TODO: run all this twice, once using out-of-transit phases only, and
        # once using in-transit phases only for comparison purposes.
        print("Computing Kp-Vsys map...")

        # Create Kp-Vsys map *per spectral segment*
        vsys_steps, Kp_steps, Kp_vsys_map_per_spec, Kp_vsys_map_combined = \
            sr.compute_Kp_vsys_map(
                cc_rvs=cc_rvs,
                ccv_per_spec=ccv_per_spec[:,in_transit],
                ccv_combined=ccv_combined[:,in_transit],
                transit_info=transit_info_seq[in_transit],
                Kp_lims=ss.Kp_lims,
                Kp_step=ss.Kp_step,
                vsys_lims=ss.kp_vsys_x_range,
                interpolation_method="linear",)

        # Store
        Kp_vsys_map_per_spec_all.append(Kp_vsys_map_per_spec)
        Kp_vsys_map_combined_all.append(Kp_vsys_map_combined)
        combined_maps.append(Kp_vsys_map_combined)

        # Plot overview Kp-Vsys map
        tplt.plot_kp_vsys_map(
            vsys_steps=vsys_steps,
            Kp_steps=Kp_steps,
            Kp_vsys_map_per_spec=Kp_vsys_map_per_spec,
            Kp_vsys_map_combined=Kp_vsys_map_combined,
            mean_spec_lambdas=np.mean(waves,axis=1),
            plot_label="{}_n{}_{}_{}".format(
                ss.label, transit_i+1, seq, species_label),
            plot_folder=plot_folder,)
        
        # Store so we can plot all the combined maps at the end
        map_title = "Night #{} ({})".format(transit_i+1, seq)
        combined_map_titles.append(map_title)

    # For this night do an autocorrelation with our template spectrum and plot
    # the results. Our objective here is to look for aliases/other cross-
    # correlation peaks that could manifest as false positives in the Kp-Vsys
    # map space.
    ac_rvs, ac_2D, ac_comb = sr.compute_template_autocorrelation(
        wave_obs=waves,
        wave_template=wave_templates[transit_i],
        flux_template=spectra_templates[transit_i],
        rv_syst=np.median(rv_bcors),
        rv_lims=ss.kp_vsys_x_range,
        rv_step=ss.Kp_step,)
    
    tplt.plot_autocorrelation(
        wave_obs=waves,
        autocorr_rvs=ac_rvs,
        autocorr_2D=ac_2D,
        autocorr_comb=ac_comb,
        plot_label="{}_n{}_{}".format(ss.label, transit_i+1, species_label),
        plot_title="{}, Night {}: {}".format(
            ss.label, transit_i+1, species_label),
        plot_folder=plot_folder,)

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
    plot_labels = ["{}_n{}_AB_{}".format(ss.label, ti+1, species_label)
                   for ti in range(n_transit)]
    plot_labels.append("{}_all_nights_{}".format(ss.label, species_label))

    plot_titles = ["Night #{} (AB)".format(ti+1) for ti in range(n_transit)]
    plot_titles.append("Combined Nights")

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
    plot_labels = ["{}_all_nights_{}".format(ss.label, species_label)]
    plot_titles = ["Combined Nights: {}".format(species_list)]
    map_masks = [np.full(n_transit, True)]

# Now combine all sequences and nights by looping over labels, titles, & masks
for label, title, map_mask in zip(plot_labels, plot_titles, map_masks):
    # Combine Kp-Vsys maps
    # TODO: Currently we just mask negative values and add in quadrature.
    map_per_spec_all_nights = \
        sr.combine_kp_vsys_map(Kp_vsys_map_per_spec_all[map_mask])
    map_combined_all_nights = \
        sr.combine_kp_vsys_map(Kp_vsys_map_combined_all[map_mask])

    combined_maps.append(map_combined_all_nights)
    combined_map_titles.append(title)

    # Plot overview Kp-Vsys map with all spectral segments
    tplt.plot_kp_vsys_map(
        vsys_steps=vsys_steps,
        Kp_steps=Kp_steps,
        Kp_vsys_map_per_spec=map_per_spec_all_nights,
        Kp_vsys_map_combined=map_combined_all_nights,
        mean_spec_lambdas=np.mean(waves,axis=1),
        plot_label=label,
        plot_folder=plot_folder,)

# Scale combined Kp-Vsys maps to be in units of SNR
combined_maps = np.stack(combined_maps)

snr_maps, max_snr, vsys_at_max_snr, kp_at_max_snr = sr.calc_Kp_vsys_map_snr(
    vsys_steps=vsys_steps,
    Kp_steps=Kp_steps,
    Kp_vsys_maps=combined_maps,
    vsys_rv_exclude=ss.kp_vsys_snr_rv_exclude)

# Make a plot of all the sequences/nights together
tplt.plot_combined_kp_vsys_map_as_snr(
    vsys_steps=vsys_steps,
    Kp_steps=Kp_steps,
    Kp_vsys_maps_snr=snr_maps,
    plot_title=combined_map_titles,
    Kp_expected=ss.kp_expected,
    max_snr_values=max_snr,
    vsys_at_max_snr=vsys_at_max_snr,
    kp_at_max_snr=kp_at_max_snr,
    plot_suptitle="[{}] {}: {}".format(rv_frame, ss.label, species_list),
    plot_label="{}_all_seq_{}".format(ss.label, species_label),
    plot_folder=plot_folder,)

# Also plot combined CCFs for each sequence separately
tplt.plot_2D_CCF_per_seq(
    cc_rvs=cc_rvs,
    ccv_per_seq=ccv_combined_all,
    plot_titles=combined_map_titles,
    planet_rvs_per_seq=planet_rvs_per_seq,
    rv_bcor_median_per_seq=rv_bcor_median_per_seq,
    in_transit_mask_per_seq=in_transit_mask_per_seq,
    plot_label="{}_{}".format(
        ss.label, species_label),
    plot_folder=plot_folder,)

#------------------------------------------------------------------------------
# Dump arrays to disk
#------------------------------------------------------------------------------
# Dump the results of CC and the resulting Kp-Vsys map to a pickle, with the
# label, number of transits, template info, and the SYSREM settings obj as a
# whole in the filename.\
fn_label = "_".join([
    "cc_results", ss.label, str(ss.n_transit), str("template"), species_label])
fn = os.path.join(ss.save_path, "{}.pkl".format(fn_label))

tu.dump_cc_results(
    filename=fn,
    cc_rvs=cc_rvs,                  # CC RV steps
    ccv_ps=ccv_per_spec_all,        # Nightly CC maps for all spectral segments
    ccv_comb=ccv_combined_all,      # Combined CC map for each night
    vsys_steps=vsys_steps,          # Vsys steps, =/= CC RV steps
    Kp_steps=Kp_steps,              # Kp steps
    Kp_vsys_map_ps=Kp_vsys_map_per_spec_all,            # Per seg. Kp-Vsys map
    Kp_vsys_map_comb=Kp_vsys_map_combined_all,          # Comb seg Kp-Vsys map
    Kp_vsys_map_ps_all_nights=map_per_spec_all_nights,  # Per seg, comb nights
    Kp_vsys_map_comb_all_nights=map_combined_all_nights,# Comb seg, comb night
    nightly_snr_maps=snr_maps,      # SNR scaled, comb seg, nightly + comb
    max_snr=max_snr,                # Max SNR per map
    vsys_at_max_snr=vsys_at_max_snr, # Vsys coord of max SNR per map
    kp_at_max_snr=kp_at_max_snr,    # Kp coord of max SNR per map
    sysrem_settings=ss,)            # Settings used for this CC