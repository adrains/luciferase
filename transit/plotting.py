"""
Plotting functions associated with our transit modelling.
"""
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import transit.utils as tu
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as plticker
from tqdm import tqdm
from astropy.stats import sigma_clip
from matplotlib.patches import Rectangle

# Ensure the plotting folder exists to save to
here_path = os.path.dirname(__file__)
plot_dir = os.path.abspath(os.path.join(here_path, "..", "plots"))

if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

def plot_iteration():
    """
    """
    pass
 

def plot_all_input_spectra(
    waves,
    fluxes_all_list,
    transit_info_list,
    n_transits):
    """Function to plot spectra at all phases and colour code by timestep, with
    each transit plotted in its own panel.

    Parameters
    ----------
    waves: 2D float array
        Wavelength scale of shape [n_spec, n_wave].

    fluxes_all_list: list of 3D float array
        List of length n_transits of flux arrays corresponding to wavelengths
        of shape [n_phase, n_spec, n_wave].
    
    transit_info_list: list of pandas DataFrames
        List of transit info (length n_transits) DataFrames containing 
        information associated with each transit time step. Each DataFrame has
        columns:

        ['mjd_start', 'mjd_mid', 'mjd_end', 'jd_start', 'jd_mid', 'jd_end',
         'airmass', 'bcor', 'hcor', 'ra', 'dec', 'exptime_sec', 'nod_pos',
         'raw_file', 'phase_start', 'is_in_transit_start', 'r_x_start',
         'r_y_start', 'r_z_start', 'v_x_start', 'v_y_start', 'v_z_start',
         's_projected_start', 'scl_start', 'mu_start',
         'planet_area_frac_start', 'phase_mid', 'is_in_transit_mid',
         'r_x_mid', 'r_y_mid', 'r_z_mid', 'v_x_mid', 'v_y_mid', 'v_z_mid',
         's_projected_mid', 'scl_mid', 'mu_mid', 'planet_area_frac_mid',
         'phase_end', 'is_in_transit_end', 'r_x_end', 'r_y_end', 'r_z_end',
         'v_x_end', 'v_y_end', 'v_z_end', 's_projected_end', 'scl_end',
         'mu_end', 'planet_area_frac_end', 'gamma', 'beta', 'delta']

    n_transits: int
        Number of transits the data represents.
    """
    #plt.close("all")
    fig, axes = plt.subplots(
        nrows=n_transits, figsize=(15, 5), sharex=True, sharey=True,)

    if n_transits == 1:
        axes = [axes]

    for transit_i in range(n_transits):
        # Get colour bar info
        jds = transit_info_list[transit_i]["jd_mid"].values
        mid_jd = np.mean(jds)
        hours = (jds - mid_jd) * 24

        # Setup colour bar
        norm = plt.Normalize(np.min(hours), np.max(hours))
        cmap = plt.get_cmap("plasma")
        colours = cmap(norm(hours))

        cmap = cm.ScalarMappable(norm=norm, cmap=cm.plasma)

        n_phase = fluxes_all_list[transit_i].shape[0]

        for phase_i in range(n_phase):
            for wave_i in range(len(waves)):
                axes[transit_i].plot(
                    waves[wave_i],
                    fluxes_all_list[transit_i][phase_i, wave_i],
                    color=colours[phase_i],
                    linewidth=0.5,
                    label=phase_i,)

        axes[transit_i].set_ylim(
            -1000, 3*np.nanmedian(fluxes_all_list[transit_i]))
        axes[transit_i].set_ylabel("Flux (counts)")

        cb = fig.colorbar(cmap, ax=axes[transit_i])
        cb.set_label("Time from mid-point (hr)")

    axes[-1].set_xlabel("Wavelength")
    plt.tight_layout()
    plt.savefig("plots/input_spectra.pdf")


def plot_component_spectra(
    waves,
    fluxes,
    telluric_tau,
    planet_trans,
    scale_vector,
    transit_num,
    star_name,
    ref_fluxes=None,
    ref_telluric_tau=None,
    ref_planet_trans=None,
    ref_scale_vector=None,
    linewidth=0.5,
    plot_suptitle=True,):
    """Plots fluxes, telluric transmission, and planet transmission in
    respective subplots. Can be used for both Aronson fitted results, or
    simulated components.

    Parameters
    ----------
    waves: float array
        Wavelength scale of shape [n_spec, n_px].

    fluxes: 2D float array
        Model stellar flux component of shape [n_spec, n_px].

    telluric_tau: 2D float array
        Model telluric tau component of shape [n_spec, n_px].

    planet_trans: 2D float array
        Model planet transmission component of shape [n_spec, n_px].

    scale_vector: 1D float array
        Adoped scale/slit losses vector of shape [n_phase].

    transit_num: int
        Transit number for use when titling plots.

    star_name: str
        Name of the star.

    ref_fluxes, ref_telluric_tau, ref_planet_trans, ref_scale_vector: 
    float array or None
        Reference vectors to plot against fluxes, tau, trans, and scale.
    """
    # Intialise subplots
    fig, (ax_flux, ax_tell, ax_trans, ax_scale) = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(20, 8),)
    
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.98, top=0.95, wspace=0.1, hspace=0.4)

    # Share x axes
    ax_flux.get_shared_x_axes().join(ax_flux, ax_tell, ax_trans)

    # Plot each spectral segment
    for spec_i in range(waves.shape[0]):
        ax_flux.plot(
            waves[spec_i],
            fluxes[spec_i],
            linewidth=linewidth,
            color="r",
            alpha=0.8,
            label="Stellar" if spec_i == 0 else None,)
        
        ax_tell.plot(
            waves[spec_i],
            np.exp(-telluric_tau[spec_i]),
            linewidth=linewidth,
            color="b",
            alpha=0.8,
            label="Telluric" if spec_i == 0 else None,)
        
        ax_trans.plot(
            waves[spec_i],
            planet_trans[spec_i],
            linewidth=0.4,
            color="g",
            alpha=0.8,
            label="Planet" if spec_i == 0 else None)
        
        ax_scale.plot(
            np.arange(len(scale_vector)),
            scale_vector,
            marker="o",
            linewidth=linewidth,
            color="c",
            alpha=0.8,
            label="Slit Loss (Norm)" if spec_i == 0 else None)
        
        ax_scale.hlines(
            y=1,
            xmin=0,
            xmax=len(scale_vector),
            colors="k",
            linestyles="dotted",)
        
        # Plot reference vectors if we have them
        if (ref_fluxes is not None
            and fluxes.shape == ref_fluxes.shape):
            ax_flux.plot(
                waves[spec_i],
                ref_fluxes[spec_i],
                linewidth=linewidth,
                color="k",
                alpha=0.8,)
            
        if (ref_telluric_tau is not None
            and telluric_tau.shape == ref_telluric_tau.shape):
            ax_tell.plot(
                waves[spec_i],
                np.exp(-ref_telluric_tau[spec_i]),
                linewidth=linewidth,
                color="k",
                alpha=0.8,)
        
        if (ref_planet_trans is not None
            and planet_trans.shape == ref_planet_trans.shape):
            ax_trans.plot(
                waves[spec_i],
                ref_planet_trans[spec_i],
                linewidth=linewidth,
                color="k",
                alpha=0.8,)
            
        if (ref_scale_vector is not None
            and scale_vector.shape == ref_scale_vector.shape):
            ax_scale.plot(
                np.arange(len(scale_vector)),
                scale_vector,
                marker="o",
                linewidth=linewidth,
                color="c",
                alpha=0.8,)

    ax_flux.set_ylabel("Flux")
    ax_tell.set_ylabel("Transmission")
    ax_trans.set_ylabel(r"$(R_P/R_\star)^2$")
    ax_scale.set_ylabel("Epoch Scale")

    ax_flux.legend(loc="center right")
    ax_tell.legend(loc="center right")
    ax_trans.legend(loc="center right")
    ax_scale.legend(loc="center right")

    ax_flux.set_xticks([])
    ax_tell.set_xticks([])

    ax_trans.xaxis.set_major_locator(plticker.MultipleLocator(base=10))
    ax_trans.xaxis.set_minor_locator(plticker.MultipleLocator(base=1))

    ax_scale.xaxis.set_major_locator(plticker.MultipleLocator(base=5))
    ax_scale.xaxis.set_minor_locator(plticker.MultipleLocator(base=1))

    if plot_suptitle:
        fig.suptitle("Transit #{:0.0f}".format(transit_num+1))
    
    ax_trans.set_xlabel(r"Wavelength (${\rm \AA}$)")
    ax_scale.set_xlabel("Epoch #")
    ax_scale.set_ylim(0.5, 1.5)

    fig_name = "plots/{}_transit_{:0.0f}_components.pdf".format(
        star_name, transit_num)
    plt.savefig(fig_name)


def plot_epoch_model_comp(
    waves,
    obs_spec,
    model,
    fluxes,
    telluric_tau,
    planet_trans,
    scale,
    transit_info,
    ref_fluxes=None,
    ref_telluric_tau=None,
    ref_planet_trans=None,
    ref_scale=None):
    """Plots diagnostic plot per phase.

    Parameters
    ----------
    waves: float array
        Wavelength scale of shape [n_spec, n_px].

    model: 3D float array
        Observed/simulated spectra of shape [n_phase, n_spec, n_px].

    model: 3D float array
        Fitted model (star + tellurics + planet) matrix of shape
        [n_phase, n_spec, n_px].

    fluxes: 2D float array
        Model stellar flux component of shape [n_spec, n_px].

    telluric_tau: 2D float array
        Model telluric tau component of shape [n_spec, n_px].

    planet_trans: 2D float array
        Model planet transmission component of shape [n_spec, n_px].
    
    scale: 1D float array
        Model scale vector of shape [n_phase].

    transit_info: pandas DataFrame
        Transit info DataFrames containing information associated with each 
        transit time step with columns:

        ['mjd_start', 'mjd_mid', 'mjd_end', 'jd_start', 'jd_mid', 'jd_end',
         'airmass', 'bcor', 'hcor', 'ra', 'dec', 'exptime_sec', 'nod_pos',
         'raw_file', 'phase_start', 'is_in_transit_start', 'r_x_start',
         'r_y_start', 'r_z_start', 'v_x_start', 'v_y_start', 'v_z_start',
         's_projected_start', 'scl_start', 'mu_start',
         'planet_area_frac_start', 'phase_mid', 'is_in_transit_mid',
         'r_x_mid', 'r_y_mid', 'r_z_mid', 'v_x_mid', 'v_y_mid', 'v_z_mid',
         's_projected_mid', 'scl_mid', 'mu_mid', 'planet_area_frac_mid',
         'phase_end', 'is_in_transit_end', 'r_x_end', 'r_y_end', 'r_z_end',
         'v_x_end', 'v_y_end', 'v_z_end', 's_projected_end', 'scl_end',
         'mu_end', 'planet_area_frac_end', 'gamma', 'beta', 'delta']

    ref_fluxes, ref_telluric_tau, ref_planet_trans, ref_scale: 
    float array or None
        Reference vectors to plot against fluxes, tau, trans, and scale.
    """
    # Pull out variables for convenience
    (n_phase, n_spec, n_wave) = obs_spec.shape

    # loop over all epochs
    for phase_i, transit_epoch in transit_info.iterrows():
        # TODO Temporary HACK
        if phase_i != 100:
            continue

        transit_i = transit_info.iloc[phase_i]["transit_num"]

        night_masks = [transit_info["transit_num"].values == trans_i
            for trans_i in range(telluric_tau.shape[0])]

        # Initialise subplot
        fig, (ax_spec, ax_flux, ax_tell, ax_trans, ax_scale) = plt.subplots(
            nrows=5,
            ncols=1,
            figsize=(15, 10),)
        
        # Setup lower panel for residuals
        plt.setp(ax_spec.get_xticklabels(), visible=False)
        divider = make_axes_locatable(ax_spec)
        res_ax = divider.append_axes("bottom", size="30%", pad=0)
        fig.add_axes(res_ax, sharex=ax_spec)
        ax_spec.get_shared_x_axes().join(
            ax_spec, res_ax, ax_flux, ax_tell, ax_trans)

        # Grab velocities for this phase
        gamma = transit_epoch["gamma"]
        beta = transit_epoch["beta"]
        delta = transit_epoch["delta"]

        for spec_i in range(n_spec):
            # Initialise derivatives
            flux_2 = tu.bezier_init(x=waves[spec_i], y=fluxes[spec_i],)
            trans_2 = tu.bezier_init(x=waves[spec_i], y=planet_trans[spec_i],)

            # -----------------------------------------------------------------
            # Plot model + residuals
            # -----------------------------------------------------------------
            ax_spec.plot(
                waves[spec_i],
                model[phase_i, spec_i],
                linewidth=0.4,
                color="m",
                alpha=0.8,
                label="Model" if spec_i == 0 else "_nolegend_",)
            
            ax_spec.plot(
                waves[spec_i],
                obs_spec[phase_i, spec_i],
                linewidth=0.4,
                color="k",
                alpha=0.8,
                label="Input" if spec_i == 0 else "_nolegend_",)
            
            resid = obs_spec[phase_i, spec_i] - model[phase_i, spec_i]

            res_ax.plot(
                waves[spec_i],
                resid,
                linewidth=0.4,
                color="m",
                alpha=0.8,)

            # -----------------------------------------------------------------
            # Plot flux
            # -----------------------------------------------------------------
            flux = tu.doppler_shift(
                x=waves[spec_i],
                y=fluxes[spec_i],
                gamma=-gamma,
                y2=flux_2)
            
            ax_flux.plot(
                waves[spec_i],
                flux,
                linewidth=0.4,
                color="r",
                alpha=0.8,
                label="Model" if spec_i == 0 else "_nolegend_",)
            
            if (ref_fluxes is not None
            and fluxes.shape == ref_fluxes.shape):
                ref_flux_2 = tu.bezier_init(
                    x=waves[spec_i],
                    y=ref_fluxes[spec_i],)

                ref_flux = tu.doppler_shift(
                    x=waves[spec_i],
                    y=ref_fluxes[spec_i],
                    gamma=-gamma,
                    y2=ref_flux_2)


                ax_flux.plot(
                    waves[spec_i],
                    ref_flux,
                    linewidth=0.4,
                    color="k",
                    alpha=0.8,
                    label="Input" if spec_i == 0 else "_nolegend_",)

            # -----------------------------------------------------------------
            # Plot tellurics
            # -----------------------------------------------------------------
            tellurics = np.exp(-telluric_tau[transit_i, spec_i])

            ax_tell.plot(
                waves[spec_i],
                tellurics,
                linewidth=0.4,
                color="b",
                alpha=0.8,
                label="Model" if spec_i == 0 else "_nolegend_",)
            
            if (ref_telluric_tau is not None
            and telluric_tau.shape == ref_telluric_tau.shape):
                ax_tell.plot(
                    waves[spec_i],
                    np.exp(-ref_telluric_tau[transit_i, spec_i]),
                    linewidth=0.4,
                    color="k",
                    alpha=0.8,
                    label="Input" if spec_i == 0 else "_nolegend_",)

            # -----------------------------------------------------------------
            # Plot planet spectrum
            # -----------------------------------------------------------------
            trans = tu.doppler_shift(
                x=waves[spec_i],
                y=planet_trans[spec_i],
                gamma=-delta,
                y2=trans_2,)
            
            ax_trans.plot(
                waves[spec_i],
                trans,
                linewidth=0.4,
                color="g",
                alpha=0.8,
                label="Model" if spec_i == 0 else "_nolegend_",)
            
            if (ref_planet_trans is not None
            and planet_trans.shape == ref_planet_trans.shape):
                ref_trans_2 = tu.bezier_init(
                    x=waves[spec_i],
                    y=ref_planet_trans[spec_i],)

                ref_trans = tu.doppler_shift(
                    x=waves[spec_i],
                    y=ref_planet_trans[spec_i],
                    gamma=-delta,
                    y2=ref_trans_2)
                
                ax_trans.plot(
                    waves[spec_i],
                    ref_trans,
                    linewidth=0.4,
                    color="k",
                    alpha=0.8,
                    label="Input" if spec_i == 0 else "_nolegend_",)
                
            # -----------------------------------------------------------------
            # Plot scale
            # -----------------------------------------------------------------
            # Plot fitted scale for this transit
            ax_scale.plot(
                scale[night_masks[transit_i]],
                linewidth=0.4,
                color="c",
                alpha=0.8,
                label="Scale" if spec_i == 0 else "_nolegend_",)
            
            # Plot marker for this phase
            ax_scale.plot(
                transit_info.iloc[phase_i]["obs_i"],
                scale[phase_i],
                linewidth=0.4,
                marker="o",
                color="c",
                alpha=0.8,)
            
            if (ref_scale is not None
            and scale.shape == ref_scale.shape):
                ax_scale.plot(
                    ref_scale[night_masks[transit_i]],
                    linewidth=0.4,
                    marker="+",
                    color="k",
                    alpha=0.8,
                    label="Input" if spec_i == 0 else "_nolegend_",)

        # Finish setup
        ax_spec.set_title("Input vs Model Spectrum")
        ax_flux.set_title("Stellar flux")
        ax_tell.set_title("Telluric Transmission")
        ax_trans.set_title("Planet Transmission")
        ax_trans.set_xlabel(r"Wavelength (${\rm \AA}$)")
        ax_scale.set_xlabel(r"Phase #")

        ax_spec.legend(loc="lower center", ncol=2)
        ax_flux.legend(loc="lower center", ncol=2)
        ax_tell.legend(loc="lower center", ncol=2)
        ax_trans.legend(loc="lower center", ncol=2)
        ax_scale.legend(loc="lower center", ncol=2)

        plt.tight_layout()


def plot_cleaned_continuum_normalised_spectra(
    waves,
    spectra_init,
    spectra_clean,
    sequences,
    sequence_masks,
    figsize=(200,5),
    linewidths=(0.2,0.5),
    plot_label="",
    plot_folder="plots/",
    plot_title="",):
    """Function to plot a comparison when running clean_and_interpolate_spectra
    from transit.utils. Plots the A/B spectra before and after, plus their
    respective means.

    Parameters
    ----------
    waves: 2D float array
        Wavelength vector of shape [n_spec, n_px]

    spectra_init, spectra_clean: 3D float arrays
        3D float arrays of shape [n_phase, n_spec, n_px] of the spectra before
        and after cleaning respectively.

    sequences: str list
        Sequence IDs, typically ["A", "B"] for nodding positions.
    
    sequence_masks: list of 1D boolean arrays
        List of boolean masks of length [n_phase] indicating which sequence
        each phase belongs to.
        
    fig_size: float array, default: (200,5)
        Shape of the figure.

    linewidths: float tuple, default: (0.1,0.5)
        Linewidths for the per-phase and mean sequence spectra respectively.

    plot_label: str, default: ""
        Filename label for plot, will be saved as sysrem_resid_<label>.pdf/png.

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots.

    plot_title: str, default: ""
        Suptitle for the plot.
    """
    (n_phase, n_spec, n_px) = spectra_init.shape

    fig, axes = plt.subplots(
        nrows=2,
        ncols=n_spec,
        figsize=figsize,
        sharex="col",
        sharey=True)
    
    fig.subplots_adjust(
        left=0.02, bottom=0.1, right=0.995, top=0.95, wspace=0.05, hspace=0.05)
    
    for seq_i, (seq, seq_mask) in enumerate(zip(sequences, sequence_masks)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ff1 = spectra_init[seq_mask]
            ff1_mn = np.nanmean(ff1, axis=0)
            ff2 = spectra_clean[seq_mask]
            ff2_mn = np.nanmean(ff2, axis=0)
        n_phase = ff1.shape[0]
        for spec_i in range(n_spec):
            for phase_i in range(n_phase):
                # Plot A/B for each phase
                axes[0, spec_i].plot(
                    waves[spec_i],
                    ff1[phase_i, spec_i],
                    linewidth=linewidths[0],
                    alpha=0.5,
                    c="r" if seq=="A" else "b",
                    label=seq if (spec_i==0 and phase_i==0) else None,)
                axes[1, spec_i].plot(
                    waves[spec_i],
                    ff2[phase_i, spec_i],
                    linewidth=linewidths[0],
                    alpha=0.5,
                    c="r" if seq=="A" else "b",)
            # Plot mean A/B for each spectral segment
            axes[0, spec_i].plot(
                waves[spec_i],
                ff1_mn[spec_i],
                linewidth=linewidths[1],
                c="darkred" if seq=="A" else "navy",
                zorder=100,
                label="{} (Mean)".format(seq) if spec_i==0 else None,)
            axes[1, spec_i].plot(
                waves[spec_i],
                ff2_mn[spec_i],
                linewidth=linewidths[1],
                c="darkred" if seq=="A" else "navy",
                zorder=100,
                label="{} (Mean)".format(seq) if spec_i==0 else None,)
            
            axes[1, spec_i].set_xlim(
                waves[spec_i,0]-0.5, waves[spec_i,-1]+0.5)
            axes[1, spec_i].tick_params(axis='x', labelrotation=45)
            axes[1, spec_i].xaxis.set_major_locator(
                plticker.MultipleLocator(base=5))
            axes[1, spec_i].xaxis.set_minor_locator(
                plticker.MultipleLocator(base=1))

    leg = axes[0, 0].legend(
        loc="upper left",
        ncol=4,
        bbox_to_anchor=(0, 1.15),)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.5)

    plt.suptitle(plot_title, fontsize="small")

    # Check save folder and save
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    if plot_label == "":
        plot_fn = os.path.join(plot_folder, "cleaned_spectra_comparison")
    else:
        plot_fn = os.path.join(
            plot_folder, "cleaned_spectra_comparison_{}".format(plot_label))

    plt.savefig("{}.pdf".format(plot_fn))
    plt.savefig("{}.png".format(plot_fn), dpi=300)


def plot_sysrem_residuals(
    waves,
    resid,
    fig_size=(18,6),
    do_sigma_clip=False,
    sigma_upper=5,
    sigma_lower=5,
    max_iterations=5,
    plot_label="",
    plot_folder="plots/",
    plot_title="",):
    """Function to plot a grid of residuals as output from SYSREM. The grid has
    n_rows = n_sysrem_iter, and n_cols = n_spec.

    Parameters
    ----------
    waves: 2D float array
        Wavelength vector of shape [n_spec, n_px]
    
    resid: 4D float array
        4D float array of residuals with shape 
        [n_sysrem_iter, n_phase, n_spec, n_px]

    fig_size: float array, default: (18,6)
        Shape of the figure.

    do_sigma_clip: boolean, default: False
        Whether to sigma clip the residuals prior to plotting to gain
        additional dynamic range.

    sigma_upper, sigma_lower: float, default: 5
        Upper and lower sigma values to clip if do_sigma_clip.

    max_iterations: int, default: 5
        Max iterations to use when sigma clipping.

    plot_label: str, default: ""
        Filename label for plot, will be saved as sysrem_resid_<label>.pdf/png.

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots.

    plot_title: str, default: ""
        Suptitle for the plot.
    """
    resid = resid.copy()

    # Grab dimensions for convenience
    (n_sysrem_iter, n_phase, n_spec, n_px) = resid.shape

    plt.close("all")
    fig, axes = plt.subplots(
        nrows=n_sysrem_iter,
        ncols=n_spec,
        #sharex=True,
        figsize=fig_size,)
    
    plt.subplots_adjust(
        left=0.025, bottom=0.075, right=0.99, top=0.925, wspace=0.1)

    for sr_iter_i in range(n_sysrem_iter):
        desc = "Plotting SYSREM resid for iter #{}".format(sr_iter_i)

        for spec_i in tqdm(range(n_spec), leave=False, desc=desc):
            # [Optional] Sigma clip the noise away to aid in visualisation
            if do_sigma_clip:
                resid_ma = sigma_clip(
                    data=resid[sr_iter_i, :, spec_i],
                    sigma_lower=sigma_lower,
                    sigma_upper=sigma_upper,
                    maxiters=max_iterations,)
                
                resid_ith = resid_ma.data
                mask = resid_ma.mask
                resid_ith[mask] = np.nan

            # Otherwise plot residuals as is
            else:
                resid_ith = resid[sr_iter_i, :, spec_i]
            
            # [x_min, x_max, y_min, y_max]
            extent = [np.min(waves[spec_i]), np.max(waves[spec_i]), n_phase, 0]
            
            # Grab axis handle for convenience
            axis = axes[sr_iter_i, spec_i]

            cmap = axis.imshow(
                X=resid_ith,
                aspect="auto",
                interpolation="none",
                extent=extent,
                cmap="plasma",)

            axis.tick_params(axis='both', which='major', labelsize="x-small")
            axis.xaxis.set_major_locator(plticker.MultipleLocator(base=4))
            axis.xaxis.set_minor_locator(plticker.MultipleLocator(base=2))
            axes[sr_iter_i, spec_i].tick_params(axis='x', labelrotation=45)

            # Only show xticks on the bottom
            if sr_iter_i != n_sysrem_iter-1:
                axis.set_xticks([])

            # Only show Y labels on the left
            if spec_i == 0:
                axis.set_ylabel(
                    ylabel="Iter #{}".format(sr_iter_i),
                    fontsize="x-small")
            
            # Only show yticks on the left
            else:
                axis.set_yticks([])
            
            # Only show titles on the top
            if sr_iter_i == 0:
                axis.set_title(
                    label=r"${:0.0f}\,\mu$m".format(np.median(waves[spec_i])),
                    fontsize="x-small")
    
    plt.suptitle(plot_title, fontsize="small")

    # Check save folder and save
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    if plot_label == "":
        plot_fn = os.path.join(plot_folder, "sysrem_resid")
    else:
        plot_fn = os.path.join(
            plot_folder, "sysrem_resid_{}".format(plot_label))

    plt.savefig("{}.pdf".format(plot_fn))
    plt.savefig("{}.png".format(plot_fn), dpi=300)


def plot_sysrem_std(
    waves,
    resid_all,
    plot_label="",
    plot_folder="plots/",
    plot_title="",):
    """Function to plot the per-segment, per-phase (computed over all px)
    standard deviation as a function of SYSREM iteration. The idea is to
    inspect the *result* of the detrending (rather than the PCA components
    themselves), and see if all phases/spectral segments are similarly being
    affected by the detrending.

    Parameters
    ----------
    waves: 2D float array
        Wavelength vector of shape [n_spec, n_px]
    
    resid_all: 4D float array
        4D float array of residuals with shape 
        [n_sysrem_iter, n_phase, n_spec, n_px]

    plot_label: str, default: ""
        Filename label for plot, will be saved as sysrem_resid_<label>.pdf/png.

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots.

    plot_title: str, default: ""
        Suptitle for the plot.
    """
    (n_sr, n_phase, n_spec, n_px) = resid_all.shape

    mean_spec_lambdas = np.mean(waves,axis=1)
    
    # Setup our x-values--we skip plotting the first two iterations for dynamic
    # range reasons.
    sysrem_iterations = np.arange(2, n_sr)

    # Plot one panel per spectral segment, with one line per phase, showing
    # the standard deviation as a function of SYSREM iteration.
    plt.close("all")
    fig, axes = plt.subplots(ncols=n_spec, figsize=(25,10), sharey=True)
    plt.subplots_adjust(
            left=0.05, bottom=0.075, right=0.99, top=0.95, wspace=0.1)

    for spec_i in range(n_spec):
        for phase_i in range(n_phase):
            cmap = cm.get_cmap("magma")
            colour = cmap(phase_i/n_phase)
            axes[spec_i].plot(
                sysrem_iterations,
                np.nanstd(resid_all[2:,phase_i,spec_i,:], axis=-1),
                c=colour,
                label=phase_i)
        axes[spec_i].set_xlabel("SYSREM iter", fontsize="small")
        axes[spec_i].set_title(
            label=r"${:0.0f}\,\mu$m".format(mean_spec_lambdas[spec_i]),
            fontsize="x-small")
        
        axes[spec_i].xaxis.set_major_locator(plticker.MultipleLocator(base=4))
        axes[spec_i].xaxis.set_minor_locator(plticker.MultipleLocator(base=1))

    axes[0].set_ylabel("Per-segment std")

    axes[-1].legend(
        ncol=n_phase//2+1,
        fontsize="small",
        loc="upper right",)
    
    plt.suptitle(plot_title, fontsize="small")

    # Check save folder and save
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    if plot_label == "":
        plot_fn = os.path.join(plot_folder, "sysrem_per_seg_per_phase_std")
    else:
        plot_fn = os.path.join(
            plot_folder, "sysrem_per_seg_per_phase_std_{}".format(plot_label))

    plt.savefig("{}.pdf".format(plot_fn))
    plt.savefig("{}.png".format(plot_fn), dpi=300)


def plot_sysrem_coefficients(
    waves,
    phases_list,
    labels_per_seq_list,
    coeff_phase_all_list,
    coeff_wave_all_list,
    plot_label="",
    plot_folder="plots/",):
    """Function to plot the per-segment fitted SYSREM coefficients (both phase
    and wavelength) for each detrending iteration for a set of observational
    sequences, e.g. multiple nights or just an A and B sequence from within a
    single night. Two plots are created, with [n_spec x n_sysrem_iter] panels,
    and lines for each plotted sequence.

    Parameters
    ----------
    waves: 2D float array
        Wavelength vector of shape [n_spec, n_px]
    
    phases_list: list of 1D float arrays
        List of 1D arrays containing the phases for each plotted sequence, of
        lengths [n_seq] and [n_phase]

    labels_per_seq_list: str list
        List of string labels to use as legend entries for each sequence, of
        length [n_seq].

    coeff_phase_all_list: list of 3D float arrays
        List of phase coefficients output from SYSREM, list length [n_seq] and
        arrays of shape [n_sysrem_iter, n_spec, n_phase].

    coeff_wave_all_list: list of 3D float arrays
        List of wavelength coefficients output from SYSREM, list length [n_seq]
        and arrays of shape [n_sysrem_iter, n_spec, n_px].

    plot_label: str, default: ""
        Filename label for plot, will be saved as 
        sysrem_coeff_[phase/wave]<label>.pdf/png.

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots.
    """
    # Grab dimensions for convenience
    n_seq = len(coeff_phase_all_list)
    (n_sr, n_spec, _) = coeff_phase_all_list[0].shape

    mean_spec_lambdas = np.mean(waves,axis=1)

    # Plot n_spec x n_sysrem_iter panels
    plt.close("all")
    fig_phase, axes_phase = plt.subplots(
        nrows=n_sr, ncols=n_spec, figsize=(25,10),)
    fig_wave, axes_wave = plt.subplots(
        nrows=n_sr, ncols=n_spec, figsize=(25,10),)
    
    for fig in (fig_phase, fig_wave):
        fig.subplots_adjust(
            left=0.03, bottom=0.05, right=0.995, top=0.925, wspace=0.2)

    # Loop over all sequences, spectral seqments, and SYSREM iterations
    for seq_i in range(n_seq):
        for spec_i in range(n_spec):
            for sr_i in range(n_sr):
                # Plot coefficients for this seq/spec/iteration
                axes_phase[sr_i, spec_i].plot(
                    phases_list[seq_i],
                    coeff_phase_all_list[seq_i][sr_i, spec_i],
                    linewidth=0.25,
                    label=labels_per_seq_list[seq_i])
                
                axes_wave[sr_i, spec_i].plot(
                    waves[spec_i],
                    coeff_wave_all_list[seq_i][sr_i, spec_i],
                    linewidth=0.25,
                    label=labels_per_seq_list[seq_i],)
                
                # Adjust axis ticks
                axes_wave[sr_i, spec_i].tick_params(
                    axis='both', which='major', labelsize="xx-small")
                axes_wave[sr_i, spec_i].yaxis.get_offset_text().set_fontsize(
                    "xx-small")
                
                axes_phase[sr_i, spec_i].tick_params(
                    axis='both', which='major', labelsize="xx-small")
                axes_phase[sr_i, spec_i].yaxis.get_offset_text().set_fontsize(
                    "xx-small")

            # Set title for each spectral segment panel of mean wavelength
            axes_wave[0, spec_i].set_title(
                label=r"${:0.0f}\,\mu$m".format(mean_spec_lambdas[spec_i]),
                fontsize="x-small")

    # Plot single legend for each plot
    leg_phase = axes_phase[0, 0].legend(
        loc="upper left",
        bbox_to_anchor=(0, 2.25),
        ncol=n_seq,
        fancybox=True,
        shadow=True,
        fontsize="small",)
    leg_wave = axes_wave[0, 0].legend(
        loc="upper left",
        bbox_to_anchor=(0, 2.25),
        ncol=n_seq,
        fancybox=True,
        shadow=True,
        fontsize="small",)

    # Increase the linewidth in the legend box
    for legobj in leg_phase.legendHandles:
        legobj.set_linewidth(1.5)

    for legobj in leg_wave.legendHandles:
        legobj.set_linewidth(1.5)

    # Set x/y labels for each axis + titles
    fig_phase.text(0.5, 0.01, "Phase", ha='center')
    fig_phase.text(0.01, 0.5, "Coefficient", va='center', rotation='vertical')

    fig_wave.text(0.5, 0.01, "Wave", ha='center')
    fig_wave.text(0.005, 0.5, "Coefficient", va='center', rotation='vertical')

    fig_phase.suptitle("Phase Coefficients")
    fig_wave.suptitle("Wavelength Coefficients")

    # Check save folder and save
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    if plot_label == "":
        plot_fn_1 = os.path.join(plot_folder, "sysrem_coeff_phase")
        plot_fn_2 = os.path.join(plot_folder, "sysrem_coeff_wave")
    else:
        plot_fn_1 = os.path.join(
            plot_folder, "sysrem_coeff_phase_{}".format(plot_label))
        plot_fn_2 = os.path.join(
            plot_folder, "sysrem_coeff_wave_{}".format(plot_label))

    fig_phase.savefig("{}.pdf".format(plot_fn_1))
    fig_phase.savefig("{}.png".format(plot_fn_1), dpi=300)

    fig_wave.savefig("{}.pdf".format(plot_fn_2))
    fig_wave.savefig("{}.png".format(plot_fn_2), dpi=300)


def plot_sysrem_cc_1D(
    cc_rvs,
    cc_values,):
    """Function to plot a 1D plot of the values obtained by cross correlating
    against a grid of SYSREM residuals. The plot has n_rows = n_sysrem_iter,
    and n_cols = 1. By 1D it is meant that the x axis is the RV value used in
    the cross correlation, and the y axis is the cross correlation value, with
    a colour bar representing the phase.

    Parameters
    ----------
    rv_steps: 1D float array
        Vector of RV steps of length [n_rv_step]
    
    cc_values: 3D float array
        3D float array of cross correlation results with shape:
        [n_sysrem_iter, n_phase, n_rv_step]
    """
    # Grab dimensions for convenience
    (n_sysrem_iter, n_phase, n_rv_step) = cc_values.shape

    plt.close("all")
    fig, axes = plt.subplots(
        nrows=n_sysrem_iter,
        ncols=1,
        sharex=True,
        figsize=(18, 6),)
    
    plt.subplots_adjust(
        left=0.05, bottom=0.1, right=0.95, top=0.95, wspace=0.1)

    for sr_iter_i in range(n_sysrem_iter):
        # Grab axis for convenience
        axis = axes[sr_iter_i]

        # Plot each phase as a separate line
        for phase_i in range(n_phase):
            # Get the colour for this phase
            cmap = cm.get_cmap("magma")
            colour = cmap(phase_i/n_phase)

            axis.plot(
                cc_rvs,
                cc_values[sr_iter_i, phase_i],
                c=colour,)

        axis.tick_params(axis='both', which='major', labelsize="x-small")
        axis.xaxis.set_major_locator(plticker.MultipleLocator(base=10))
        axis.xaxis.set_minor_locator(plticker.MultipleLocator(base=5))

        # Only show xticks on the bottom
        if sr_iter_i != n_sysrem_iter-1:
            axis.set_xticks([])

        # Colour bar
        ticks_norm = np.arange(0,1.25,0.25)
        ticks_rescaled = (ticks_norm * n_phase).astype(int)

        divider = make_axes_locatable(axis)
        cb_ax = divider.append_axes("right", size="1%", pad="1%")
        cbar = fig.colorbar(
            cm.ScalarMappable(cmap=cmap), cax=cb_ax, ticks=ticks_norm)
        cbar.ax.set_yticklabels(ticks_rescaled)
        cbar.set_label("Phase")


def plot_2D_CCF_per_spectral_segment(
    cc_rvs,
    ccv_per_spec,
    ccv_combined=None,
    mean_spec_lambdas=None,
    planet_rvs=None,
    in_transit_mask=None,
    fig_size=(18, 6),
    plot_label="",
    plot_folder="plots/",
    xtick_major=40,
    xtick_minor=20,):
    """Function to plot a 2D plot of the values obtained by cross correlating
    against a grid of SYSREM residuals. The plot has n_rows = n_sysrem_iter,
    and n_cols = n_spec (+1 if plotting the combined CCF of all spectral
    segments). By 2D it is meant that the x axis is the RV value used in
    the cross correlation, and the y axis is the phase, with the cross 
    correlation value being represented by a colour bar.

    This function should be used for visualising the CCF for separate spectral
    segments *within* a given observational sequence (either night, or night+
    nodding position).

    Parameters
    ----------
    cc_rvs: 1D float array
        Vector of RV steps of length [n_rv_step]
    
    ccv_per_spec: 4D float array
        4D float array of cross correlation results with shape:
        [n_sysrem_iter, n_phase, n_spec, n_rv_step].

    ccv_combined: 3D float array, default: None
        3D float array of the *combined* cross correlations for each SYSREM
        iteration of shape [n_sysrem_iter, n_phase, n_rv_step].

    mean_spec_lambdas: float array or None, default: None
        Mean values of each spectral segment  shape [n_spec].

    planet_rvs: 1D float array or None, default: None
        Array of planet RVs of shape [n_phase].

    in_transit_mask: bool array
        Boolean array of shape [n_phase] indicating when the planet is
        transiting.
    
    fig_size: float tuple, default: (18, 6)
        Size of the figure.

    plot_label: str, default: ""
        Unique identifier label to add to plot filename.

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots.

    xtick_major, xtick_minor: float, default: 20, 10
        Major and minor ticks for the x axis of the CCF in km/s.
    """
    # If we've been given a set of combined cross-correlation values, 
    # concatenate these to the end of our array so they can be plotted on their
    # own panel.
    if ccv_combined is not None:
        cc_values = np.concatenate(
            (ccv_per_spec, ccv_combined[:,:,None,:]), axis=2)
    else:
        cc_values = ccv_per_spec.copy()

    # Grab dimensions for convenience
    (n_sysrem_iter, n_phase, n_spec, n_rv_step) = cc_values.shape

    plt.close("all")
    fig, axes = plt.subplots(
        nrows=n_sysrem_iter,
        ncols=n_spec,
        sharex=True,
        figsize=fig_size,)
    
    # For consistency, ensure we have a 2D array of axes (even if we don't)
    if n_spec == 1:
        axes = axes[:,None]
    
    plt.subplots_adjust(
        left=0.05, bottom=0.1, right=0.95, top=0.95, wspace=0.1)

    # [x_min, x_max, y_min, y_max]
    extent = [np.min(cc_rvs), np.max(cc_rvs), n_phase, 0]

    #--------------------------------------------------------------------------
    # Plot cross-correlation per spectral segment
    #--------------------------------------------------------------------------
    # Loop over all SYSREM iterations
    for sr_iter_i in range(n_sysrem_iter):

        desc = "Plotting CC values for SYSREM iter #{}".format(sr_iter_i)
        
        # Loop over all spectral segments
        for spec_i in tqdm(range(n_spec), leave=False, desc=desc):
            # Grab axis for convenience
            axis = axes[sr_iter_i, spec_i]

            cmap = axis.imshow(
                X=cc_values[sr_iter_i, :, spec_i],
                aspect="auto",
                interpolation="none",
                extent=extent,)
                #norm=colors.LogNorm(vmin=vmin, vmax=vmax,))
                #norm=colors.PowerNorm(gamma=5))

            axis.tick_params(axis='both', which='major', labelsize="xx-small")
            axis.xaxis.set_major_locator(
                plticker.MultipleLocator(base=xtick_major))
            axis.xaxis.set_minor_locator(
                plticker.MultipleLocator(base=xtick_minor))
            axes[sr_iter_i, spec_i].tick_params(axis='x', labelrotation=45)

            # Only show titles on the top (and if we've been given them)
            if (sr_iter_i == 0 and spec_i == n_spec-1 
                and ccv_combined is not None):
                axis.set_title(
                    label="Combined",
                    fontsize="x-small",
                    color="r",
                    weight="bold",)

            elif sr_iter_i == 0 and mean_spec_lambdas is not None:
                axis.set_title(
                    label=r"${:0.0f}\,\mu$m".format(mean_spec_lambdas[spec_i]),
                    fontsize="x-small")

            # Only show xticks on the bottom
            if sr_iter_i != n_sysrem_iter-1:
                axis.set_xticks([])
            else:
                axis.set_xlabel("RV Shift (km/s)", fontsize="x-small")

            # Only show yticks on the left
            if spec_i != 0:
                    axes[sr_iter_i, spec_i].set_yticks([])
            else:
                axis.set_ylabel("Phase (#)", fontsize="x-small")

            # Plot planet trace if we have it. Note that we only want to show
            # the trace for the phases where the planet is *not* transiting so
            # as to not cover the trace as visible in the CCF itself.
            if planet_rvs is not None and in_transit_mask is not None:
                planet_rvs_masked = planet_rvs.copy()
                planet_rvs_masked[in_transit_mask] = np.nan

                axis.plot(
                    planet_rvs_masked,
                    np.arange(n_phase),
                    linestyle="-",
                    color="white",
                    linewidth=0.5,)

        # Highlight the combined axes
        if ccv_combined is not None:
            axis.spines['top'].set_color("red")
            axis.spines['bottom'].set_color("red")
            axis.spines['left'].set_color("red")
            axis.spines['right'].set_color("red")

            axis.spines['top'].set_linewidth(2)
            axis.spines['bottom'].set_linewidth(2)
            axis.spines['left'].set_linewidth(2)
            axis.spines['right'].set_linewidth(2)

    # Check save folder and save
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    if plot_label == "":
        plot_fn = os.path.join(plot_folder, "ccf_2D_per_spectral_segment")
    else:
        plot_fn = os.path.join(
            plot_folder, "ccf_2D_per_spectral_segment_{}".format(plot_label))

    plt.savefig("{}.pdf".format(plot_fn))
    plt.savefig("{}.png".format(plot_fn), dpi=300)


def plot_2D_CCF_per_seq(
    cc_rvs,
    ccv_per_seq,
    plot_titles,
    planet_rvs_per_seq,
    in_transit_mask_per_seq,
    rv_bcor_median_per_seq,
    fig_size=(18, 10),
    plot_label="",
    plot_folder="plots/",
    xtick_major=40,
    xtick_minor=20,):
    """Function to plot a 2D plot of the values obtained by cross correlating
    against a grid of SYSREM residuals. The plot has n_rows = n_sysrem_iter,
    and n_cols = n_seq. For each panel the x axis is the RV value used in the
    cross correlation, and the y axis is the phase, with the cross correlation
    value being represented by a colour bar. We also overplot the expected
    velocity trace of the planet and the median telluric rest frame position.

    This function should be used for plotting the conbined CCFs for separate
    sequences (either separate nights, or night + nod pos combos), e.g. four
    columns for Night 1 (A), Night 1 (B), Night 2 (A), Night 2 (B).

    Parameters
    ----------
    cc_rvs: 1D float array
        Vector of RV steps of length [n_rv_step]
    
    ccv_per_seq: List of 3D float arrays
        List of 3D float array of cross correlation results for each sequence
        with shape: [n_seq [n_sysrem_iter, n_phase, n_rv_step]].

    plot_titles: str array
        Array of labels to be used as per-seq plot titles of shape [n_seq].

    planet_rvs_per_seq: 1D float array
        Array of planet RVs of shape [n_seq].

    in_transit_mask_per_seq: list of bool arrays
        List of length [n_seq] bool arrays of length [n_phase] indicating which
        phases are in transit or not.

    rv_bcor_median_per_seq: 1D float array
        Array of length [n_seq] containing the median offset between the system
        frame RV and the telluric rest frame, i.e. Vsys + bcor.
    
    fig_size: float tuple, default: (18, 6)
        Size of the figure.

    plot_label: str, default: ""
        Unique identifier label to add to plot filename.

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots.

    xtick_major, xtick_minor: float, default: 20, 10
        Major and minor ticks for the x axis of the CCF in km/s.
    """
    # Grab dimensions for convenience
    n_seq = len(ccv_per_seq)
    (n_sysrem_iter, n_phase, n_rv_step) = ccv_per_seq[0].shape

    plt.close("all")
    fig, axes = plt.subplots(
        nrows=n_sysrem_iter,
        ncols=n_seq,
        sharex=True,
        figsize=fig_size,)
    
    # For consistency, ensure we have a 2D array of axes (even if we don't)
    #if n_spec == 1:
    #    axes = axes[:,None]
    
    plt.subplots_adjust(
        left=0.05, bottom=0.1, right=0.95, top=0.95, wspace=0.1)

    #--------------------------------------------------------------------------
    # Plot cross-correlation per spectral segment
    #--------------------------------------------------------------------------
    # Loop over all sequences
    for seq_i in range(n_seq):
        desc = "Plotting CC values for sequence {}/{}".format(seq_i+1, n_seq)
        
        (n_sysrem_iter, n_phase, n_rv_step) = ccv_per_seq[seq_i].shape

        # [x_min, x_max, y_min, y_max]
        extent = [np.min(cc_rvs), np.max(cc_rvs), n_phase, 0]

        # Loop over all SYSREM iterations
        for sr_iter_i in tqdm(range(n_sysrem_iter), desc=desc, leave=False):
            # Grab axis for convenience
            axis = axes[sr_iter_i, seq_i]

            cmap = axis.imshow(
                X=ccv_per_seq[seq_i][sr_iter_i],
                aspect="auto",
                interpolation="none",
                extent=extent,)

            axis.tick_params(axis='both', which='major', labelsize="xx-small")
            axis.xaxis.set_major_locator(
                plticker.MultipleLocator(base=xtick_major))
            axis.xaxis.set_minor_locator(
                plticker.MultipleLocator(base=xtick_minor))
            axes[sr_iter_i, seq_i].tick_params(axis='x', labelrotation=45)

            # Only show titles on the top (and if we've been given them)
            if sr_iter_i == 0 and plot_titles is not None:
                axis.set_title(
                    label=plot_titles[seq_i],
                    fontsize="x-small")

            # Only show xticks on the bottom
            if sr_iter_i != n_sysrem_iter-1:
                axis.set_xticks([])
            else:
                axis.set_xlabel("RV Shift (km/s)", fontsize="x-small")

            # Only show yticks on the left
            if seq_i != 0:
                    axes[sr_iter_i, seq_i].set_yticks([])
            else:
                axis.set_ylabel("Phase (#)", fontsize="x-small")

            # Plot planet trace if we have it. Note that we only want to show
            # the trace for the phases where the planet is *not* transiting so
            # as to not cover the trace as visible in the CCF itself.
            planet_rvs_masked = planet_rvs_per_seq[seq_i].copy()
            planet_rvs_masked[in_transit_mask_per_seq[seq_i]] = np.nan

            axis.plot(
                planet_rvs_masked,
                np.arange(n_phase),
                linestyle="-",
                color="white",
                linewidth=0.5,)
            
            # Also plot where the location of the telluric rest-frame.
            rv_bcor = np.full(n_phase, rv_bcor_median_per_seq[seq_i])
            rv_bcor[in_transit_mask_per_seq[seq_i]] = np.nan

            axis.plot(
                -1*rv_bcor,
                np.arange(n_phase),
                linestyle="-",
                color="white",
                linewidth=0.5,)


    # Check save folder and save
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    if plot_label == "":
        plot_fn = os.path.join(plot_folder, "ccf_2D_per_seq")
    else:
        plot_fn = os.path.join(
            plot_folder, "ccf_2D_per_seq_{}".format(plot_label))

    plt.savefig("{}.pdf".format(plot_fn))
    plt.savefig("{}.png".format(plot_fn), dpi=300)


def plot_kp_vsys_map(
    vsys_steps,
    Kp_steps,
    Kp_vsys_map_per_spec,
    Kp_vsys_map_combined=None,
    mean_spec_lambdas=None,
    fig_size=(18, 6),
    plot_label="",
    plot_folder="plots/",):
    """Function to plot a 2D plot of Kp-Vsys plots from the results of cross-
    correlation run on SYSREM residuals. The plot has n_rows = n_sysrem_iter,
    and n_cols = 1. By 2D it is meant that the x axis is the RV value used in
    the cross correlation, and the y axis is the Kp value, with the SNR value
    being represented by a colour bar.

    Parameters
    ----------
    vsys_steps: 1D float array
        Vector of RV steps of length [n_rv_step]
    
    Kp_steps: 1D float array
        Vector of Kp steps of length [n_Kp_steps]

    Kp_vsys_map_per_spec: 4D float array
        4D float array of Kp-Vsys maps with shape:
        [n_sysrem_iter, n_spec, n_Kp_steps, n_rv_step]

    Kp_vsys_map_combined: 3D float array, default: None
        3D float array of the *combined* Kp-Vsys map of shape: 
        [n_sysrem_iter, n_Kp_steps, n_rv_step].

    mean_spec_lambdas: float array or None, default: None
        Mean values of each spectral segment  shape [n_spec].
    
    fig_size: float tuple, default: (18, 6)
        Size of the figure.

    plot_label: str, default: ""
        Unique identifier label to add to plot filename.

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots.
    """
    # If we've been given a set of combined Kp-Vsys map, concatenate these to
    # the end of our array so they can be plotted on their own panel.
    if Kp_vsys_map_combined is not None:
        Kp_vsys_map = np.concatenate(
            (Kp_vsys_map_per_spec, Kp_vsys_map_combined[:,None,:,:]), axis=1)
    else:
        Kp_vsys_map = Kp_vsys_map_per_spec.copy()

    # Grab dimensions for convenience
    (n_sysrem_iter, n_spec, n_Kp_steps, n_rv_step) = Kp_vsys_map.shape

    plt.close("all")
    fig, axes = plt.subplots(
        nrows=n_sysrem_iter,
        ncols=n_spec,
        sharex=True,
        figsize=fig_size,)
    
    # For consistency, ensure we have a 2D array of axes (even if we don't)
    if n_spec == 1:
        axes = axes[:,None]
    
    plt.subplots_adjust(
        left=0.05, bottom=0.1, right=0.95, top=0.95, wspace=0.1)

    # [x_min, x_max, y_min, y_max]
    extent = [
        np.min(vsys_steps), np.max(vsys_steps),
        np.min(Kp_steps), np.max(Kp_steps)]

    # Loop over all SYSREM iterations
    for sr_iter_i in range(n_sysrem_iter):
        desc = "Plotting Kp-Vsys map for SYSREM iter #{}".format(sr_iter_i)
        
        # Loop over all spectral segments
        for spec_i in tqdm(range(n_spec), leave=False, desc=desc):
            # Grab axis for convenience
            axis = axes[sr_iter_i, spec_i]

            cmap = axis.imshow(
                X=Kp_vsys_map[sr_iter_i, spec_i],
                aspect="auto",
                interpolation="none",
                extent=extent,
                origin="lower",)

            axis.tick_params(axis='both', which='major', labelsize="x-small")
            axis.xaxis.set_major_locator(plticker.MultipleLocator(base=20))
            axis.xaxis.set_minor_locator(plticker.MultipleLocator(base=10))
            axes[sr_iter_i, spec_i].tick_params(axis='x', labelrotation=45)

            # Only show titles on the top (and if we've been given them)
            if (sr_iter_i == 0 and spec_i == n_spec-1 
                and Kp_vsys_map_combined is not None):
                axis.set_title(
                    label="Combined",
                    fontsize="x-small",
                    color="r",
                    weight="bold",)

            elif sr_iter_i == 0 and mean_spec_lambdas is not None:
                axis.set_title(
                    label=r"${:0.0f}\,\mu$m".format(mean_spec_lambdas[spec_i]),
                    fontsize="x-small")

            # Only show xticks on the bottom
            if sr_iter_i != n_sysrem_iter-1:
                axis.set_xticks([])
            else:
                axis.set_xlabel("RV Shift (km/s)", fontsize="x-small")

            # Only show yticks on the left
            if spec_i != 0:
                    axes[sr_iter_i, spec_i].set_yticks([])
            else:
                axis.set_ylabel(r"$K_P$ (km/s)", fontsize="x-small")

        # Highlight the combined axes
        if Kp_vsys_map_combined is not None:
            axis.spines['top'].set_color("red")
            axis.spines['bottom'].set_color("red")
            axis.spines['left'].set_color("red")
            axis.spines['right'].set_color("red")

            axis.spines['top'].set_linewidth(2)
            axis.spines['bottom'].set_linewidth(2)
            axis.spines['left'].set_linewidth(2)
            axis.spines['right'].set_linewidth(2)

    # Check save folder and save
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    if plot_label == "":
        plot_fn = os.path.join(plot_folder, "kp_vsys_2D")
    else:
        plot_fn = os.path.join(
            plot_folder, "kp_vsys_2D_{}".format(plot_label))

    plt.savefig("{}.pdf".format(plot_fn))
    plt.savefig("{}.png".format(plot_fn), dpi=300)


def plot_regrid_diagnostics_rv(rvs_all, wave_adopt, detectors,):
    """Plot RVs as output from regridding.
    """
    plt.close("all")
    fig, (ax_ord, ax_det) = plt.subplots(2, figsize=(20,6), sharex=True)
    cmap = cm.get_cmap("plasma")

    # Panel #1: RV for each spectral segment
    cmap = cm.get_cmap("plasma")
    for spec_i in range(18):
        wl_mid = ((np.median(wave_adopt[spec_i]) - np.min(wave_adopt)) 
            / (np.max(wave_adopt) - np.min(wave_adopt)))
        colour = cmap(wl_mid)
        ax_ord.plot(
            rvs_all[:,spec_i],
            color=colour,
            label="{:0.0f} nm".format(int(np.median(wave_adopt[spec_i]))))
    ax_ord.set_ylabel("RV (km/s)")

    # Add legend (but sort)
    # https://stackoverflow.com/questions/22263807/
    # how-is-order-of-items-in-matplotlib-legend-determined
    handles, labels = ax_ord.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax_ord.legend(handles, labels, loc="upper center", ncol=6)
    plt.tight_layout()

    # Panel #2: median RV for each detector
    for det_i in [1,2,3]:
        det_mask = detectors == det_i
        rvs = np.median(rvs_all[:, det_mask], axis=1)
        ax_det.plot(rvs, label="Detector {:0.0f}".format(det_i))
    ax_det.set_xlabel("Phase")
    ax_det.set_ylabel("RV (km/s)")
    ax_det.legend(loc="lower center", ncol=3)
    
    ax_det.set_xlim(-0.5, len(rvs_all)+0.5)
    ax_det.xaxis.set_major_locator(plticker.MultipleLocator(base=5))
    ax_det.xaxis.set_minor_locator(plticker.MultipleLocator(base=1))

    plt.tight_layout()
    plt.savefig("plots/regrid_diagnostic_rv.pdf")
    plt.savefig("plots/regrid_diagnostic_rv.png", dpi=300)


def plot_regrid_diagnostics_img(fluxes, detectors, wave_adopt, sigma_upper=4,):
    """Plot aligned spectra as output from regridding.
    """
    (n_phase, n_spec, n_px) = fluxes.shape

    plt.close("all")
    fig, axes = plt.subplots(3, sharey=True, figsize=(20,10))

    for det_i in range(3):
        det_mask = detectors == det_i+1
        fluxes_det = fluxes[:, det_mask].reshape(n_phase, n_spec//3 * n_px)
        axes[det_i].imshow(
            fluxes_det, aspect="auto", interpolation="none")
        axes[det_i].set_title("Detector #{:0.0f}".format(det_i+1))
        axes[det_i].set_ylabel("Phase")

        wave_mids = np.median(wave_adopt[det_mask], axis=1).astype(int)
        for wl_i, wl in enumerate(wave_mids):
            px = n_px//2 + wl_i * n_px
            phase = n_phase // 2
            txt = axes[det_i].text(
                x=px,
                y=phase,
                s="{:0.0f} nm".format(wl),
                horizontalalignment="center",)
            txt.set_bbox(dict(facecolor="white", alpha=0.4, edgecolor="white"))
    
    plt.tight_layout()
    plt.savefig("plots/regrid_diagnostic_img.pdf")
    plt.savefig("plots/regrid_diagnostic_img.png", dpi=300)


def plot_combined_kp_vsys_map_as_snr(
    vsys_steps,
    Kp_steps,
    Kp_vsys_maps_snr,
    plot_title,
    Kp_expected=None,
    max_snr_values=None,
    vsys_at_max_snr=None,
    kp_at_max_snr=None,
    plot_suptitle="",
    fig_size=(5, 10),
    plot_label="",
    tick_spacing=2.0,
    plot_folder="plots/",):
    """Function to plot a 2D plot of Kp-Vsys plots from the results of cross-
    correlation run on SYSREM residuals. The plot has n_rows = n_sysrem_iter,
    and n_cols = n_maps (e.g. night 1, night 2). By 2D it is meant that the 
    x axis is the RV value used in the cross correlation, and the y axis is the
    Kp value, with the SNR value being represented by a colour bar.

    Providing Kp_expected plots a white crosshair at the expected Kp value; and 
    providing max_snr_values, vsys_at_max_snr, and kp_at_max_snr plots a red
    crosshair at the observed maximum SNR value.

    Parameters
    ----------
    vsys_steps: 1D float array
        Vector of RV steps of length [n_rv_step]
    
    Kp_steps: 1D float array
        Vector of Kp steps of length [n_Kp_steps]

    Kp_vsys_maps_snr: 3D or 4D float array
        Float array of the *combined* Kp-Vsys map/s with shape: 
        [n_sysrem_iter, n_Kp_steps, n_rv_step] 
        or [n_map, n_sysrem_iter, n_Kp_steps, n_rv_step].

    plot_title: str or list of str
        Title/s for each Kp-Vsys map.

    Kp_expected: float, default: None
        Expected Kp value of the planet in km/s from the literature.

    max_snr_values, vsys_at_max_snr, kp_at_max_snr: 2D float array or None
        Maximum SNR, and corresponding coordinates in velocity space, of shape
        [n_map, n_sysrem_iter] (as computed from sysrem.calc_Kp_vsys_map_snr).

    plot_suptitle: str, default: ""
        Super title for the figure.

    fig_size: float tuple, default: (6, 10)
        Size of the figure, will have n_sysrem_iter rows and a single column.

    plot_label: str, default: ""
        Unique identifier label to add to plot filename.

    tick_spacing: float, default: 2.0
        Colourbar tick spacing in units of SNR.

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots. 
    """
    # Grab dimensions for convenience (this depends on how many maps we have)
    if len(Kp_vsys_maps_snr.shape) == 3:
        (n_sysrem_iter, n_Kp_steps, n_rv_step) = Kp_vsys_maps_snr.shape
        Kp_vsys_maps_snr = Kp_vsys_maps_snr[None,:,:,:]
        n_maps = 1
        plot_title = [plot_title]
    else:
        (n_maps, n_sysrem_iter, n_Kp_steps, n_rv_step) = Kp_vsys_maps_snr.shape
        fig_size = (fig_size[0]*n_maps, fig_size[1])

    plt.close("all")
    fig, axes = plt.subplots(
        ncols=n_maps,
        nrows=n_sysrem_iter,
        sharex=True,
        figsize=fig_size,)
    
    # Force axes object to be 2D
    if len(axes.shape) != 2:
        axes = axes[:, None]
    
    plt.subplots_adjust(
        left=0.05, bottom=0.05, right=0.98, top=0.94, wspace=0.02, hspace=0.1)

    # [x_min, x_max, y_min, y_max]
    extent = [
        np.min(vsys_steps), np.max(vsys_steps),
        np.min(Kp_steps), np.max(Kp_steps)]

    # Loop over all maps
    for map_i in range(n_maps):
        # Loop over all SYSREM iterations
        for sr_iter_i in range(n_sysrem_iter):
            # Grab SNR for convenience
            snr = Kp_vsys_maps_snr[map_i, sr_iter_i]

            # Grab axis for convenience
            axis = axes[sr_iter_i, map_i]

            cmap = axis.imshow(
                X=snr,
                aspect="auto",
                interpolation="none",
                extent=extent,
                origin="lower",)
            
            # Sort out our colourbar
            cb = fig.colorbar(cmap, ax=axis)

            min_snr = np.nanmin(snr)
            max_snr = np.nanmax(snr)
            delta_snr = max_snr - min_snr

            ticks = np.arange(
                np.floor(min_snr), np.ceil(max_snr), tick_spacing)

            cb.set_ticks(ticks)
            cb.set_ticklabels(ticks)

            cb.ax.minorticks_on()
            cb.ax.tick_params(labelsize="small", rotation=0)

            # Axis ticks
            axis.tick_params(axis='both', which='major', labelsize="x-small")
            axis.xaxis.set_major_locator(plticker.MultipleLocator(base=20))
            axis.xaxis.set_minor_locator(plticker.MultipleLocator(base=10))
            axes[sr_iter_i, map_i].tick_params(axis='x', labelrotation=45)

            axis.yaxis.set_major_locator(plticker.MultipleLocator(base=50))
            axis.yaxis.set_minor_locator(plticker.MultipleLocator(base=25))

            # Only show colour bar label on right
            if map_i+1 == n_maps:
                cb.set_label("SNR", fontsize="large")
            
            # Only show yticks on left
            if map_i != 0:
                axis.set_yticks([])

            # Only show xticks on the bottom, and title on the top
            if sr_iter_i == 0:
                axis.set_title(
                    plot_title[map_i], fontdict={'fontsize':"small"})
                axis.set_xticks([])
            if sr_iter_i != n_sysrem_iter-1:
                axis.set_xticks([])
            else:
                axis.set_xlabel("RV Shift (km/s)", fontsize="x-small")

            #------------------------------------------------------------------
            # Plot 'crosshair' for expected and observed Kp/Vsys max SNR
            #------------------------------------------------------------------
            # Plot expected Kp (from the literature)
            if Kp_expected is not None:
                axis.vlines(
                    x=0,
                    ymin=Kp_steps[0],
                    ymax=Kp_steps[-1],
                    linestyles="dashed",
                    colors="w",
                    linewidth=0.5,)
                
                axis.hlines(
                    y=Kp_expected,
                    xmin=vsys_steps[0],
                    xmax=vsys_steps[-1],
                    linestyles="dashed",
                    colors="w",
                    linewidth=0.5,)

            # Plot observed Kp/Vsys max
            if vsys_at_max_snr is not None and kp_at_max_snr is not None:
                # Plot *observed* Kp
                axis.vlines(
                    x=vsys_at_max_snr[map_i, sr_iter_i],
                    ymin=Kp_steps[0],
                    ymax=Kp_steps[-1],
                    linestyles="dashed",
                    colors="r",
                    linewidth=0.5,)
                
                axis.hlines(
                    y=kp_at_max_snr[map_i, sr_iter_i],
                    xmin=vsys_steps[0],
                    xmax=vsys_steps[-1],
                    linestyles="dashed",
                    colors="r",
                    linewidth=0.5,)
                
                # Add a text box summarising the maximum
                snr_txt = \
                    r"SNR~{:0.1f} [${:0.1f}, {:0.1f}$ km$\,$s$^{{-1}}$]"
                vsys_max = vsys_at_max_snr[map_i, sr_iter_i]
                kp_max = kp_at_max_snr[map_i, sr_iter_i]
                snr_max = max_snr_values[map_i, sr_iter_i]

                axis_txt = axis.text(
                    x=vsys_steps[-1]*0.6,
                    y=Kp_steps[-1]*0.1,
                    s=snr_txt.format(
                        snr_max, vsys_max, kp_max),
                    fontsize="xx-small",
                    color="r",
                    horizontalalignment="center",)

                axis_txt.set_bbox(
                    dict(facecolor="w", alpha=0.2, edgecolor="w"))

    fig.suptitle(plot_suptitle, fontsize="medium")

    # Check save folder and save
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    if plot_label == "":
        plot_fn = os.path.join(plot_folder, "kp_vsys_snr")
    else:
        plot_fn = os.path.join(
            plot_folder, "kp_vsys_snr_{}".format(plot_label))

    plt.savefig("{}.pdf".format(plot_fn))
    plt.savefig("{}.png".format(plot_fn), dpi=300)


def visualise_templates(
    waves,
    sr_settings,
    syst_info, 
    species_to_cc=["H2O", "CO", "NH3", "H2S"],
    y_min=0.99,
    y_max=1.005,
    figsize=(20,6)):
    """Plots comparison of continuum normalised *single species* exoplanet
    atmosphere template spectra with overlaid observed wavelength scale.

    TODO: due to the way that tu.prepare_cc_template is currently implemented,
    we have to repeatedly update/restore the contents of ss

    Parameters
    ----------
    waves: 2D float array
        Observed wavelength scale of shape [n_spec, n_px]

    sr_settings: YAMLSettings Object
        SYSREM settings object.

    syst_info: pandas DataFrame
        Dataframe containing information about the star+planet system.

    species_to_cc: str list, default: ["H2O", "CO", "NH3", "H2S"]
        List of molecular species.
        
    y_min: float, default: 0.99
        Minimum limit on y axis.

    y_max: float, default: 1.005
        Maximum limit on y axis.

    figsize: float tuple, default: (20,6)
        Figure size.
    """
    plt.close("all")
    fig, axis = plt.subplots(figsize=figsize,)

    all_templates = []

    # Loop over all species, continuum normalise, smooth, and plot.
    for species in species_to_cc:
        # Update species list, but keep a record of old value to restore later
        old_species = sr_settings.species_to_cc
        sr_settings.species_to_cc = [species]

        wave_template, spectrum_template = tu.prepare_cc_template(
            cc_settings=sr_settings,
            syst_info=syst_info,
            templ_wl_nm_bounds=(19000,25000),
            continuum_resolving_power=300,)
        
        # Restore old list of species
        sr_settings.species_to_cc = old_species
        
        all_templates.append(all_templates)

        axis.plot(
            wave_template,
            spectrum_template,
            linewidth=0.5,
            label=species,
            alpha=0.5)
   
   # Make lines in legend thicker for visibility
    leg = plt.legend(ncol=len(species_to_cc))

    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.5)

    # Now plot rectangular patches corresponding to CRIRES wavelength scale
    delta_y = y_max - y_min

    for spec_i in range(len(waves)):
        w_min = waves[spec_i].min()
        w_max = waves[spec_i].max()
        delta_wave = w_max - w_min
        rect = Rectangle(
            xy=(w_min, y_min),
            width=delta_wave,
            height=delta_y,
            facecolor='grey',
            alpha=0.1)
        axis.add_patch(rect)

    axis.set_xlim(wave_template[0], wave_template[-1])
    axis.set_ylim(y_min, y_max)
    plt.tight_layout()


def plot_autocorrelation(
    wave_obs,
    autocorr_rvs,
    autocorr_2D,
    autocorr_comb,
    plot_label,
    plot_title,
    figsize=(18, 4),
    plot_folder="plots/",):
    """For each spectral segment and given an appropriate systemic velocity, 
    plots the autocorrelation of the exoplanet atmosphere spectral template
    over a range of RVs. The goal is to check for aliases away from 0 km/s that
    could manifest as false positives in the Kp-Vsys map space.

    Takes as input the output of sysrem.compute_template_autocorrelation.

    Parameters
    ----------
    wave_obs: 2D float array
        Observed wavelength vector of shape [n_spec, n_px]

    autocorr_rvs: 1D float array
        Wavelengths over which the autocorrelation was computed, [n_rvs].
    
    autocorr_2D: 2D float array
        Autocorrelation for each spectral segment of shape [n_spec, n_rvs].

    autocorr_comb: 2D float array
        Autocorrelation combining all spectral segments of shape [n_rvs].

    plot_label: str
        Label to be added to saved plot filename.

    plot_title: str
        Title for the plot.
    
    figsize: float tuple, default: (18, 4)
        Size of the figure.

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots.
    """
    (n_spec, n_px) = wave_obs.shape

    plt.close("all")
    fig, axes = plt.subplots(1,n_spec+1, figsize=figsize)

    # Plot autocorrelation for each spectral segment
    for spec_i in range(n_spec):
        axes[spec_i].plot(autocorr_rvs, autocorr_2D[spec_i], linewidth=0.5)
        axes[spec_i].vlines(
            x=0,
            ymin=np.min(autocorr_2D[spec_i]),
            ymax=1, linestyles="dashed",
            linewidth=0.5, color="k")
        axes[spec_i].set_title("{:0.0f} nm".format(np.mean(wave_obs[spec_i])))
        axes[spec_i].set_xlabel("RV (km/s)")
        axes[spec_i].set_yticks([])
        axes[spec_i].xaxis.set_major_locator(plticker.MultipleLocator(base=20))
        axes[spec_i].xaxis.set_minor_locator(plticker.MultipleLocator(base=10))

    # And plot the autocorrelation after combining all spectral segments
    axes[-1].plot(autocorr_rvs, autocorr_comb, linewidth=0.5)
    axes[-1].vlines(
        x=0,
        ymin=np.min(autocorr_comb),
        ymax=1,
        linestyles="dashed",
        linewidth=0.5,
        color="k")
    axes[-1].set_yticks([])
    axes[-1].set_xlabel("RV (km/s)")
    axes[-1].set_title("Combined")
    axes[-1].xaxis.set_major_locator(plticker.MultipleLocator(base=20))
    axes[-1].xaxis.set_minor_locator(plticker.MultipleLocator(base=10))

    # Other plot settings + save
    fig.suptitle(plot_title, fontsize="medium")

    plt.tight_layout()

    # Check save folder and save
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    plot_fn = os.path.join(plot_folder, "autocorr_{}".format(plot_label))

    plt.savefig("{}.pdf".format(plot_fn))
    plt.savefig("{}.png".format(plot_fn), dpi=300)


def plot_in_and_out_of_transit_histograms(
    ccv_per_spec,
    ccv_combined,
    in_transit,
    mean_spec_lambdas,
    fig_size=(6, 8),
    plot_label="",
    plot_title="",
    plot_folder="plots/",
    n_bins=75,):
    """Function to plot a grid of histograms comparing the cross-correlation
    values for in-transit vs out-of-transit phases.

    Parameters
    ----------
    ccv_per_spec: 4D float array or None
        4D float array of cross correlation results with shape:
        [n_sysrem_iter, n_phase, n_spec, n_rv_step].

    ccv_combined: 3D float array or None
        3D float array of the *combined* cross correlations for each SYSREM
        iteration of shape [n_sysrem_iter, n_phase, n_rv_step].

    in_transit: bool array
        Boolean array with True for phases in-transit, and False for phases not
        in transit.

    mean_spec_lambdas: float array or None, default: None
        Mean values of each spectral segment  shape [n_spec].

    figsize: float tuple, default: (6, 8)
        Size of the figure.

    plot_label: str
        Label to be added to saved plot filename.

    plot_title: str
        Title for the plot.

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots.

    n_bins: int, default:75
        Number of bins for the histogram.
    """
    if ccv_per_spec is None and ccv_combined is not None:
        cc_values = ccv_combined[:,:,None,:]

    # If we've been given a set of combined cross-correlation values, 
    # concatenate these to the end of our array so they can be plotted on their
    # own panel.
    elif ccv_combined is not None:
        cc_values = np.concatenate(
            (ccv_per_spec, ccv_combined[:,:,None,:]), axis=2)
        
    else:
        cc_values = ccv_per_spec.copy()

    # Grab dimensions for convenience
    (n_sysrem_iter, n_phase, n_spec, n_rv_step) = cc_values.shape

    plt.close("all")
    fig, axes = plt.subplots(
        nrows=n_sysrem_iter,
        ncols=n_spec,
        figsize=fig_size,)
    
    # For consistency, ensure we have a 2D array of axes (even if we don't)
    if n_spec == 1:
        axes = axes[:,None]
    
    plt.subplots_adjust(
        left=0.05, bottom=0.05, right=0.95, top=0.925, wspace=0.1, hspace=0.3)

    #--------------------------------------------------------------------------
    # Plot cross-correlation per spectral segment
    #--------------------------------------------------------------------------
    # Loop over all SYSREM iterations
    for sr_iter_i in range(n_sysrem_iter):

        desc = "Plotting hist for SYSREM iter #{}".format(sr_iter_i)
        
        # Loop over all spectral segments
        for spec_i in tqdm(range(n_spec), leave=False, desc=desc):
            # Grab axis for convenience
            axis = axes[sr_iter_i, spec_i]

            ccv_it = cc_values[sr_iter_i, in_transit, spec_i].flatten()
            ccv_oot = cc_values[sr_iter_i, ~in_transit, spec_i].flatten()

            _ = axis.hist(
                ccv_it,
                bins=n_bins,
                density=True,
                label="In Transit",
                color="r",
                alpha=0.5)
            
            _ = axis.hist(
                ccv_oot,
                bins=n_bins,
                density=True,
                label="Out of Transit",
                color="k",
                alpha=0.5,)

            axis.tick_params(labelsize="xx-small")
            axis.set_yticks([])

            offset = axis.xaxis.get_offset_text()
            offset.set_size("xx-small")

            # Only show titles on the top (and if we've been given them)
            if (sr_iter_i == 0 and spec_i == n_spec-1 
                and ccv_combined is not None):
                axis.set_title(
                    label="Combined",
                    fontsize="x-small",
                    color="r",
                    weight="bold",)

            elif sr_iter_i == 0 and mean_spec_lambdas is not None:
                axis.set_title(
                    label=r"${:0.0f}\,\mu$m".format(mean_spec_lambdas[spec_i]),
                    fontsize="x-small")

    axes[0,0].legend(
        loc="upper center",
        ncol=2,
        fontsize="xx-small",
        bbox_to_anchor=(0.5, 1.05),)

    fig.suptitle(plot_title, fontsize="medium")

    # ---------
    # Check save folder and save
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    if plot_label == "":
        plot_fn = os.path.join(plot_folder, "hist_comp")
    else:
        plot_fn = os.path.join(
            plot_folder, "hist_comp_{}".format(plot_label))

    plt.savefig("{}.pdf".format(plot_fn))
    plt.savefig("{}.png".format(plot_fn), dpi=300)