"""
Plotting functions associated with our transit modelling.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import transit.utils as tu
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as plticker
from tqdm import tqdm
from astropy.stats import sigma_clip

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
    ref_scale_vector=None,):
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
    
    # Share x axes
    ax_flux.get_shared_x_axes().join(ax_flux, ax_tell, ax_trans)

    # Plot each spectral segment
    for spec_i in range(waves.shape[0]):
        ax_flux.plot(
            waves[spec_i],
            fluxes[spec_i],
            linewidth=0.4,
            color="r",
            alpha=0.8,)
        
        ax_tell.plot(
            waves[spec_i],
            np.exp(-telluric_tau[spec_i]),
            linewidth=0.4,
            color="b",
            alpha=0.8,)
        
        ax_trans.plot(
            waves[spec_i],
            planet_trans[spec_i],
            linewidth=0.4,
            color="g",
            alpha=0.8,)
        
        ax_scale.plot(
            np.arange(len(scale_vector)),
            scale_vector,
            marker="o",
            linewidth=0.4,
            color="c",
            alpha=0.8,)
        
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
                linewidth=0.4,
                color="k",
                alpha=0.8,)
            
        if (ref_telluric_tau is not None
            and telluric_tau.shape == ref_telluric_tau.shape):
            ax_tell.plot(
                waves[spec_i],
                np.exp(-ref_telluric_tau[spec_i]),
                linewidth=0.4,
                color="k",
                alpha=0.8,)
        
        if (ref_planet_trans is not None
            and planet_trans.shape == ref_planet_trans.shape):
            ax_trans.plot(
                waves[spec_i],
                ref_planet_trans[spec_i],
                linewidth=0.4,
                color="k",
                alpha=0.8,)
            
        if (ref_scale_vector is not None
            and scale_vector.shape == ref_scale_vector.shape):
            ax_scale.plot(
                np.arange(len(scale_vector)),
                scale_vector,
                marker="o",
                linewidth=0.4,
                color="c",
                alpha=0.8,)

    fig.suptitle("Transit #{:0.0f}".format(transit_num+1))
    ax_flux.set_title("Stellar flux")
    ax_tell.set_title("Telluric Transmission")
    ax_trans.set_title("Planet Transmission")
    ax_trans.set_xlabel(r"Wavelength (${\rm \AA}$)")
    ax_scale.set_xlabel("Epoch #")
    ax_scale.set_ylim(0.5, 1.5)
    plt.tight_layout()

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

        # Save


def plot_sysrem_residuals(
    waves,
    resid,
    fig_size=(18,6),
    do_sigma_clip=False,
    sigma_upper=5,
    sigma_lower=5,
    max_iterations=5,):
    """Function to plot a grid of residuals as output from SYSREM. The grid has
    n_rows = n_sysrem_iter, and n_cols = n_spec.

    Parameters
    ----------
    wave: 2D float array
        Wavelength vector of shape [n_spec, n_px]
    
    resid: 4D float array
        3D float array of residuals with shape 
        [n_sysrem_iter, n_phase, n_spec, n_px]

    fig_size: float array, default: (18,6)
        Shape of the figure.
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
        left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.1)

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
                #vmin=vmin,
                #vmax=vmax,
                extent=extent,
                #norm=colors.LogNorm(vmin=vmin, vmax=vmax,))
                #norm=colors.PowerNorm(gamma=2),
                cmap="plasma",)
                #cmap="PRGn")

            axis.tick_params(axis='both', which='major', labelsize="x-small")
            axis.xaxis.set_major_locator(plticker.MultipleLocator(base=4))
            axis.xaxis.set_minor_locator(plticker.MultipleLocator(base=2))
            axes[sr_iter_i, spec_i].tick_params(axis='x', labelrotation=45)

            # Only show xticks on the bottom
            if sr_iter_i != n_sysrem_iter-1:
                axis.set_xticks([])

            # Only show Y labels on the left
            if spec_i == 0:
                axis.set_ylabel("Iter #{}".format(sr_iter_i))
            
            # Only show yticks on the left
            else:
                axis.set_yticks([])
            
            # Show just one colour bar per iteration
            #ticks_norm = np.arange(0,1.25,0.25)
            #ticks_rescaled = (ticks_norm * (vmax-vmin) + vmin).astype(int)

            #fig.subplots_adjust(right=0.8)
            divider = make_axes_locatable(axis)
            cb_ax = divider.append_axes("right", size="5%", pad="1%")

            #cb_ax = fig.add_axes([0.9, 0.05, 0.025, 0.9])  # [L, B, W, H]
            cbar = fig.colorbar(cmap, cax=cb_ax, aspect=5)
            #cbar.ax.set_yticklabels(ticks_rescaled)
            #cbar.set_label("Mean Wavelength of Spectral Segment")


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


def plot_sysrem_cc_2D(
    cc_rvs,
    ccv_per_spec,
    ccv_combined=None,
    mean_spec_lambdas=None,
    planet_rvs=None,
    fig_size=(18, 6),
    plot_label="",):
    """Function to plot a 2D plot of the values obtained by cross correlating
    against a grid of SYSREM residuals. The plot has n_rows = n_sysrem_iter,
    and n_cols = 1. By 2D it is meant that the x axis is the RV value used in
    the cross correlation, and the y axis is the phase, with the cross 
    correlation value being represented by a colour bar.

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
    
    fig_size: float tuple, default: (18, 6)
        Size of the figure.

    plot_label: str, default: ""
        Unique identifier label to add to plot filename.
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

            axis.tick_params(axis='both', which='major', labelsize="x-small")
            axis.xaxis.set_major_locator(plticker.MultipleLocator(base=10))
            axis.xaxis.set_minor_locator(plticker.MultipleLocator(base=5))
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

            # Plot planet trace if we have it
            if planet_rvs is not None:
                axis.plot(planet_rvs, np.arange(n_phase), "--", color="white",)

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

    if plot_label == "":
        plt.savefig("plots/crosscorr_2D.pdf")
        plt.savefig("plots/crosscorr_2D.png", dpi=300)
    else:
        plt.savefig("plots/crosscorr_2D_{}.pdf".format(plot_label))
        plt.savefig("plots/crosscorr_2D_{}.png".format(plot_label), dpi=300)


def plot_kp_vsys_map(
    cc_rvs,
    Kp_steps,
    Kp_vsys_map_per_spec,
    Kp_vsys_map_combined=None,
    mean_spec_lambdas=None,
    fig_size=(18, 6),
    plot_label="",):
    """Function to plot a 2D plot of Kp-Vsys plots from the results of cross-
    correlation run on SYSREM residuals. The plot has n_rows = n_sysrem_iter,
    and n_cols = 1. By 2D it is meant that the x axis is the RV value used in
    the cross correlation, and the y axis is the Kp value, with the SNR value
    being represented by a colour bar.

    Parameters
    ----------
    cc_rvs: 1D float array
        Vector of RV steps of length [n_rv_step]
    
    Kp_steps: 1D float array
        Vector of Kp steps of length [n_Kp_steps]

    Kp_vsys_map_per_spec: 4D float array
        4D float array of Kp-Vsys maps with shape:
        [n_sysrem_iter, n_spec, n_Kp_steps, n_rv_step]

    Kp_vsys_map_combined: 3D float array, default: None
        3D floar array of the *combined* Kp-Vsys map of shape: 
        [n_sysrem_iter, n_Kp_steps, n_rv_step].

    mean_spec_lambdas: float array or None, default: None
        Mean values of each spectral segment  shape [n_spec].
    
    fig_size: float tuple, default: (18, 6)
        Size of the figure.

    plot_label: str, default: ""
        Unique identifier label to add to plot filename.
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
        np.min(cc_rvs), np.max(cc_rvs), np.min(Kp_steps), np.max(Kp_steps)]

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

    if plot_label == "":
        plt.savefig("plots/kp_vsys_2D.pdf")
        plt.savefig("plots/kp_vsys_2D.png", dpi=300)
    else:
        plt.savefig("plots/kp_vsys_2D_{}.pdf".format(plot_label))
        plt.savefig("plots/kp_vsys_2D_{}.png".format(plot_label), dpi=300)


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