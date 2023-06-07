"""
Plotting functions associated with our transit modelling.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import transit.utils as tu
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


def plot_component_spectra(
    waves,
    fluxes,
    telluric_tau,
    planet_trans,
    scale_vector,
    transit_num,
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

    scale_vector: 1D ffloat array
        Adoped scale/slit losses vector of shape [n_phase].

    ref_fluxes, ref_telluric_tau, ref_planet_trans, ref_scale_vector: float 
    array or None
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


def plot_epoch_model_comp(
    waves,
    obs_spec,
    model,
    fluxes,
    telluric_tau,
    planet_trans,
    transit_info,
    ref_fluxes=None,
    ref_telluric_tau=None,
    ref_planet_trans=None,):
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

    ref_fluxes, ref_telluric_tau, ref_planet_trans: float array or None
        Reference spectra to plot against fluxes, tau, and trans.
    """
    # pull out variables for convenience
    (n_phase, n_spec, n_wave) = obs_spec.shape

    # loop over all epochs
    for phase_i, transit_epoch in transit_info.iterrows():
        # TODO Temporary HACK
        if phase_i != 0:
            continue

        # initialise subplot
        fig, (ax_spec, ax_flux, ax_tell, ax_trans) = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(15, 10),
            sharex=True,)
        
        # Setup lower panel for residuals
        plt.setp(ax_spec.get_xticklabels(), visible=False)
        divider = make_axes_locatable(ax_spec)
        res_ax = divider.append_axes("bottom", size="30%", pad=0)
        fig.add_axes(res_ax, sharex=ax_spec)
        ax_spec.get_shared_x_axes().join(ax_spec, res_ax)

        # Grab velocities for this phase
        gamma = transit_epoch["gamma"]
        beta = transit_epoch["beta"]
        delta = transit_epoch["delta"]

        for spec_i in range(n_spec):
            # Initialise derivatives
            flux_2 = tu.bezier_init(x=waves[spec_i], y=fluxes[spec_i],)
            tau_2 = tu.bezier_init(x=waves[spec_i], y=telluric_tau[spec_i],)
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
            tellurics = np.exp(-telluric_tau[spec_i])

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
                    np.exp(-ref_telluric_tau[spec_i]),
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

        # Finish setup
        ax_spec.set_title("Input vs Model Spectrum")
        ax_flux.set_title("Stellar flux")
        ax_tell.set_title("Telluric Transmission")
        ax_trans.set_title("Planet Transmission")
        ax_trans.set_xlabel(r"Wavelength (${\rm \AA}$)")

        ax_spec.legend(loc="lower center", ncol=2)
        ax_flux.legend(loc="lower center", ncol=2)
        ax_tell.legend(loc="lower center", ncol=2)
        ax_trans.legend(loc="lower center", ncol=2)

        plt.tight_layout()

        # Save