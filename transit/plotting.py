"""
Plotting functions associated with our transit modelling.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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

        axes[transit_i].set_ylabel("Flux (counts)")

        cb = fig.colorbar(cmap, ax=axes[transit_i])
        cb.set_label("Time from mid-point (hr)")

    axes[-1].set_xlabel("Wavelength")
    plt.tight_layout()


def plot_component_spectra(waves, fluxes, telluric_tau, planet_trans,):
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
    """
    # Intialise subplots
    fig, (ax_flux, ax_tell, ax_trans) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(15, 5),
        sharex=True,)

    # Plot each spectral segment
    for spec_i in range(waves.shape[0]):
        ax_flux.plot(
            waves[spec_i],
            fluxes[spec_i],
            linewidth=0.4,
            color="r")
        
        ax_tell.plot(
            waves[spec_i],
            np.exp(-telluric_tau[spec_i]),
            linewidth=0.4,
            color="b")
        
        ax_trans.plot(
            waves[spec_i],
            planet_trans[spec_i],
            linewidth=0.4,
            color="g")

    ax_flux.set_title("Stellar flux")
    ax_tell.set_title("Telluric Transmission")
    ax_trans.set_title("Planet Transmission")
    ax_trans.set_xlabel(r"Wavelength (${\rm \AA}$)")
    plt.tight_layout()