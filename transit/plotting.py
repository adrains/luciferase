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
 

def plot_all_input_spectra(waves, fluxes_all, jds):
    """Function to plot spectra at all phases and colour code by timestep.

    Parameters
    ----------
    wave_observed: 2D float array
        Wavelength scale of shape [n_spec, n_wave].

    fluxes_model: 3D float array
        Flux array corresponding to wavelengths of shape 
        [n_phase, n_spec, n_wave].
    
    jds: 1D float array
        Array of timesteps.
    """
    plt.close("all")
    fig, ax = plt.subplots(figsize=(15, 5))

    mid_jd = np.mean(jds)
    hours = (jds - mid_jd) * 24

    # Setup colour bar
    norm = plt.Normalize(np.min(hours), np.max(hours))
    cmap = plt.get_cmap("plasma")
    colours = cmap(norm(hours))

    cmap = cm.ScalarMappable(norm=norm, cmap=cm.plasma)

    for phase_i in range(64):
        for wave_i in range(len(waves)):
            ax.plot(
                waves[wave_i],
                fluxes_all[phase_i, wave_i],
                color=colours[phase_i],
                linewidth=0.5,
                label=phase_i,)

    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux (counts)")

    cb = fig.colorbar(cmap)
    cb.set_label("Time from mid-point (hr)")
    plt.tight_layout()