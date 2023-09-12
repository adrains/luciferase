"""This script is for working with the output of the inverse method as run on
*simulated* data (i.e. we exactly know the input components).
"""
import numpy as np
import transit.utils as tu
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import astropy.constants as const
import matplotlib.ticker as plticker
from scipy.interpolate import interp1d
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Load in settings file
# -----------------------------------------------------------------------------
yaml_settings_file = "scripts_transit/transit_model_settings.yml"
ms = tu.load_yaml_settings(yaml_settings_file)

# -----------------------------------------------------------------------------
# Load in spectra
# -----------------------------------------------------------------------------
# Load in observed data
waves, fluxes_list, sigmas_list, det, orders, transit_info_list, syst_info = \
    tu.load_transit_info_from_fits(ms.save_path, ms.fn_label, ms.n_transit)

# Load in recovered aronson spectrum
model_obs, model_flux, model_tau, model_trans, model_scale, model_mask = \
    tu.load_simulated_model_results_from_fits(
        fits_load_dir=ms.save_path,
        label=ms.fn_label,
        n_transit=ms.n_transit,)

# Create a mask based on telluric regions
ignore_mask = 10**-np.median(model_tau, axis=0) < 0.1

# Apply
model_trans[ignore_mask] = np.nan

# -----------------------------------------------------------------------------
# Load in templates
# -----------------------------------------------------------------------------
# Load in templates
templ_wave, templ_spec_all, templ_info = \
    tu.load_transmission_templates_from_fits(fits_file=ms.template_fits)

# Convert to nm
templ_wave /= 10

# Clip edges
templ_spec_all = templ_spec_all[:,10:-10]
templ_wave = templ_wave[10:-10]

molecules = templ_info.columns.values

# -----------------------------------------------------------------------------
# Fitting planet transmission spectra
# -----------------------------------------------------------------------------
(n_phase, n_spec, n_px) = model_obs.shape

CC_RV_STEP = 0.1
CC_RV_LIMS = (-200,200)

MINOR_CC_Y_TICK = 250
MAJOR_CC_Y_TICK = 500

CC_Y_LIMS = (-500, 1500)

cross_corr_rvs = np.arange(CC_RV_LIMS[0], CC_RV_LIMS[1]+CC_RV_STEP, CC_RV_STEP)

plt.close("all")
fig, axes = plt.subplots(len(templ_spec_all), sharex=True, figsize=(16, 10))

wl_min = np.min(waves)
wl_max = np.max(waves)

# Match each template against each spectral segment
for templ_i, templ_spec in enumerate(templ_spec_all):
    
    curr_mols = molecules[templ_info.iloc[templ_i].values]
    
    molecule_str = ("{}+"*len(curr_mols))[:-1].format(*tuple(curr_mols))

    txt = "Template #{}: {}".format(templ_i, molecule_str)

    best_rvs = np.full(n_spec, np.nan)

    cross_corr_all = []

    for spec_i in tqdm(range(n_spec), desc=txt, leave=False):
        # Construct ID for spectral segment
        label = r"Det {}, Ord, {}, $\overline{{\lambda}}$".format(
            det[spec_i], orders[spec_i], np.mean(waves[spec_i]))

        # Normalise transmission spectrum
        norm_model_trans = \
            model_trans[spec_i] - np.nanmedian(model_trans[spec_i])
        norm_model_trans /= np.nanstd(model_trans[spec_i])

        # Construct an interpolator for the template
        temp_interp = interp1d(
            x=templ_wave,
            y=templ_spec,
            bounds_error=False,
            fill_value=np.nan,)
        
        # Initialise an empty cross correlation array
        cross_corr_vals = np.full_like(cross_corr_rvs, np.nan)

        for rv_i, rv in enumerate(cross_corr_rvs):
            # Doppler shift for new wavelength scale
            wave_rv_shift = waves[spec_i] * (1- rv/(const.c.si.value/1000))

            # Interpolate to wavelength scale
            tspec_rv_shift = temp_interp(wave_rv_shift)

            # Normalise template transmission spectrum
            norm_templ_trans = tspec_rv_shift - np.nanmedian(tspec_rv_shift)
            norm_templ_trans /= np.nanstd(norm_templ_trans)

            # Cross correlate
            cross_corr_vals[rv_i] = np.nansum(
                norm_model_trans * norm_templ_trans)

        best_rvs[spec_i] = cross_corr_rvs[np.argmax(cross_corr_vals)]

        # Plot cross-correlation results for this template
        cmap = cm.get_cmap("magma")
        colour = cmap((np.mean(waves[spec_i]-wl_min)/(wl_max-wl_min)))

        axes[templ_i].set_title(molecule_str)
        axes[templ_i].plot(
            cross_corr_rvs, cross_corr_vals, label=label, color=colour)

        axes[templ_i].yaxis.set_major_locator(
            plticker.MultipleLocator(base=MAJOR_CC_Y_TICK))
        axes[templ_i].yaxis.set_minor_locator(
            plticker.MultipleLocator(base=MINOR_CC_Y_TICK))

        axes[templ_i].set_ylim(*CC_Y_LIMS)

        # Only show x ticks on lowest panel
        if templ_i != len(templ_spec_all)-1:
            axes[templ_i].set_xticks([])

        # Store all cross correlation results
        cross_corr_all.append(cross_corr_vals)
    
# Global colourbar
ticks_norm = np.arange(0,1.25,0.25)
ticks_rescaled = (ticks_norm * (wl_max-wl_min) + wl_min).astype(int)

fig.subplots_adjust(right=0.8)
cb_ax = fig.add_axes([0.9, 0.05, 0.025, 0.9])  # [L, B, W, H]
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), cax=cb_ax, ticks=ticks_norm)
cbar.ax.set_yticklabels(ticks_rescaled)
cbar.set_label("Mean Wavelength of Spectral Segment")

# Remaining setup
axes[-1].set_xlim(CC_RV_LIMS[0], CC_RV_LIMS[1])

axes[-1].set_xlabel(r"RV Shift (km s${^{-1}}$)")

fig.subplots_adjust(
    hspace=0.5, wspace=0.001, right=0.88, left=0.05, top=0.95, bottom=0.05)

# Save plot
plt.savefig("plots/cross_corr_recovery.pdf")