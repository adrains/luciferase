"""Script to combine multiple HARPS spectra together.
"""
import os
import sys
import glob
import numpy as np
from astropy.io import fits
import astropy.constants as const
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import Normalize

sys.path.insert(1, "/home/arains/code/luciferase/")

SNR_THRESHOLD = 5
FIG_SIZE = (18,6)
N_TO_PLOT = 2
COLOURBAR = "SNR"

# Grab files
root =  "/Users/arains/Dropbox/code/luciferase/data/spectra/"
#folder = "wasp76_harps"    # F7V
folder = "wasp127_harps"    # G5V
folder = "wasp107_harps"    # K7V
#folder = "L98-59_harps"     # M3V
spec_path = os.path.join(root, folder, "*.fits")
spectra_files = glob.glob(spec_path)
spectra_files.sort()

star = folder.replace("_harps", "")

waves = []
fluxes = []
sigmas = []
snrs = []
bcors = []
rvs = []
mjds = []

for spec_i, spec_file in enumerate(spectra_files):
    with fits.open(spec_file) as ff:
        # Check SNR
        snr = ff[0].header["SNR"]
        bcor = ff[0].header["HIERARCH ESO DRS BERV"]
        rv = ff[0].header["HIERARCH ESO DRS CAL TH LAMP OFFSET"]
        mjd = ff[0].header["MJD-OBS"]

        if snr < SNR_THRESHOLD:
            continue

        wave = ff[1].data["WAVE"][0]
        flux = ff[1].data["FLUX"][0]
        sigma = ff[1].data["ERR"][0]

        waves.append(wave)
        fluxes.append(flux)
        sigmas.append(sigma)
        snrs.append(snr)
        bcors.append(bcor)
        rvs.append(rv)
        mjds.append(mjd)

mjd_min = np.nanmin(mjds)
mjd_max = np.nanmax(mjds)
snr_min = np.nanmin(snrs)
snr_max = np.nanmax(snrs)

# Now adopt first wavelength scale
wave_0 = waves[0]
fluxes_0 = []

plt.close("all")
fig, ax = plt.subplots(figsize=FIG_SIZE)

cmap = cm.get_cmap("viridis")

for spec_i in range(len(fluxes)):
    # Create interpolator
    interp_flux = interp1d(
        x=waves[spec_i],
        y=fluxes[spec_i],
        kind="linear",
        bounds_error=False,
        assume_sorted=True)

    # Shift RV
    #flux_rf = interp_flux(
    #    wave_0 * (1+(rvs[spec_i]-bcors[spec_i])/(const.c.si.value/1000)))

    # Interpolate flux to common wavelength scale
    flux_0 = interp_flux(wave_0)
    fluxes_0.append(flux_0)
    
    # Plot (potentially limiting the number of lines for performance reasons)
    if spec_i % N_TO_PLOT == 0:
        if COLOURBAR == "SNR":
            colour = cmap((snrs[spec_i]-snr_min)/(snr_max-snr_min))
        elif COLOURBAR == "MJD":
            colour = cmap((mjds[spec_i]-mjd_min)/(mjd_max-mjd_min))

        ax.plot(
            waves[spec_i],
            fluxes[spec_i]/np.nanmedian(fluxes[spec_i]),
            linewidth=0.2,
            color=colour,
            alpha=0.9,)

# Stack and combine the resulting fluxes
fluxes_0 = np.vstack(fluxes_0)
flux_comb = np.sum(fluxes_0, axis=0)
snr_comb = np.nanmedian(flux_comb) / np.sqrt(np.nanmedian(flux_comb))

# Plot the combined spectrum
ax.plot(wave_0, flux_comb/np.nanmedian(flux_comb),linewidth=0.4, c="black")

# Plot colour bar
if COLOURBAR == "SNR":
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(snr_min, snr_max))
    cb = fig.colorbar(sm, ax=ax, pad=0.01)
    cb.set_label("SNR", labelpad=10)

elif COLOURBAR == "MJD":
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(mjd_min, mjd_max))
    cb = fig.colorbar(sm, ax=ax, pad=0.01)
    cb.set_label("MJD", labelpad=10)

# Setup rest of plot
ax.set_ylim(0,2.5)
ax.set_xlim(wave_0[0]-5, wave_0[-1]+5)
ax.set_xlabel("Wavelength")
ax.set_ylabel("Flux")
title = "{}, {:0.0f} spectra, Combined SNR~{:0.0f}".format(
    star, len(snrs), snr_comb)
ax.set_title(title)
#plt.tight_layout()

# Save plot
plot_save_path = os.path.join(
    root, folder, "{}_snr_{:0.0f}_nspec_{:0.0f}.pdf".format(
        star, snr_comb, len(snrs)))
plt.savefig(plot_save_path, bbox_inches='tight')

# Print summary
print("Median SNR of {:0.0f} inidividual exposures: {:0.0f}".format(
    len(snrs), np.nanmedian(snrs)))
print("Combined SNR: {:0.0f}".format(snr_comb))

# Save combined spectrum using the first fits file again
with fits.open(spectra_files[0]) as hdul:
    hdul[1].data["FLUX"][0] = flux_comb
    hdul[1].data["ERR"][0] = flux_comb / np.sqrt(flux_comb)

    fits_save_path = os.path.join(
        root, folder, "{}_snr_{:0.0f}_nspec_{:0.0f}.fits".format(
            star, snr_comb, len(snrs)))
    hdul.writeto(fits_save_path, overwrite=True)