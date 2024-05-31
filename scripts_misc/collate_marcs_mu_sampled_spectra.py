"""Script to import set of MARCS spectra sampled at different mu viewing angles
across the disk. All spectra should be in the same folder, and are expected to
share a wavelength scale. We expect [wavelength, spec_cont_norm, spec_flux]
columns for the imported spectra.
"""
import glob
import numpy as np
from tqdm import tqdm
from astropy.io import fits

# Save file name
save_path = "templates/WASP107_MARCS_4420g4.61z+0.00m0.683mu49.fits"

# Import files
fits_path = "templates/syntspec/*.sp"
all_files = glob.glob(fits_path)
all_files.sort()

# Check dimensions
data = np.loadtxt(all_files[0]).T
n_wave = len(data[0])
n_mu = len(all_files)

# Setup data structures
waves_all = np.empty((n_mu, n_wave))
spec_cont_norm_all = np.empty((n_mu, n_wave))
spec_fluxes_all = np.empty((n_mu, n_wave))
mu_angles = np.empty((n_mu))

# Loop over all files
for file_i, file in enumerate(tqdm(all_files, desc="Loading spectra")):
    # Load in file
    data = np.loadtxt(file).T

    # Save arrays
    waves_all[file_i] = data[0]
    spec_cont_norm_all[file_i] = data[1]
    spec_fluxes_all[file_i] = data[2]

    # Pull the mu value out of the filename
    # e.g. 4420g4.61z+0.00m0.683t00.851170.sp
    mu_angles[file_i] = float(file.split("/")[-1].split("t")[-1][:-3])

# All done, some checks
assert np.sum(np.argsort(mu_angles) - np.arange(n_mu)) == 0   # mus are sorted
assert np.sum(np.diff(waves_all, axis=0)) == 0                # waves are equal

# All wavelength vectors are the same
waves = waves_all[0]

# Now we can save all this to a new fits file. Format:
#  HDU 0: wl vector
#  HDU 1: continuum normalised spectra
#  HDU 2: unnormalised spectra
#  HDU 3: sampled mu angles
# Intialise HDU List
hdu = fits.HDUList()

# HDU 0: wavelength scale
wave_img =  fits.PrimaryHDU(waves)
wave_img.header["EXTNAME"] = ("WAVES", "Wavelength scale")
hdu.append(wave_img)

spec_norm_img =  fits.PrimaryHDU(spec_cont_norm_all)
spec_norm_img.header["EXTNAME"] = (
    "SPEC_CONT_NORM", "MARCS continuum normalised spectra.")
hdu.append(spec_norm_img)

# HDU 2: unnormalised spectra
spec_flux_img =  fits.PrimaryHDU(spec_fluxes_all)
spec_flux_img.header["EXTNAME"] = (
    "SPEC_FLUXES", "MARCS unnormalised spectral fluxes.")
hdu.append(spec_flux_img)

# HDU 3: unnormalised spectra
mu_img =  fits.PrimaryHDU(mu_angles)
mu_img.header["EXTNAME"] = (
    "MU_ANGLES", "mu angles used by MARCS to sample across the stellar disc.")
hdu.append(mu_img)

# Done, save
hdu.writeto(save_path, overwrite=True)