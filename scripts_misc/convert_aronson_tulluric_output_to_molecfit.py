"""
"""
import os
import sys
import numpy as np
import pandas as pd
import subprocess
from astropy.io import fits
import matplotlib.pyplot as plt
import luciferase.spectra as lspec

sys.path.insert(1, "/home/arains/code/luciferase/")

root_path = "data_reduction/wasp127_tellurics"

# This is the 25th observation for the nodding A position, which is roughly in
# the centre of the set of observations
shell_fits_file =  os.path.join(root_path,"cr2res_obs_nodding_extractedA.fits")

# Output files from Aronson method
airmass_file = os.path.join(root_path, "airmass.dat")
ccd_data_paths = [
    os.path.join(root_path, "telluric_ccd1.dat"),
    os.path.join(root_path, "telluric_ccd2.dat"),
    os.path.join(root_path, "telluric_ccd3.dat"),]

# Read in files
airmasses = np.loadtxt(airmass_file)
airmass = airmasses[25]

ccd_names = ["CHIP1.INT1", "CHIP2.INT1", "CHIP3.INT1",]
ORDER_MIN = 2
ORDER_MAX = 7
PX_TO_PAD = 10

default_sigma = 0.01

wl_cols = ["{:02.0f}_01_WL".format(i) for i in range(ORDER_MAX,ORDER_MIN-1,-1)]
tau_cols = ["tau_order_{}".format(i) for i in range(ORDER_MAX,ORDER_MIN-1, -1)]

cols = []
for wl, tau in zip(wl_cols, tau_cols):
    cols.append(wl)
    cols.append(tau)

# Duplicate fits file
telluric_fits_path = shell_fits_file.replace(".fits", "_telluric.fits")
bashCommand = "cp {} {}".format(shell_fits_file, telluric_fits_path)
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

# Populate new fits file
with fits.open(telluric_fits_path, mode="update") as tulluric_fits:
    # Loop over each CCD
    for ccd_name, data_path in zip(ccd_names, ccd_data_paths):
        # Load in data for this chip
        data = pd.read_csv(data_path, delim_whitespace=True, names=cols,)

        # And loop over each order
        for wl_col, tau_col in zip(wl_cols, tau_cols):
            print(ccd_name, wl_col)
            # Write wavelength data (probably unnecessary?). We need to pad
            # 10 px onto either end of the array.
            wls = data[wl_col].values.astype(float)

            delta_wl = np.nanmedian(np.abs(wls[1:] - wls[:-1]))
            wls_pad_start = np.arange(
                wls[0]-delta_wl, wls[0]-delta_wl*(PX_TO_PAD+1), -delta_wl)[::-1]
            wls_pad_end = np.arange(
                wls[-1]+delta_wl, wls[-1]+delta_wl*(PX_TO_PAD+1), delta_wl)
            
            wls = np.concatenate((wls_pad_start, wls, wls_pad_end))
            tulluric_fits[ccd_name].data[wl_col] = wls

            # Write spectrum data
            spec_col = wl_col.replace("WL", "SPEC")

            taus = data[tau_col].values.astype(float)

            # Hack cause of correcting Nik's file format wrong
            if ccd_name == "CHIP3.INT1":
                if tau_col == "tau_order_2" or tau_col == "tau_order_7":
                    mask = np.argwhere(taus == 0.0)[:,0]
                    taus[mask] = 9.9999

            transmission = np.concatenate((
                np.full(PX_TO_PAD, np.nan),
                np.exp(-airmass*taus),
                np.full(PX_TO_PAD, np.nan),))
            tulluric_fits[ccd_name].data[spec_col] = transmission

             # Write sigma data, setting to a uniform default
            err_col = wl_col.replace("WL", "ERR")
            sigma = np.full(transmission.shape, default_sigma)
            tulluric_fits[ccd_name].data[err_col] = sigma

            plt.plot(wls, transmission, linewidth=0.4)

    # Change airmass to 1 in fits file (AZ also needed?)
    tulluric_fits[0].header["HIERARCH ESO TEL AIRM START"] = 1.0
    tulluric_fits[0].header["HIERARCH ESO TEL AIRM END"] = 1.0

# Load in and plot to check
ob = lspec.initialise_observation_from_crires_fits(telluric_fits_path,)