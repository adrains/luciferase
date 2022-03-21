"""Script to prepare SOF files for reduction of raw calibrations. Note that
since we take the precomputed wavelength file rather than computing it,
ourselves, and since we don't need a reduced dark for nodding observations, we
only need to process the raw files associated with the flats.

Note that currently this script assumes that all files in the directory were
taken with the same instrument settings (i.e. grating).

Run as
------
python make_calibration_sofs.py [wl_setting]

where [wl_setting] is the grating setting, e.g. K2148.

Output
------
1x sof for darks
1x sofs for flats
1x reduce_raw_cals.sh shell script
"""
import sys
import numpy as np
import os
import glob
import subprocess
from astropy.io import fits

# Get grating setting
wl_setting = sys.argv[1]

TRACE_WAVE = "/home/tom/pCOMM/cr2re-calib/{}_tw.fits".format(wl_setting)
DETLIN_COEFF = "/home/tom/pCOMM/cr2res_cal_detlin_coeffs.fits"

# Currently an issue with detlin adding noise due to quality of data, so 
# advisable to leave out.
use_detlin = False

# Get current working directory
cwd = os.getcwd()

# Get a list of just the calibration files
fits_fns = glob.glob("CRIRE.*.fits")
fits_fns.sort()

# We only care about calibration files, so grab these and their exposure times.
flat_fns = []
flat_exps = []

dark_fns = []
dark_exps = []
dark_ndits = []

for fn in fits_fns:
    if fits.getval(fn, "OBJECT") == "FLAT":
        flat_fns.append(fn)
        flat_exps.append(fits.getval(fn, "HIERARCH ESO DET SEQ1 DIT"))
        wl_setting = fits.getval(fn, "HIERARCH ESO INS WLEN ID")

    elif fits.getval(fn, "OBJECT") == "DARK":
        dark_fns.append(fn)
        dark_exps.append(fits.getval(fn, "HIERARCH ESO DET SEQ1 DIT"))

        # Get NDIT, assuming it is the same for all files.
        dark_ndits.append(fits.getval(fn, "HIERARCH ESO DET NDIT"))

# There should only be a single flat exposure
assert len(set(flat_exps)) == 1
flat_exp = flat_exps[0]

dark_fns = np.array(dark_fns)
dark_exps = np.array(dark_exps)
dark_ndits = np.array(dark_ndits)

# Raise an exception if we don't have an matched set of exposure times
if flat_exp not in dark_exps:
    raise Exception("Flat and dark exposure times not the same!")

flat_dark_ndits = dark_ndits[dark_exps == flat_exp]

# There should only be a single value of NDIT.....if there isn't things are 
# more complicated.
assert len(set(flat_dark_ndits)) == 1
ndit = flat_dark_ndits[0]

# Before we start looping, archive the old reduce.sh script if it exists
shell_script = "reduce_cals.sh"

if os.path.isfile(shell_script):
    bashCommand = "mv {} {}.old".format(shell_script, shell_script)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

# And make a new splice script
with open(shell_script, "w") as rs:
    rs.write("#!/bin/bash\n")

cmd = "chmod +x {}".format(shell_script)
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

# Make SOF file for darks. The esorex recipe can do all our darks at once, so
# we may as well add them all.
dark_sof = "dark.sof"

with open(dark_sof, 'w') as sof:
    for dark in dark_fns:
        sof.writelines("{}\tDARK\n".format(os.path.join(cwd, dark)))

# Write SOF for flats.
flat_sof = "flat.sof"

# Format the flat exposure time so it writes properly. Keep up to 5 decimal
# places, or cast to integer if they're the same.
if flat_exp == int(flat_exp):
    flat_exp_str = "{:0.0f}".format(flat_exp)
else:
    flat_exp_str = "{:0.5f}".format(flat_exp)

# Since we've reduced all the darks, they'll be named separately for each exp,
#  grating, and NDIT setting.
dark_bpm = "cr2res_cal_dark_{}_{}x{}_bpm.fits".format(
    wl_setting, flat_exp_str, ndit)
dark_master = "cr2res_cal_dark_{}_{}x{}_master.fits".format(
    wl_setting, flat_exp_str, ndit)

with open(flat_sof, 'w') as sof:
    for flat in flat_fns:
        sof.writelines("{}\tFLAT\n".format(os.path.join(cwd, flat)))

    sof.writelines("{}\tCAL_DARK_BPM\n".format(os.path.join(cwd, dark_bpm)))
    sof.writelines("{}\tCAL_DARK_MASTER\n".format(os.path.join(cwd, dark_master)))

    sof.writelines("{}\tUTIL_WAVE_TW\n".format(TRACE_WAVE))

    if use_detlin:
        sof.writelines("{}\tCAL_DETLIN_COEFFS\n".format(DETLIN_COEFF))

# And finally write the file containing esorex reduction commands
with open(shell_script, 'a') as ww:
    esorex_cmd = ("esorex cr2res_cal_dark {}\n".format(dark_sof))
    ww.write(esorex_cmd)

    esorex_cmd = ("esorex cr2res_cal_flat {}\n".format(flat_sof))
    ww.write(esorex_cmd)

