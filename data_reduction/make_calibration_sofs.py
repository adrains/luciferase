"""Script to prepare SOF files for reduction of raw calibrations. Note that
since we take the precomputed wavelength file rather than computing it,
ourselves, and since we don't need a reduced dark for nodding observations, we
only need to process the raw files associated with the flats.

Run as
------
python make_calibration_sofs.py

Output
------
1x sof for darks
1x sofs for flats
1x reduce_raw_cals.sh shell script
"""
import numpy as np
import os
import glob
import subprocess
from astropy.io import fits

TRACE_WAVE = "/home/tom/pCOMM/cr2re-calib/K2148_tw.fits"
DETLIN_COEFF = "/home/tom/pCOMM/cr2res_cal_detlin_coeffs.fits"

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

for fn in fits_fns:
    if fits.getval(fn, "OBJECT") == "FLAT":
        flat_fns.append(fn)
        flat_exps.append(fits.getval(fn, "HIERARCH ESO DET SEQ1 DIT"))

    elif fits.getval(fn, "OBJECT") == "DARK":
        dark_fns.append(fn)
        dark_exps.append(fits.getval(fn, "HIERARCH ESO DET SEQ1 DIT"))

# There should only be a single flat exposure
assert len(set(flat_exps)) == 1
flat_exp = flat_exps[0]

dark_fns = np.array(dark_fns)
dark_exps = np.array(dark_exps)

# Raise an exception if we don't have an matched set of exposure times
if flat_exp not in dark_exps:
    raise Exception("Flat and dark exposure times not the same!")

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

# First do dark
darks = dark_fns[dark_exps == flat_exp]

dark_sof = "dark.sof"

with open(dark_sof, 'w') as sof:
    for dark in darks:
        sof.writelines("{}\tDARK\n".format(dark))

# Now do flats
flat_sof = "flat.sof"

with open(flat_sof, 'w') as sof:
    for flat in flat_fns:
        sof.writelines("{}\tFLAT\n".format(flat))

    sof.writelines("cr2res_cal_dark_bpm.fits\tCAL_DARK_BPM\n")
    sof.writelines("cr2res_cal_dark_master.fits\tCAL_DARK_MASTER\n")

    sof.writelines("{}\tUTIL_WAVE_TW\n".format(TRACE_WAVE))
    sof.writelines("{}\tCAL_DETLIN_COEFFS\n".format(DETLIN_COEFF))

# And finally write the file containing esorex reduction commands
with open(shell_script, 'a') as ww:
    esorex_cmd = ("esorex cr2res_cal_dark {}\n".format(dark_sof))
    ww.write(esorex_cmd)

    esorex_cmd = ("esorex cr2res_cal_flat {}\n".format(flat_sof))
    ww.write(esorex_cmd)

