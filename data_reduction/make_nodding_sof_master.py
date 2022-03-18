"""Script to prepare SOF files for reducing master nodding reductions. Files
are saved to a subdirectory with the grating setting name.

Note that currently this script assumes that all files in the directory were
taken with the same instrument settings (i.e. grating).

Run as
------
python make_nodding_sof_master.py [setting] [NDIT]

Where [setting] is the grating setting as listed in the fits headers.

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

# Get NDIT if provided, otherwise assume NDIT=1
if len(sys.argv) > 2:
    ndit = sys.argv[2]
else:
    ndit = 1

# Get current working directory
cwd = os.getcwd()

bpm_file = "cr2res_cal_dark_{}_1.42705x{}_bpm.fits".format(
    wl_setting, ndit)

DARK_BPM = os.path.join(cwd, bpm_file)
MASTER_FLAT = os.path.join(cwd, "cr2res_cal_flat_Open_master_flat.fits")
TRACE_WAVE = "/home/tom/pCOMM/cr2re-calib/{}_tw.fits".format(wl_setting)

# No point continuing if our calibration files don't exist
if (not os.path.isfile(DARK_BPM) or not os.path.isfile(MASTER_FLAT)
    or not os.path.isfile(TRACE_WAVE)):
    raise Exception("Calibration files not found.")

# Get a list of just the observations at our grating 
fits_fns = glob.glob("CRIRE.*.fits")
fits_fns.sort()

obs_fns = []

for fn in fits_fns:
    obs_cat = fits.getval(fn, "HIERARCH ESO DPR CATG")

    if obs_cat == "SCIENCE":
        obs_setting = fits.getval(fn, "HIERARCH ESO INS WLEN ID")

        if obs_setting == wl_setting:
            obs_fns.append(fn)

# Raise an exception if we don't have any files
if len(obs_fns) < 1:
    raise Exception("No files found!")

# TODO raise error if we don't have an even number of raw frames
pass

# Check the new subdirector exists
if not os.path.isdir(wl_setting):
    os.mkdir(wl_setting)

# Before we start looping, archive the old reduce.sh script if it exists
shell_script = os.path.join(
    cwd, wl_setting, "reduce_master_{}.sh".format(wl_setting))

if os.path.isfile(shell_script):
    bashCommand = "mv {} {}.old".format(shell_script, shell_script)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

# And make a new script
with open(shell_script, "w") as rs:
    rs.write("#!/bin/bash\n")

cmd = "chmod +x {}".format(shell_script)
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

obs_sof = os.path.join(cwd, wl_setting, "{}.sof".format(wl_setting))

with open(obs_sof, 'w') as sof:
    # First write all science files
    for obs_fn in obs_fns:
        obs_fn_path = os.path.join(cwd, obs_fn)
        sof.writelines("{}\tOBS_NODDING_OTHER\n".format(obs_fn_path))

    # Then write bad pixel mask, master flat, and wave trace wave
    sof.writelines("{}\tCAL_DARK_BPM\n".format(DARK_BPM))
    sof.writelines("{}\tCAL_FLAT_MASTER\n".format(MASTER_FLAT))
    sof.writelines("{}\tUTIL_WAVE_TW\n".format(TRACE_WAVE))

# And finally write the a file containing esorex reduction commands
with open(shell_script, 'a') as ww:
    ww.write("cd {}\n".format(os.path.join(cwd, wl_setting)))
    esorex_cmd = ('esorex cr2res_obs_nodding --extract_swath_width=800'
                    + ' --extract_oversample=12 --extract_height=30 '
                    + obs_sof + '\n')
    ww.write(esorex_cmd)