"""
Script to generate SOFs and a reduction shell script to splice the A and B
master reduced nodded spectra together.

Run as:
    python make_nodding_sof_splice_master.py [folder]

where [folder] is the directory of the master observations, and the script is
called from the main directory.
"""
import sys
import numpy as np
import glob
from astropy.io import fits
import os
import subprocess

# Get current working directory
cwd = os.getcwd()

# Get which nAB set we want to reduce
master_folder = sys.argv[1]

# Check to see if we have a blaze file
blaze_file = os.path.join(cwd, "cr2res_cal_flat_Open_blaze.fits")

if not os.path.isfile(blaze_file):
    raise FileNotFoundError("No blaze file, aborting.")

# Before we start looping, archive the old reduce.sh script if it exists
splice_script = "splice_master.sh"

if os.path.isfile(splice_script):
    bashCommand = "mv {} {}.old".format(splice_script, splice_script)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

# And make a new splice script
with open(splice_script, "w") as rs:
    rs.write("#!/bin/bash\n")

cmd = "chmod +x {}".format(splice_script)
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

# Write one SOF file for each of A and B
for nod in ("A", "B", "_combined"):
    sof_file = os.path.join(cwd, master_folder, "splice{}.sof".format(nod))

    # Write the SOF file along with calibration files at the end
    with open(sof_file, 'w') as sof:
        # First write the science file
        sof.writelines(
            ("cr2res_obs_nodding_extracted{}.fits".format(nod) 
                + "\tUTIL_EXTRACT_1D\n"))

        # Now write the blaze file
        sof.writelines("{}\tCAL_FLAT_EXTRACT_1D\n".format(blaze_file))

        # Write the trace wave file
        sof.writelines(
            "/home/tom/pCOMM/cr2re-calib/K2148_tw.fits\tUTIL_WAVE_TW\n")

    # And finally write the a file containing esorex reduction commands
    with open(splice_script, 'a') as ww:
        if nod == "A":
            ww.write("cd {}\n".format(master_folder))
        esorex_cmd = ("esorex cr2res_util_splice {}\n".format(sof_file))
        ww.write(esorex_cmd)