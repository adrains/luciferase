"""
Script to generate SOFs and a reduction shell script to splice reduced nodded 
spectra together for time series data.

Run as
------

python make_nodding_sof_splice_nAB.py [nAB]

where [nAB] is the number of A/B pairs combined.
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
nAB = int(sys.argv[1])

# Check to see if we have a blaze file
blaze_file = os.path.join(cwd, "cr2res_cal_flat_Open_blaze.fits")

if not os.path.isfile(blaze_file):
    raise FileNotFoundError("No blaze file, aborting.")

# Get a list of the nodding AB set folders that we want to reduce
nodding_folders = glob.glob(os.path.join(cwd, "{}xAB_*/".format(nAB)))
nodding_folders.sort()

# Input checking
if nAB < 1:
    raise ValueError("Warning, nAB must be >= 1.")
elif len(nodding_folders) < 1:
    raise ValueError("Warning, no nodding folders found!")

# Before we start looping, archive the old reduce.sh script if it exists
splice_script = "splice_{}xAB.sh".format(nAB)

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

# Write one SOF per AB set and corresponding subdirectory
for nod_set_i, nod_folder in enumerate(nodding_folders):
    # Get the nod folder
    nod_subdir = nod_folder.split(os.sep)[-2]

    # Check to see we have our nodding reduced science data
    sci_fits = os.path.join(
        nod_folder, "cr2res_obs_nodding_extracted_combined.fits")

    if not os.path.isfile(sci_fits):
        print("No combined science file for nod {}, continuing.".format(
            nod_subdir))
        continue
    
    date_sci = fits.getval(sci_fits, "DATE-OBS").split("T")[0]
    date_blaze = fits.getval(blaze_file, "DATE-OBS").split("T")[0]
    
    print("Generating SOF for {}. Science date: {}, blaze date: {}.".format(
        nod_subdir, date_sci, date_blaze))

    # Write one SOF file for each of A and B
    for nod in ("A", "B"):
        sof_file = os.path.join(nod_folder, "{}_splice{}.sof".format(
            nod_subdir, nod))

        # Write the SOF file along with calibration files at the end
        with open(sof_file, 'w') as sof:
            # First write the combines science file
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
                ww.write("cd {}\n".format(nod_folder))
            esorex_cmd = ("esorex cr2res_util_splice {}\n".format(sof_file))
            ww.write(esorex_cmd)