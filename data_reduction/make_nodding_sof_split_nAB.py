"""Originally written by Alexis Lavail, modified by Adam Rains.

Script to generate a series of SOF files for CRIRES+ reductions in nodding mode
for time-series observations.

Run as:
    python3 nodding-make_sofs.py [nAB]

Where you replace [nAB] with the number of nodding observations you would like
to reduce together. E.G. nAB=1 will combine the nearest AB/BA pair in time, 
whereas nAB=2 will combine the nearest AABB/BBAA or ABAB/BABA pair in time. The
exact format will depend on the specifics of how the nodding observations were
conducted.

Creates a nod.sof SOF file and a reduce_[nAB]xAB.sh script that reduces your 
nodding observations for each nodding set, and places each in a separate 
subfolder for set/timestep with format [nAB]xAB_[ii]/ where ii is the nodding
set.
"""
from functools import reduce
import sys
import numpy as np
import glob
from astropy.io import fits
import os
import subprocess

# Get current working directory
cwd = os.getcwd()

# Get the number of frames to combine
nAB = int(sys.argv[1])

# Get the list of science fits files
files_all = np.sort(glob.glob('CR*fits'))

# Go through the list and remove any raw calibration files
sci_files = []

for sci_file in files_all:
    if fits.getval(sci_file, "HIERARCH ESO DPR CATG") == "SCIENCE":
        sci_files.append(sci_file)

n_sci = len(sci_files)

# Input checking
if nAB < 1:
    raise ValueError("Warning, nAB must be >= 1.")
elif (n_sci // 2) % nAB != 0:
    raise ValueError("Warning, n_sci/2 frames must be divisible by nAB.")

# Check calibration file exists
calib_file = "calib.sof"
if not os.path.isfile(calib_file):
    raise FileNotFoundError("No calib.sof file, aborting.")

# Sort the science observations into either A or B frames
sci_A, sci_B = [], []
for sci_i in range(n_sci):
    hd = fits.open(sci_files[sci_i])
    if hd[0].header['HIERARCH ESO SEQ NODPOS'] == "A":
        sci_A.append(sci_files[sci_i])
        
    elif hd[0].header['HIERARCH ESO SEQ NODPOS'] == "B":
        sci_B.append(sci_files[sci_i])

# Abort if we don't have an equal number of A and B frames
if len(sci_A) != len(sci_B):
    raise Exception("Warning! # A frames =/= # B frames, aborting.")

# Before we start looping, archive the old reduce.sh script if it exists
reduce_script = "reduce_{}xAB.sh".format(nAB)

if os.path.isfile(reduce_script):
    bashCommand = "mv {} {}.old".format(reduce_script, reduce_script)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

# And make a new reduction script
with open(reduce_script, "w") as rs:
    rs.write("#!/bin/bash\n")

cmd = "chmod +x {}".format(reduce_script)
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

print("{} science files found, #A nod frames == #B nod frames.".format(n_sci))
print("Generating SOFs for {} timesteps with nAB={} per step.\n".format(
    n_sci//nAB//2, nAB))

# Construct SOF files and subdirs for each set of nAB nodding observations
for nod_set_i in range(n_sci//nAB//2):
    # Make the new subdirectory if it doesn't already exist
    dir = "{}xAB_{:02.0f}".format(nAB, nod_set_i)
    
    if not os.path.isdir(dir):
        try:
            os.mkdir(dir)
        except OSError as error:
            print(error)
    
    # Prepare the new SOF filepath
    sof_file = os.path.join(
        cwd, dir, "{}xAB_{:02.0f}.sof".format(nAB, nod_set_i))
    print(sof_file)

    # Write the SOF file along with calibration files at the end
    cal = open(calib_file, "r")
    
    with open(sof_file, 'w') as sof:
        # Step through and write every nodding pair up to nAB
        for file_i in range(nAB):
            counter = nod_set_i * nAB + file_i
            sof.write("{}\t OBS_NODDING_OTHER\n".format(
                os.path.join(cwd, sci_A[nod_set_i])))
            sof.write("{}\t OBS_NODDING_OTHER\n".format(
                os.path.join(cwd, sci_B[nod_set_i])))

        # Now append the calibration details
        sof.writelines(cal.readlines())
        cal.close()

    # And finally write the a file containing esorex reduction commands
    with open(reduce_script, 'a') as ww:
        ww.write("cd {}\n".format(os.path.join(cwd, dir)))
        esorex_cmd = ('esorex cr2res_obs_nodding --extract_swath_width=800'
                      + ' --extract_oversample=12 --extract_height=30 '
                      + sof_file + '\n')
        ww.write(esorex_cmd)