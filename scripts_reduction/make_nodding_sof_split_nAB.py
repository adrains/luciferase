"""Script to prepare SOF files and shell script for reduction of time series
CRIRES+ nodded spectra. This is part of a series of python scripts to assist
with reducing raw CRIRES+ data, which should be run in the following order:

    1 - make_calibration_sof.py             [reducing calibration files]
    2 - make_nodding_sof_master.py          [reducing master AB nodded spectra]
    3 - make_nodding_sof_split_nAB.py       [reducing nodding time series]
    4 - blaze_correct_nodded_spectra.py     [blaze correcting all spectra]
    5 - make_diagnostic_reduction_plots.py  [creating multipage PDF diagnostic]

This script assumes that the raw calibration data *and* the master AB nodded 
spectra has already been reduced using make_calibration_sof.py and 
make_nodding_sof_master.py respectively. It prepares one SOF file per time step
(each placed within a separate subfolder) and a single shell script to run the 
time-series science reduction. We assume that all raw data and reduced 
calibration data is located in the same directory and was taken by the same 
instrument/grating settings. We reference the SOF created for the master
reduction--specifically the calibration files listed within--when creating our
SOFs here, which is why this script requires make_nodding_sof_master.py to have
been run previously.

Run as
------
python make_nodding_sof_split_nAB.py [nAB] [master_sof]

Where you replace [nAB] with the number of nodding observations you would like
to reduce together. E.G. nAB=1 will combine the nearest AB/BA pair in time, 
whereas nAB=2 will combine the nearest AABB/BBAA or ABAB/BABA pair in time. The
exact format will depend on the specifics of how the nodding observations were
conducted.

Creates a nod.sof SOF file and a reduce_[nAB]xAB.sh script that reduces your 
nodding observations for each nodding set, and places each in a separate 
subfolder for set/timestep with format [nAB]xAB_[ii]/ where ii is the nodding
set.

Output
------
This script outputs a sof file for *each* set of AB frames to a new subfolder,
and a single .sh file to call esorex to begin the reductions for *all* sets of
AB frames. Note that, like the master reductions, we don't need to provide a
master dark frame since the process of nodding obviates the need.

[nAB]_[nod_set_i]/[nAB]_[nod_set_i].sof
    file/path/[nAB]_[nod_set_i]/sci_frame_a_1.fits    OBS_NODDING_OTHER
    file/path/[nAB]_[nod_set_i]/sci_frame_b_1.fits    OBS_NODDING_OTHER
    ...
    file/path/[nAB]_[nod_set_i]/sci_frame_a_n.fits    OBS_NODDING_OTHER
    file/path/[nAB]_[nod_set_i]/sci_frame_b_n.fits    OBS_NODDING_OTHER
    file/path/trace_wave.fits                         CAL_WAVE_TW
    static/detlin_coeffs.fits                         CAL_DETLIN_COEFFS
    file/path/dark_bpm.fits                           CAL_FLAT_BPM
    file/path/master_flat.fits                        CAL_FLAT_MASTER

reduce_[nAB]xAB.sh
    #!/bin/bash
    cd file/path/[nAB]_00/
    esorex cr2res_obs_nodding --extract_swath_width=400 --extract_oversample=12
        --extract_height=30 file/path/[nAB]_00/[nAB]_00.sof
    ...
    cd file/path/[nAB]_NN/
    esorex cr2res_obs_nodding --extract_swath_width=800 --extract_oversample=10
        --extract_height=30 --extract_smooth_slit=10 --extract_smooth_spec=0.00
        --cosmics=TRUE file/path/[nAB]_NN/[nAB]_NN.sof
"""
import sys
import numpy as np
import glob
from astropy.io import fits
import os
import subprocess

# -----------------------------------------------------------------------------
# Settings & setup
# -----------------------------------------------------------------------------
# Get current working directory
cwd = os.getcwd()

# Get the number of frames to combine
nAB = int(sys.argv[1])

# Get the SOF file used to reduce the master observations
master_sof = sys.argv[2]

# Sanity check print
if ".sof" not in master_sof:
    print("Are you sure {} is a .sof file?".format(master_sof))

# Swath width in spectral dimension for extraction with reduce algorithm
SWATH_WIDTH = 800           # Default: 800

# Factor by which to oversample the extraction
EXTRACT_OVERSAMPLE = 10     # Default: 7

# Amount of slit to extract
EXTRACT_HEIGHT = 30         # Default: -1 (i.e. full slit)

# Regularization parameter for the slit-illumination vector, should not be set
# below 1.0 to avoid ringing. Default: 2.0
EXTRACT_SMOOTH_SLIT = 10

# Analogous to the previous, but along the spectrum instead. Defaults
# to 0.0 to not degrade resolution, but can be increased as needed.
EXTRACT_SMOOTH_SPEC = 0.0

# Find and mark cosmic rays hits as bad. Default: False
DO_COSMIC_CORR = True

# -----------------------------------------------------------------------------
# Read in files
# -----------------------------------------------------------------------------
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

# We'll take the calibration file information directly from the SOF used to
# reduce the master observations
if not os.path.isfile(master_sof):
    raise FileNotFoundError("No master SOF file, aborting.")
else:
    # Open the master SOF and grab only the lines containing calibration files.
    # These will be the lines *without* 'OBS_NODDING_OTHER'
    with open(master_sof, "r") as cal:
        cal_lines_all = cal.readlines()
        cal_lines = \
            [cl for cl in cal_lines_all if "OBS_NODDING_OTHER" not in cl]

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

# -----------------------------------------------------------------------------
# Initialise new shell script
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Write SOF and update shell script for each AB pair
# -----------------------------------------------------------------------------
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

    with open(sof_file, 'w') as sof:
        # Step through and write every nodding pair up to nAB
        for file_i in range(nAB):
            counter = nod_set_i * nAB + file_i
            sof.write("{}\t OBS_NODDING_OTHER\n".format(
                os.path.join(cwd, sci_A[nod_set_i])))
            sof.write("{}\t OBS_NODDING_OTHER\n".format(
                os.path.join(cwd, sci_B[nod_set_i])))

        # Now append the calibration details
        sof.writelines(cal_lines)

    # And finally write the a file containing esorex reduction commands
    with open(reduce_script, 'a') as ww:
        ww.write("cd {}\n".format(os.path.join(cwd, dir)))
        esorex_cmd = (
            'esorex cr2res_obs_nodding '
            + '--extract_swath_width={:0.0f} '.format(SWATH_WIDTH)
            + '--extract_oversample={:0.0f} '.format(EXTRACT_OVERSAMPLE)
            + '--extract_height={:0.0f} '.format(EXTRACT_HEIGHT)
            + '--extract_smooth_slit={:0.1f} '.format(EXTRACT_SMOOTH_SLIT)
            + '--extract_smooth_spec={:0.2f} '.format(EXTRACT_SMOOTH_SPEC)
            + '--cosmics={} '.format(str(DO_COSMIC_CORR).upper())
            + sof_file + '\n')
        ww.write(esorex_cmd)