"""Script to prepare SOF files and shell script for reduction of the master set
of CRIRES+ nodded spectra. This is part of a series of python scripts to assist
with reducing raw CRIRES+ data, which should be run in the following order:

    1 - make_calibration_sof.py             [reducing calibration files]
    2 - make_nodding_sof_master.py          [reducing master AB nodded spectra]
    3 - make_nodding_sof_split_nAB.py       [reducing nodding time series]
    4 - blaze_correct_nodded_spectra.py     [blaze correcting all spectra]
    5 - make_diagnostic_reduction_plots.py  [creating multipage PDF diagnostic]

This script assumes that the raw calibration data has already been reduced
using make_calibration_sof.py and prepares one SOF file and a single shell 
script to run the science reduction. We assume that all raw data and reduced
calibration data is located in the same directory and was taken by the same 
instrument/grating settings. 

We preferentially use the BPM generated from the flats, rather than the darks.
This is because in the IR the darks aren't truly dark due to thermal emission,
and this causes issues with the global outlier detection used by default in the
YJHK bands. As such, it is more robust to use the BPM generated from the flats
while also setting a high BPM kappa threshold when calling cr2res_cal_dark.
This ensures we only ever use the flat BPM, and our master dark isn't affected
by erroneous bad pixel flags.

Note that static trace wave files are inappropriate to use for extracting
science data as they do not account for shifts in the wavelength scale due to 
warming/cooling of the CRIRES instrument (as happened in late 2022). Static
detector linearity files are appropriate so long as the SNR of our observations
do not exceed SNR~200.

Run as
------
python make_nodding_sof_master.py [setting]

Where [setting] is the grating setting as listed in the fits headers.

Output
------
This script outputs a sof file for the master reduction, and and associated .sh
file to call esorex to begin the reduction. Note that we do not need to provide
a master dark to the reduction as the nodding process obviates the necessity.
Both the SOF file and shell script are created in a new subdirector named after
the wavelength setting.

file/path/[setting]/[setting].sof
    file/path/sci_frame_1.fits    OBS_NODDING_OTHER
    ...
    file/path/sci_frame_n.fits    OBS_NODDING_OTHER
    file/path/trace_wave.fits     CAL_FLAT_TW_MERGED
    static/detlin_coeffs.fits     CAL_DETLIN_COEFFS
    file/path/dark_bpm.fits       CAL_FLAT_BPM
    file/path/master_flat.fits    CAL_FLAT_MASTER

file/path/reduce_master_[setting].sh
    #!/bin/bash
    cd file/path/[setting]/
    esorex cr2res_obs_nodding --extract_swath_width=800 --extract_oversample=10
        --extract_height=30 --extract_smooth_slit=10 --extract_smooth_spec=0.00
        --cosmics=TRUE file/path/[setting]/[setting].sof
"""
import sys
import os
import glob
import subprocess
from astropy.io import fits

# -----------------------------------------------------------------------------
# Settings & setup
# -----------------------------------------------------------------------------
# Get grating setting
wl_setting = sys.argv[1]

# Get current working directory
cwd = os.getcwd()

DETLIN_COEFF = "/home/tom/pCOMM/cr2res_cal_detlin_coeffs.fits"
MASTER_FLAT = os.path.join(cwd, "cr2res_cal_flat_Open_master_flat.fits")
FLAT_BPM = os.path.join(cwd, "cr2res_cal_flat_Open_bpm.fits")
TRACE_WAVE = os.path.join(cwd, "cr2res_cal_flat_tw_merged.fits")

# No point continuing if our calibration files don't exist
if not os.path.isfile(MASTER_FLAT):
    raise Exception("Master flat not found.")

if not os.path.isfile(FLAT_BPM):
    raise Exception("Flat BPM not found.")

if not os.path.isfile(TRACE_WAVE):
    raise Exception("Trace wave not found.")

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
EXTRACT_SMOOTH_SPEC = 0.01

# Find and mark cosmic rays hits as bad. Default: False
DO_COSMIC_CORR = True

# Detector linearity is computed from a series of flats with a range of NDIT in
# three different grating settings to illuminate all pixels. This process takes
# around ~8 hours and the resulting calibration file has SNR~200. Note that for
# science files with SNR significantly above this, the linearisation process
# will actually *degrade* the data quality. Linearity is primarily a concern
# for science cases where one is interested in the relative depths of 
# absorption features rather than their locations (i.e. an abundance analysis
# would suffer more than a simply cross correlation). The use of detlin is
# *not* recommended for calibration frames.
use_detlin = True

# -----------------------------------------------------------------------------
# Read in files
# -----------------------------------------------------------------------------
# Get a list of just the observations at our grating 
fits_fns = glob.glob("CRIRE.*.fits")
fits_fns.sort()

obs_fns = []
n_a_frames = 0
n_b_frames = 0

for fn in fits_fns:
    obs_cat = fits.getval(fn, "HIERARCH ESO DPR CATG")

    if obs_cat == "SCIENCE":
        obs_setting = fits.getval(fn, "HIERARCH ESO INS WLEN ID")

        if obs_setting == wl_setting:
            obs_fns.append(fn)

            # Count nod positions
            if fits.getval(fn, 'HIERARCH ESO SEQ NODPOS') == "A":
                n_a_frames += 1
            elif fits.getval(fn, 'HIERARCH ESO SEQ NODPOS') == "B":
                n_b_frames += 1

# Raise an exception if we don't have any files, or if we have an unmatched set
# of AB pairs
if len(obs_fns) < 1:
    raise Exception("No files found!")
elif n_a_frames != n_b_frames:
    raise Exception("Unmatched set of AB pairs, #A = {}, #B = {}".format(
        n_a_frames, n_b_frames))

# -----------------------------------------------------------------------------
# Initialise new shell script
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Write new shell script
# -----------------------------------------------------------------------------
obs_sof = os.path.join(cwd, wl_setting, "{}.sof".format(wl_setting))

with open(obs_sof, 'w') as sof:
    # First write all science files
    for obs_fn in obs_fns:
        obs_fn_path = os.path.join(cwd, obs_fn)
        sof.writelines("{}\tOBS_NODDING_OTHER\n".format(obs_fn_path))

    # Then write trace wave, detlin, bpm, and master flat
    sof.writelines("{}\tCAL_FLAT_TW_MERGED\n".format(TRACE_WAVE))

    if use_detlin:
        sof.writelines("{}\tCAL_DETLIN_COEFFS\n".format(DETLIN_COEFF))

    sof.writelines("{}\tCAL_FLAT_BPM\n".format(FLAT_BPM))
    sof.writelines("{}\tCAL_FLAT_MASTER\n".format(MASTER_FLAT))

# And finally write the a file containing esorex reduction commands
with open(shell_script, 'a') as ww:
    ww.write("cd {}\n".format(os.path.join(cwd, wl_setting)))
    esorex_cmd = (
        'esorex cr2res_obs_nodding '
        + '--extract_swath_width={:0.0f} '.format(SWATH_WIDTH)
        + '--extract_oversample={:0.0f} '.format(EXTRACT_OVERSAMPLE)
        + '--extract_height={:0.0f} '.format(EXTRACT_HEIGHT)
        + '--extract_smooth_slit={:0.1f} '.format(EXTRACT_SMOOTH_SLIT)
        + '--extract_smooth_spec={:0.2f} '.format(EXTRACT_SMOOTH_SPEC)
        + '--cosmics={} '.format(str(DO_COSMIC_CORR).upper())
        + obs_sof + '\n')
    ww.write(esorex_cmd)