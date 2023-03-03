"""Script to prepare SOF files and shell script for reduction of raw CRIRES+ 
calibration files. This is part of a series of python scripts to assist with
reducing raw CRIRES+ data, which should be run in the following order:

    1 - make_calibration_sof.py             [reducing calibration files]
    2 - make_nodding_sof_master.py          [reducing master AB nodded spectra]
    3 - make_nodding_sof_split_nAB.py       [reducing nodding time series]
    4 - blaze_correct_nodded_spectra.py     [blaze correcting all spectra]
    5 - make_diagnostic_reduction_plots.py  [creating multipage PDF diagnostic]

The reduction approach for calibration data follows 'The Simple Way' outlined
in the data reduction manual: we only use the following three esorex commands: 
cr2res_cal_dark, cr2res_cal_flat, and cr2res_cal_wave. We prepare one SOF file
for each of these commands, and a single shell script to run the  commands one
after the other. It is assumed that all raw data is located in the same
directory and was taken by the same instrument/grating settings. 

We preferentially use the BPM generated from the flats, rather than the darks.
This is because in the IR the darks aren't truly dark due to thermal emission,
and this causes issues with the global outlier detection used by default in the
YJHK bands. As such, it is more robust to use the BPM generated from the flats
while also setting a high BPM kappa threshold when calling cr2res_cal_dark.
This ensures we only ever use the flat BPM, and our master dark isn't affected
by erroneous bad pixel flags.

While static trace wave files are inappropriate to use for extracting science
data, it is acceptable to pass them to cr2res_cal_flat as cr2res_cal_wave later
updates the tracewave solution in the working directory. Static detector
linearity files are appropriate so long as the SNR of our observations do not
exceed SNR~200.

Not all calibration files will necessarily be automatically associated with the
science frames when downloading from the ESO archive. In this case, they can be
downloaded manually by querying the night in question using a specific grating.
These calibrations are taken under the program ID 60.A-9051(A). For simplicity,
the settings for querying should be:
 - DATE OBS     --> YYYY MM DD (sometimes the night, otherwise morning after)
 - DPR CATG     --> CALIB
 - INS WLEN ID  --> [wl_setting]

The CRIRES specific archive is: https://archive.eso.org/wdb/wdb/eso/crires/form

Run as
------
python make_calibration_sofs.py [wl_setting]

where [wl_setting] is the grating setting, e.g. K2148.

Output
------
The following files are created by this routine, which by default contain:

dark.sof
    file/path/dark_1.fits         DARK
    ...
    file/path/dark_n.fits         DARK

flat.sof
    file/path/flat_1.fits         FLAT
    ...
    file/path/flat_n.fits         FLAT
    static/trace_wave.fits        UTIL_WAVE_TW
    file/path/dark_master.fits    CAL_DARK_MASTER

wave.sof
    file/path/wave_une_1.fits     WAVE_UNE
    ...
    file/path/wave_une_n.fits     WAVE_UNE
    file/path/wave_fpet_1.fits    WAVE_FPET
    ...
    file/path/wave_fpet_n.fits    WAVE_FPET
    static/trace.fits             UTIL_WAVE_TW
    file/path/flat_bpm.fits       CAL_FLAT_BPM
    file/path/master_dark.fits    CAL_DARK_MASTER
    file/path/master_flat.fits    CAL_FLAT_MASTER
    static/lines.fits             EMISSION_LINES

reduce_cals.sh:
    esorex cr2res_cal_dark  dark.sof --bpm_kappa=1000
    esorex cr2res_cal_flat  flat.sof --bpm_low=0.8 --bpm_high=1.2
    esorex cr2res_cal_wave  wave.sof
"""
import sys
import numpy as np
import os
import glob
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
# Get grating setting
wl_setting = sys.argv[1]

# Static calibration files
TRACE_WAVE = "/home/tom/pCOMM/cr2re-calib/{}_tw.fits".format(wl_setting)
DETLIN_COEFF = "/home/tom/pCOMM/cr2res_cal_detlin_coeffs.fits"
EMISSION_LINES_WC = \
    "/home/tom/pipes/svn.cr2re/cr2re-calib/lines_u_*_{}.fits"

EMISSION_LINE_FILES = glob.glob(EMISSION_LINES_WC.format(wl_setting))

# There should only be a single emission line file
if len(EMISSION_LINE_FILES) != 1:
    raise FileNotFoundError("There should only be a single emission line file")
else:
    EMISSION_LINE = EMISSION_LINE_FILES[0]

# Reduction parameters
BPM_KAPPA = 1000        # Default: -1, controls bad pixel threshold
BPM_LOW = 0.5           # Default: 0.8, controls *relative* bad pixel threshold
BPM_HIGH = 2.0          # Default: 1.2, controls *relative* bad pixel threshold

# Detector linearity is computed from a series of flats with a range of NDIT in
# three different grating settings to illuminate all pixels. This process takes
# around ~8 hours and the resulting calibration file has SNR~200. Note that for
# science files with SNR significantly above this, the linearisation process
# will actually *degrade* the data quality. Linearity is primarily a concern
# for science cases where one is interested in the relative depths of 
# absorption features rather than their locations (i.e. an abundance analysis
# would suffer more than a simply cross correlation). The use of detlin is
# *not* recommended for calibration frames.
use_detlin = False

# Whether to provide a BPM to the flat field routine. Since we by default are
# not making use of the BPM computed from the darks, there is no point in
# providing the dark BPM to the flats.
provide_bpm_to_flat = False

# -----------------------------------------------------------------------------
# Read in files
# -----------------------------------------------------------------------------
# Get current working directory
cwd = os.getcwd()

# Get a list of just the calibration files
fits_fns = glob.glob("CRIRE.*.fits")
fits_fns.sort()

# Initialise columns for our calibration data frame
cal_fns = []
objects = []
exp_times = []
wl_settings = []
ndits = []

cal_frames_kinds = ["FLAT", "DARK", "WAVE,UNE", "WAVE,FPET"]

# Populate our columns
for fn in fits_fns:
    # Grab the header
    header = fits.getheader(fn)

    # Skip those without required keywords (e.g."EMISSION_LINES, PHOTO_FLUX)
    if "HIERARCH ESO INS WLEN ID" not in header or "OBJECT" not in header:
        continue
    # Skip anything other than the specific calibration frames we want
    elif header["OBJECT"] not in cal_frames_kinds:
        continue
    # And skip anything not in our current wavelength setting
    elif header["HIERARCH ESO INS WLEN ID"] != wl_setting:
        continue
    
    # Compile info
    cal_fns.append(fn)
    objects.append(fits.getval(fn, "OBJECT"))
    exp_times.append(fits.getval(fn, "HIERARCH ESO DET SEQ1 DIT"))
    wl_settings.append(fits.getval(fn, "HIERARCH ESO INS WLEN ID"))
    ndits.append(fits.getval(fn, "HIERARCH ESO DET NDIT"))

# Create dataframe
cal_frames = pd.DataFrame(
    data=np.array([cal_fns, objects, exp_times, wl_settings, ndits]).T,
    index=np.arange(len(cal_fns)),
    columns=["fn", "object", "exp", "wl_setting", "ndit"],)

# Sort this by fn
cal_frames.sort_values(by="fn", inplace=True)

# -----------------------------------------------------------------------------
# Check set of calibration files for consistency
# -----------------------------------------------------------------------------
is_flat = cal_frames["object"] == "FLAT"
is_dark = cal_frames["object"] == "DARK"
is_wave_une = cal_frames["object"] == "WAVE,UNE"
is_wave_fpet = cal_frames["object"] == "WAVE,FPET"

# We should have a complete set of calibration frames
if np.sum(is_flat) == 0:
    raise FileNotFoundError("Missing flat fields.")
elif np.sum(is_dark) == 0:
    raise FileNotFoundError("Missing dark frames.")
elif np.sum(is_wave_une) == 0:
    raise FileNotFoundError("Missing wave UNe frames.")
elif np.sum(is_wave_fpet) == 0:
    raise FileNotFoundError("Missing wave FPET frames.")

# There should only be a single exposure time and NDIT for flats and wave cals
flat_exps = set(cal_frames[is_flat]["exp"])
wave_une_exps = set(cal_frames[is_wave_une]["exp"])
wave_fpet_exps = set(cal_frames[is_wave_fpet]["exp"])

flat_ndits = set(cal_frames[is_flat]["ndit"])
wave_une_ndits = set(cal_frames[is_wave_une]["ndit"])
wave_fpet_ndits = set(cal_frames[is_wave_fpet]["ndit"])

if  len(flat_ndits) > 1:
    raise Exception("There should only be one flat NDIT settting")
if len(wave_une_exps) > 1 or len(wave_une_ndits) > 1:
    raise Exception("There should only be one wave Une exp/NDIT settting")
if len(wave_fpet_exps) > 1 or len(wave_fpet_ndits) > 1:
    raise Exception("There should only be one wave FPET exp/NDIT settting")

# If we have multiples exposures for the flats, raise a warning, plot a
# diagnostic, and continue with the higher exposure. The user can then quickly
# check the plot for saturation and hopefully keep things the same.
if len(flat_exps) > 1 or len(flat_ndits) > 1:
    print('Warning, multiple sets of flats with exps: {}'.format(flat_exps))
    print("Have adopted higher exp. Check flat_comparison.pdf for saturation.")

    # Grab one of each flat (since we've sorted by filename, we should be safe
    # to grab the first and last file)
    fns = cal_frames[is_flat]["fn"].values[[0,-1]]
    exps = cal_frames[is_flat]["exp"].values[[0,-1]]

    # Plot a comparison for easy reference to check for saturation
    fig, axes = plt.subplots(2,3)
    for flat_i in range(2):
        with fits.open(fns[flat_i]) as flat:
            for chip_i in np.arange(1,4):
                xx = axes[flat_i, chip_i-1].imshow(flat[chip_i].data)
                fig.colorbar(xx, ax=axes[flat_i, chip_i-1])

            axes[flat_i, 0].set_ylabel("Exp = {}".format(exps[flat_i]))

    plt.savefig("flat_comparison.pdf")
    plt.close("all")

    # Grab the larger exposure
    flat_exp = \
        list(flat_exps)[np.argmax(np.array(list(flat_exps)).astype(float))]

else:
    flat_exp = list(flat_exps)[0]

wave_une_exp = list(wave_une_exps)[0]
wave_fpet_exp = list(wave_fpet_exps)[0]

flat_ndit = list(flat_ndits)[0]
wave_une_ndit = list(wave_une_ndits)[0]
wave_fpet_ndit = list(wave_fpet_ndits)[0]

# Raise an exception if we don't have a dark associated with each set of cals
if not np.any(np.isin(cal_frames[is_dark]["exp"], flat_exp)):
    raise Exception("No darks for flats with exp={} sec".format(flat_exp))
if not np.any(np.isin(cal_frames[is_dark]["exp"], wave_une_exp)):
    raise Exception("No darks for UNe with exp={} sec".format(wave_une_exp))
if not np.any(np.isin(cal_frames[is_dark]["exp"], wave_fpet_exp)):
    raise Exception("No darks for FPET with exp={} sec".format(wave_fpet_exp))

# Check that we have darks with exposure times and NDIT matching our flats, and
# if so prepare the filenames for the associated master darks
matches_flat_settings_mask = np.logical_and(
    cal_frames[is_dark]["exp"] == flat_exp,
    cal_frames[is_dark]["ndit"] == flat_ndit,)

if np.sum(matches_flat_settings_mask) == 0:
    raise Exception("No darks matching flat exposure and NDIT.")
else:
    # Format the flat exposure time so it writes properly. Keep up to 5 decimal
    # places, or cast to integer if they're the same.
    if float(flat_exp) == int(float(flat_exp)):
        flat_exp_str = "{:0.0f}".format(int(float(flat_exp)))
    else:
        flat_exp_str = "{:0.5f}".format(float(flat_exp))

    flat_master_dark = "cr2res_cal_dark_{}_{}x{}_master.fits".format(
        wl_setting, flat_exp_str, flat_ndit)
    flat_master_dark_bpm = "cr2res_cal_dark_{}_{}x{}_bpm.fits".format(
        wl_setting, flat_exp_str, flat_ndit)
    
# Check that we have darks with exposure times and NDIT matching our UNe frames
# and if so prepare the filenames for the associated master darks
matches_une_settings_mask = np.logical_and(
    cal_frames[is_wave_une]["exp"] == wave_une_exp,
    cal_frames[is_wave_une]["ndit"] == wave_une_ndit)

if np.sum(matches_une_settings_mask) == 0:
    raise Exception("No darks matching wave UNe exposure and NDIT.")
else:
    # Format the UNe exposure time so it writes properly. Keep up to 5 decimal
    # places, or cast to integer if they're the same.
    if float(wave_une_exp) == int(float(wave_une_exp)):
        wave_une_exp_str = "{:0.0f}".format(int(float(wave_une_exp)))
    else:
        wave_une_exp_str = "{:0.5f}".format(float(wave_une_exp))

    wave_une_master_dark = "cr2res_cal_dark_{}_{}x{}_master.fits".format(
        wl_setting, wave_une_exp_str, flat_ndit)
    wave_une_master_dark_bpm = "cr2res_cal_dark_{}_{}x{}_bpm.fits".format(
        wl_setting, wave_une_exp_str, flat_ndit)

# -----------------------------------------------------------------------------
# Initialise new shell script
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Write SOF file for darks
# -----------------------------------------------------------------------------
# Make SOF file for darks. The esorex recipe can do all our darks at once, so
# we may as well add them all.
dark_sof = "dark.sof"

with open(dark_sof, 'w') as sof:
    for dark in cal_frames[is_dark]["fn"]:
        sof.writelines("{}\tDARK\n".format(os.path.join(cwd, dark)))

# -----------------------------------------------------------------------------
# Write SOF for flats
# -----------------------------------------------------------------------------
flat_sof = "flat.sof"

with open(flat_sof, 'w') as sof:
    for flat in cal_frames[is_flat]["fn"]:
        sof.writelines("{}\tFLAT\n".format(os.path.join(cwd, flat)))

    sof.writelines("{}\tUTIL_WAVE_TW\n".format(TRACE_WAVE))

    if use_detlin:
        sof.writelines("{}\tCAL_DETLIN_COEFFS\n".format(DETLIN_COEFF))

    sof.writelines("{}\tCAL_DARK_MASTER\n".format(
        os.path.join(cwd, flat_master_dark)))

    if provide_bpm_to_flat:
        sof.writelines("{}\tCAL_DARK_BPM\n".format(
            os.path.join(cwd, flat_master_dark_bpm)))

# -----------------------------------------------------------------------------
# Write SOF for wave
# -----------------------------------------------------------------------------
wave_sof = "wave.sof"

flat_bpm = "cr2res_cal_flat_Open_bpm.fits"
flat_master = "cr2res_cal_flat_Open_master_flat.fits"

with open(wave_sof, 'w') as sof:
    for wave_une in cal_frames[is_wave_une]["fn"]:
        sof.writelines("{}\tWAVE_UNE\n".format(os.path.join(cwd, wave_une)))

    for wave_fpet in cal_frames[is_wave_fpet]["fn"]:
        sof.writelines("{}\tWAVE_FPET\n".format(os.path.join(cwd, wave_fpet)))

    sof.writelines("{}\tUTIL_WAVE_TW\n".format(TRACE_WAVE))

    if use_detlin:
        sof.writelines("{}\tCAL_DETLIN_COEFFS\n".format(DETLIN_COEFF))

    sof.writelines("{}\tCAL_FLAT_BPM\n".format(os.path.join(cwd, flat_bpm)))

    sof.writelines("{}\tCAL_DARK_MASTER\n".format(
        os.path.join(cwd, wave_une_master_dark)))

    sof.writelines("{}\tCAL_FLAT_MASTER\n".format(
        os.path.join(cwd, flat_master)))

    sof.writelines("{}\tEMISSION_LINES\n".format(EMISSION_LINE))

# -----------------------------------------------------------------------------
# Write shell script
# -----------------------------------------------------------------------------
# And finally write the file containing esorex reduction commands
with open(shell_script, 'a') as ww:
    esorex_cmd = ("esorex cr2res_cal_dark --bpm_kappa={} {}\n".format(
        BPM_KAPPA, dark_sof))
    ww.write(esorex_cmd)

    esorex_cmd = (
        "esorex cr2res_cal_flat --bpm_low={} --bpm_high={} {}\n".format(
            BPM_LOW, BPM_HIGH, flat_sof))
    ww.write(esorex_cmd)

    esorex_cmd = ("esorex cr2res_cal_wave {}\n".format(wave_sof))
    ww.write(esorex_cmd)
