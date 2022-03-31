"""Script to manually perform blaze correction on 1D extracted nodded spectra,
given the appropriate calibration blaze flat, and save as a new fits file.

Run as
------
python blaze_correct_nodded_spectra.py [data_path] [cal_flat_blaze_path]

where [data_path] is the path to the data (which can include wildcards for
globbing), and [cal_flat_blaze_path] is the calibration flat blaze fits file.
Make sure to wrap any wildcard filepaths in quotes to prevent globbing
happening on the command line (versus in python).

Blaze corrected spectra are saved in the same location as the original science
file, only now with a new extension (by default _baze_corr).
"""
import os
import sys
import glob
import subprocess
import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits

# This is the default list of files within the provided directory to correct
FILE_PATTERNS = [
    "cr2res_obs_nodding_extractedA.fits",
    "cr2res_obs_nodding_extractedB.fits",
    "cr2res_obs_nodding_extracted_combined.fits",
]

def blaze_corr_fits(
    sci_fits_path,
    blaze_fits_path,
    new_ext="blaze_corr",
    fits_ext_names=("CHIP1.INT1", "CHIP2.INT1", "CHIP3.INT1"),):
    """Correct a single extracted nodding spectrum for the blaze function.

    Assume that both filenames are in absolute form.
    """
    # Duplicate existing fits file with new extension
    new_sci_fits = sci_fits_path.replace(".fits", "_{}.fits".format(new_ext))
    bashCommand = "cp {} {}".format(sci_fits_path, new_sci_fits)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # Load in blaze function and science frame
    with fits.open(new_sci_fits, mode="update") as sci_fits, \
        fits.open(blaze_fits_path) as blz_fits:
        # Find global maximum for blaze function to divide by
        blaze_max = 0

        # This is a little inefficient, but we're going to have 2 separate for
        # loops looping over the extensions and columns. The first (this set)
        # will be used for setup (error checking and looking for the maximum
        # value in counts of the blaze file to normalise it by), and the second
        # will be used to actually do the blaze normalisation. This is mostly
        # due to fits record arrays being a pain to work with in a vectorised
        # way.

        # First loop: error checking and finding max value in counts
        for fits_ext in fits_ext_names:
            # Get column names and verify they're consistent across files
            cols_sci = sci_fits[fits_ext].data.columns.names
            cols_blz = blz_fits[fits_ext].data.columns.names

            col_match = [cs == cb for (cs, cb) in zip(cols_sci, cols_blz)]

            # If not consistent, no point continuing
            if not np.all(col_match):
                raise Exception("Columns don't match!")

            # Look for the maximum counts, assuming that the columns are 
            # ordered as [SPEC, ERR, WL] and update if we find a new maximum.
            for col in cols_blz[::3]:
                curr_max = np.nanmax(blz_fits[fits_ext].data[col])

                if curr_max > blaze_max:
                    blaze_max = curr_max

        # Second loop: normalising science data
        for fits_ext in fits_ext_names:
            # Get column names
            cols = sci_fits[fits_ext].data.columns.names

            # Blaze correct science data with normalised blaze function
            for flux_col, err_col in zip(cols[::3], cols[1::3]):
                # Suppress division by zero and nan warnings. This should be
                # fine so long as we use a proper bad pixel mask during 
                # subsequent analysis.
                with np.errstate(divide="ignore", invalid="ignore"):
                    # Normalise blaze by dividing by maximum
                    blz = blz_fits[fits_ext].data[flux_col] / blaze_max
                    e_blz = blz_fits[fits_ext].data[err_col] / blaze_max
                    
                    sci = sci_fits[fits_ext].data[flux_col].copy()
                    e_sci = sci_fits[fits_ext].data[err_col].copy()

                    # Blaze correct science frame and propagate uncertainties
                    sci_fits[fits_ext].data[flux_col] = sci / blz
                    sci_fits[fits_ext].data[err_col] = \
                        sci * np.sqrt((e_sci/sci)**2 + (e_blz/blz)**2)

        # Save updated fits file
        sci_fits.flush()


# When called as a script, we want to accept as input a path (potentially) with
# wildcards) and from that perform blaze correction on all files within 
# matching the FILE_PATTERNS listed above.
if __name__ == "__main__":
    # Double check we haven't been given too many inputs
    if len(sys.argv) > 4:
        exception_text = (
            "Warning, too many inputs ({})! "
            "Make sure to wrap paths with wildcard characters in quotes.")

        raise Exception(exception_text.format(len(sys.argv)))

    # Take as input the folder/s to consider
    data_dir = sys.argv[1]

    # And the location of the blaze file as outputted from the flat reduction
    blaze_fits = sys.argv[2]

    # And the new extension
    if len(sys.argv) > 3:
        new_ext = sys.argv[3]
    else:
        new_ext = "blaze_corr"

    if not os.path.isfile(blaze_fits):
        raise FileNotFoundError("Specified blaze file not found.")

    # Look for each pattern
    fits_files = []

    for pattern in FILE_PATTERNS:
        fits_files += glob.glob(os.path.join(data_dir, pattern))

    fits_files.sort()

    print("{} files found to blaze correct.".format(len(fits_files)))

    # Blaze correct each file individually
    for sci_fits in fits_files:
        baze_corr_fits = os.path.join(
            os.path.dirname(sci_fits),
            blaze_fits.replace(".fits", "_{}.fits".format(new_ext)))

        print("Blaze correcting: {} --> {}".format(sci_fits, baze_corr_fits))
        blaze_corr_fits(sci_fits, blaze_fits)

    print("\n{} files blaze corrected successfully!".format(len(fits_files)))