"""Script to manually perform blaze correction on 1D extracted nodded spectra,
given the appropriate calibration blaze flat, and save as a new fits file. This
is part of a series of python scripts to assist with reducing raw CRIRES+ data,
which should be run in the following order:

    1 - make_calibration_sof.py             [reducing calibration files]
    2 - make_nodding_sof_master.py          [reducing master AB nodded spectra]
    3 - make_nodding_sof_split_nAB.py       [reducing nodding time series]
    4 - blaze_correct_nodded_spectra.py     [blaze correcting all spectra]
    5 - make_diagnostic_reduction_plots.py  [creating multipage PDF diagnostic]

Note that this does *not* call esorex, as currently its blaze correction
routine only runs alongside its splicing routine.

Run as
------
python blaze_correct_nodded_spectra.py [path] [cal_flat_blaze_path] [n_spec]

where [path] is the path to the data (which can include wildcards for
globbing), [cal_flat_blaze_path] is the calibration flat blaze fits file, and
[n_spec] is the number of spectra extracted along the slit in the case of 
observing extended objects (if not provided, defaults to 1).

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
from astropy.io import fits

# This is the default list of files within the provided directory to correct
FILE_PATTERNS = [
    "cr2res_obs_nodding_extractedA.fits",
    "cr2res_obs_nodding_extractedB.fits",
    "cr2res_obs_nodding_extracted_combined.fits",
    "cr2res_obs_pol_pol_specA.fits",
    "cr2res_obs_pol_pol_specB.fits",
    "cr2res_obs_pol_pol_spec_combined.fits",
]

# These headers should be the same across the science and blaze files
HEADER_EQUALITIES_TO_ENFORCE = [
    "HIERARCH ESO INS WLEN ID",
    "HIERARCH ESO INS GRAT1 MINORD",
    "HIERARCH ESO INS GRAT1 MAXORD",
    "HIERARCH ESO INS GRAT1 ZP_ORD",
    "HIERARCH ESO INS WLEN MAXCORD",
    "HIERARCH ESO INS WLEN MINCORD",
]

# Extension to the filename of the blaze corrected file
NEW_FN_EXT = "blaze_corr"

def blaze_corr_fits(
    sci_fits_path,
    blaze_fits_path,
    n_spec=1,
    new_ext="blaze_corr",
    detector_names=("CHIP1.INT1", "CHIP2.INT1", "CHIP3.INT1"),):
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

        # Check for header equality, and raise an exception if any of the
        # headers we're checking don't match
        for header in HEADER_EQUALITIES_TO_ENFORCE:
            if sci_fits[0].header[header] != blz_fits[0].header[header]:
                raise ValueError(
                    "Header '{}' unmatched across science and blaze".format(
                        header))

        # If our headers are all equal, we can grab the minimum an maximum
        # orders used in the fits table column names when doing blaze 
        # correction. Note that these are just the minimum and maximum orders
        # that might have *some* data, rather than the *complete* orders.
        header = sci_fits[0].header
        min_order = (header["HIERARCH ESO INS GRAT1 MINORD"]
                     - header["HIERARCH ESO INS GRAT1 ZP_ORD"] + 1)
        
        max_order = (header["HIERARCH ESO INS GRAT1 MAXORD"]
                     - header["HIERARCH ESO INS GRAT1 ZP_ORD"] + 1)

        # Determine whether this is a polarisation observation or not, as the
        # file format will be different. Polarisation observations will have
        # the 'SPU' (Spherical Polarisation Unit) optic listed.
        if sci_fits[0].header["HIERARCH ESO INS1 OPTI1 ID"] == "SPU":
            is_polarisation_ob = True
        
        # Nodding observations should be listed as 'FREE'
        elif sci_fits[0].header["HIERARCH ESO INS1 OPTI1 ID"] == "FREE":
            is_polarisation_ob = False

        # Not sure what other options there are, but break just in case
        else:
            raise Exception("Unknown observation type.")

        # Find global maximum for blaze function to divide by
        blaze_max = 0

        # This is a little inefficient, but we're going to have 2 separate for
        # loops looping over the extensions and columns. The first (this set)
        # will be used to look for the maximum value in counts of the blaze 
        # file to normalise it by), and the second will be used to actually do
        # the blaze normalisation. This is mostly due to fits record arrays 
        # being a pain to work with in a vectorised way.

        # First loop: finding max value in counts, loop over every detector,
        # and every order.
        for det in detector_names:
            for order_i in range(min_order, max_order+1):
                col_name = "{:02.0f}_01_SPEC".format(order_i)
                
                # Skip the column if the order is missing for this detector
                if col_name not in blz_fits[det].data.columns.names:
                    continue

                # Look for the maximum counts and update if we find a new max
                curr_max = np.nanmax(blz_fits[det].data[col_name])

                if curr_max > blaze_max:
                    blaze_max = curr_max

        # Second loop: normalising science data. Note that the blaze file 
        # should only have a single wave/spec/sigma combination for each 
        # detector/order combination, but a science file might have more in the
        # case of extended objects (e.g. Venus) where multiple spectra are 
        # extracted at different slit positions for each order. This is set by
        # the n_spec parameter. 
        # TODO: get this in a neater way from the header if possible
        for det in detector_names:
            for order_i in range(min_order, max_order):
                # Construct blaze column names
                blz_spec_col = "{:02.0f}_01_SPEC".format(order_i)
                blz_sigma_col = "{:02.0f}_01_ERR".format(order_i)

                for spec_i in range(1, n_spec+1):
                    # Construct science column names, noting that nodding and
                    # polarimetric observations have different formats
                    if is_polarisation_ob:
                        sci_spec_col = "{:02.0f}_INTENS".format(order_i)
                        sci_sigma_col = "{:02.0f}_INTENS_ERR".format(order_i)
                    else:
                        sci_spec_col = "{:02.0f}_{:02.0f}_SPEC".format(
                            order_i, spec_i)
                        sci_sigma_col = "{:02.0f}_{:02.0f}_ERR".format(
                            order_i, spec_i)

                    # Skip the column if the order is missing for this detector
                    if sci_spec_col not in sci_fits[det].data.columns.names:
                        continue

                    # Suppress division by zero and nan warnings. This should
                    # be fine so long as we use a proper bad pixel mask during 
                    # subsequent analysis.
                    with np.errstate(divide="ignore", invalid="ignore"):
                        # Normalise blaze by dividing by maximum
                        blz = blz_fits[det].data[blz_spec_col] / blaze_max
                        e_blz = blz_fits[det].data[blz_sigma_col] / blaze_max
                        
                        sci = sci_fits[det].data[sci_spec_col].copy()
                        e_sci = sci_fits[det].data[sci_sigma_col].copy()

                        # Blaze correct science frame & propagate uncertainties
                        sci_fits[det].data[sci_spec_col] = sci / blz
                        sci_fits[det].data[sci_sigma_col] = \
                            sci/blz * np.sqrt((e_sci/sci)**2 + (e_blz/blz)**2)

        # Save updated fits file
        sci_fits.flush()

# -----------------------------------------------------------------------------
# Run blaze correction
# -----------------------------------------------------------------------------
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

    # The location of the blaze file as outputted from the flat reduction
    blaze_fits = sys.argv[2]

    # And, optionally, n_spec: the number of spectra extracted for each order
    # at different slit positions
    if len(sys.argv) == 4:
        n_spec = int(sys.argv[3])

    # Default to 1 if not provided
    else:
        print("Defaulting n_spec to 1")
        n_spec = 1

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
            sci_fits.replace(".fits", "_{}.fits".format(NEW_FN_EXT)))

        print("Blaze correcting: {} --> {}".format(sci_fits, baze_corr_fits))
        blaze_corr_fits(sci_fits, blaze_fits, n_spec)

    print("\n{} files blaze corrected successfully!".format(len(fits_files)))