"""
Script to prepare a blaze corrected and spliced spectrum for use with molecfit.

Usage
-----
Run as a python script with two input arguments:
    python3 cr2rem_drs2molecfit_spliced.py [1] [2]
where [1] is a blaze corrected spliced cr2res DRS extracted spectrum and [2] is
the 1D extracted spectrum that was used to make it.

Output
------
 (1) SCIENCE.fits - SCIENCE file for Molecfit
 (2) WAVE_INCLUDE.fits - WAVE_INCLUDE file for Molecfit
 (3) ATM_PARAMETERS_EXT.fits - ATM_PARAMETERS_EXT file for Molecfit
 (4) PIXEL_EXCLUDE.fits

Adapted from code written by Alexis Lavail, modified by Adam Rains.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys

# Take as input an 1D extracted, spliced, blaze corrected fits file. For this
# format, the data array header keywords are:
#   [SPLICED_1D_WL, SPLICED_1D_SPEC, SPLICED_1D_ERR]
input_sci_file = sys.argv[1]

# Also take as input the corresponding unspliced 1D extracted spectrum so we
# can determine the first spectral order number for use in naming the new fits
# extensions. e.g. for K2148 the first order is 2.
unspliced_sci_file = sys.argv[2]

with fits.open(unspliced_sci_file) as unspliced_sci:
    # Get column names, do string formating to find minumum order. Column names
    # have format [n_order]_[n_detector]_[SPEC/ERR/WAVE].
    columns = unspliced_sci[1].data.columns.names

    first_order = np.min([int(col.split("_")[0]) for col in columns])

# Initialise pixel counter. This is used to count the number of pixels in a
# given spectral segment, as the blaze correction and splicing routine removes
# pixels from the end of each segment, meaning len(segment) =/= n_px_chip.
px_so_far = 0

# Spliced blaze corrected spectra have fewer pixels at the high end of each 
# order. We want to padd them back up to the original number for compatability 
# with molecfit.
n_px_chip = 2048

# Initialise our output filename
output = "SCIENCE.fits"

with fits.open(input_sci_file) as sci_fits_in:
    # Grab the header information from the primary HDU
    primary_hdu = sci_fits_in[0]

    # Initialise HDU lists for each of our output fits files, assigning each
    # the primary HDU from our input science file.
    hdul_output = fits.HDUList([primary_hdu])
    hdul_winc = fits.HDUList([primary_hdu])
    hdul_atm =  fits.HDUList([primary_hdu])
    hdul_excl =  fits.HDUList([primary_hdu])

    # Now initialise other arrays
    wmin, wmax  = [], []
    map2chip, map2chip_excl = [], []
    atm_parms_ext = []
    pixExclmin,pixExclmax = [], []
    wexclmin, wexclmax = [], []

    # For convenience, grab data arrays
    wave_all = sci_fits_in[1].data["SPLICED_1D_WL"]
    spectra_all = sci_fits_in[1].data["SPLICED_1D_SPEC"]
    sigma_all = sci_fits_in[1].data["SPLICED_1D_ERR"]

    # We want to put each segment from the spliced spectrum into its own HDU
    delta_wl = np.abs(wave_all[:-1] - wave_all[1:])
    gap_px = np.argwhere(delta_wl > 10*np.median(delta_wl))[:,0] + 1
    gap_px = np.concatenate((gap_px, [len(wave_all)]))  # Go to end of array
    spec_px_low = 0

    # Loop over every spectral segment. Unlike the 1D extracted spectra from
    # the nodding routine, the spectra here are already in wavelength order.
    for spec_i, spec_px_high in enumerate(gap_px):
        # Get the chip and spectral order indices
        chip_i = spec_i % 3 + 1
        order_i = first_order + 1 * (spec_i // 3)

        # Grab the particular spectral segment we're considering
        wave = wave_all[spec_px_low:spec_px_high]
        spectra = spectra_all[spec_px_low:spec_px_high]
        sigma = sigma_all[spec_px_low:spec_px_high]

        # Pad arrays if length is below n_px
        if len(wave) < n_px_chip:
            # Get wavelength spacing
            delta_wave = np.median(wave[1:] - wave[:-1])

            # Extend wavelength vector
            n_pad = n_px_chip - len(wave)
            wave_to_pad = wave[-1] + np.arange(1, n_pad+1) * delta_wave

            # Initialise empty spectral and sigma arrays
            spectra_to_pad = np.full(n_pad, np.nan)
            sigma_to_pad = np.full(n_pad, np.nan)

            # Pad
            wave = np.concatenate((wave, wave_to_pad))
            spectra = np.concatenate((spectra, spectra_to_pad))
            sigma = np.concatenate((sigma, sigma_to_pad))

        #print("Chip {}, Order {}".format(chip_i, order_i))

        # Create fits columns for WAVE, SPEC, and ERR
        col1 = fits.Column(name='WAVE', format='D', array=wave)
        col2 = fits.Column(name='SPEC', format='D', array=spectra)
        col3 = fits.Column(name='ERR', format='D', array=sigma)

        # Create fits table HDU with WAVE SPEC ERR
        table_hdu = fits.BinTableHDU.from_columns([col1, col2, col3])

        # Append table HDU to output HDU list
        hdul_output.append(table_hdu)

        # Append wmin for given order to wmin array and convert to micron
        wmin.append(np.min(wave)*0.001)

        # Append wmax for giver order to wmax array and convert to micron
        wmax.append(np.max(wave)*0.001)

        # Setup the columns for PIXEL_EXCLUDE. The goal here is to exclude the 
        # edge pixels on each side of the detector (which are physically masked
        # and don't receive flux) to avoid edge effects when molecfit does 
        # continuum fitting with a polynomial as they are physically masked and
        # don't receive flux.
        pixExclmin.append(px_so_far)
        pixExclmin.append(len(wave)-40 + px_so_far)
        pixExclmax.append(40 + px_so_far)
        pixExclmax.append(len(wave) + px_so_far)

        #wexclmin.append((np.min(wave))*0.001)
        #wexclmax.append((np.min(wave)+0.2)*0.001)

        #wexclmin.append((np.max(wave)-0.2)*0.001)
        #wexclmax.append((np.max(wave))*0.001)
        
        hdul_output[spec_i+1].header.append(
            ("EXTNAME",
            "det{:02.0f}_{:02.0f}_01".format(chip_i, order_i),
            "order/detector"),
            end=True)
        map2chip_excl.append(spec_i+1)
        map2chip_excl.append(spec_i+1)

        # Append order counter to map2chip
        map2chip.append(spec_i+1) 
        if spec_i+1 == 1:
            atm_parms_ext.append(0)
        else:
            atm_parms_ext.append(1)

        # Update the new minimum for the next spectral segment
        spec_px_low = spec_px_high

        # Index pixel counter
        px_so_far += len(wave)

    # Setup the columns for the WAVE_INCLUDE output file
    CONT_FIT_FLAG = np.ones(len(wmin))
    wmin_winc = fits.Column(name='LOWER_LIMIT', format='D', array=wmin)
    wmax_winc = fits.Column(name='UPPER_LIMIT', format='D', array=wmax)
    map2chip_winc = fits.Column(
        name='MAPPED_TO_CHIP', format='I', array=map2chip)
    contflag_winc = fits.Column(
        name='CONT_FIT_FLAG', format='I', array=CONT_FIT_FLAG)

    # Creates WAVE_INCLUDE table HDU and appends to the HDU list
    table_hdu_winc = fits.BinTableHDU.from_columns(
        [wmin_winc, wmax_winc, map2chip_winc, contflag_winc])
    hdul_winc.append(table_hdu_winc)

    pmin_excl = fits.Column(name='LOWER_LIMIT', format='K', array=pixExclmin)
    pmax_excl = fits.Column(name='UPPER_LIMIT', format='K', array=pixExclmax)

    #map2chip_excl = fits.Column(
        # name='MAPPED_TO_CHIP', format='I', array=map2chip_excl)
    #table_hdu_pexcl = fits.BinTableHDU.from_columns(
        # [pmin_excl, pmax_excl, map2chip_excl])
    #hdul_excl.append(table_hdu_pexcl)
    #hdul_excl.writeto('PIXEL_EXCLUDE.fits', overwrite=True)

    #pmin_excl = fits.Column(name='LOWER_LIMIT', format='D', array=wexclmin)   
    #pmax_excl = fits.Column(name='UPPER_LIMIT', format='D', array=wexclmax)
    #map2chip_excl = fits.Column(
        # name='MAPPED_TO_CHIP', format='I', array=map2chip_excl)

    table_hdu_pexcl = fits.BinTableHDU.from_columns([pmin_excl, pmax_excl])
    hdul_excl.append(table_hdu_pexcl)
    hdul_excl.writeto('PIXEL_EXCLUDE.fits', overwrite=True)

    # Write to the output file
    hdul_output.writeto(output, overwrite=True)

    # Write to WAVE_INCLUDE file
    hdul_winc.writeto('WAVE_INCLUDE.fits', overwrite=True)

    col1_atm = fits.Column(
        name='ATM_PARAMETERS_EXT', format='I', array=atm_parms_ext)
    table_hdu_atm = fits.BinTableHDU.from_columns([col1_atm])
    hdul_atm.append(table_hdu_atm)
    hdul_atm.writeto('MAPPING_ATMOSPHERIC.fits', overwrite=True)