"""
"""
import glob
import numpy as np
import pandas as pd
from astropy.io import fits


def load_time_series_spectra(spectra_path_with_wildcard):
    """Imports time series spectra from globbed fits files given a filepath 
    with wildcard.
    """
    # Grab filenames and sort
    spec_seq_files = glob.glob(spectra_path_with_wildcard)
    spec_seq_files.sort()

    # Put all spectra in arrays
    wave = []
    spectra = []
    e_spectra = []
    times = []

    for spec_file in spec_seq_files:
        spec_fits = fits.open(spec_file)
        wave.append(spec_fits[1].data["SPLICED_1D_WL"])
        spectra.append(spec_fits[1].data["SPLICED_1D_SPEC"])
        e_spectra.append(spec_fits[1].data["SPLICED_1D_ERR"])
        times.append(spec_fits[0].header["DATE-OBS"].split("T")[-1].split(".")[0])

    obj = fits.getval(spec_file, "OBJECT")

    wave = np.array(wave)
    spectra = np.array(spectra)
    e_spectra = np.array(e_spectra)

    # Mask spectra
    bad_px_mask = np.logical_or(spectra > 10, spectra < 0)
    #masked_spec = np.ma.masked_array(spectra, ~mask)

    return wave, spectra, e_spectra, bad_px_mask