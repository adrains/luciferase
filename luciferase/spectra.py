"""
"""
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt

VALID_NOD_POS = ["A", "B", None,]


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


class Spectrum1D(object):
    """Base class to represent a single spectral segment.

    Parameters
    ----------
    wave, flux, sigma: float array
        Wavelength, flux, and uncertainty vectors for the spectrum.

    bad_px_mask: boolean array
        Bad pixel mask for the spectrum. True corresponds to bad pixels.

    detector_i, order_i: int
        Integer label for the detector and order assocated with the spectrum.
    """
    def __init__(
        self,
        wave,
        flux,
        sigma,
        bad_px_mask,
        detector_i,
        order_i,):
        """
        """
        # Do initial checking of array lengths. After this initial check, we
        # reference everything to n_px.
        if (len(wave) != len(flux) and len(wave) != len(sigma) 
            and len(wave) != len(bad_px_mask)):
            raise ValueError(
                ("Error, wave, spectrum, e_spectrum, and bad_px_mask must"
                 "have the same length."))

        self.n_px = len(wave)
        self.wave = wave
        self.flux = flux
        self.sigma = sigma
        self.bad_px_mask = bad_px_mask
        self.detector_i = detector_i
        self.order_i = order_i

    @property
    def n_px(self):
        return self._n_px

    @n_px.setter
    def n_px(self, value):
        self._n_px = int(value)

    @property
    def wave(self):
        return self._wave

    @wave.setter
    def wave(self, value):
        # Check dimensions
        if len(value) != self.n_px:
            raise ValueError("Error, length of array != n_px.")
        else:
            self._wave = np.array(value)

    @property
    def flux(self):
        return self._flux

    @flux.setter
    def flux(self, value):
        # Check dimensions
        if len(value) != self.n_px:
            raise ValueError("Error, length of array != n_px.")
        else:
            self._flux = np.array(value)

            # Also update SNR
            self.update_snr()

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        # Check dimensions
        if len(value) != self.n_px:
            raise ValueError("Error, length of array != n_px.")
        else:
            self._sigma = np.array(value)

    @property
    def bad_px_mask(self):
        return self._bad_px_mask

    @bad_px_mask.setter
    def bad_px_mask(self, value):
        # Check dimensions
        if len(value) != self.n_px:
            raise ValueError("Error, length of array != n_px.")
        else:
            self._bad_px_mask = np.array(value).astype(bool)

    @property
    def detector_i(self):
        return self._detector_i

    @detector_i.setter
    def detector_i(self, value):
        self._detector_i = int(value)

    @property
    def order_i(self):
        return self._order_i

    @order_i.setter
    def order_i(self, value):
        self._order_i = int(value)

    @property
    def snr(self):
        return self._snr

    @snr.setter
    def snr(self, value):
        self._snr = int(value)

    def update_snr(self):
        """Simple function to recompute SNR assuming Poisson uncertainties."""
        self.snr = np.nanmedian(self.flux) / np.sqrt(np.nanmedian(self.flux))

    def __str__(self):
        """String representation of Spectrum1D details."""
        info_str = (
            "Detector {:0.0f}, ".format(self.detector_i),
            "Order {:0.0f}, ".format(self.order_i),
            "npx {:0.0f}, ".format(self.n_px),
            "lambda mid {:0.1f} um, ".format(np.nanmean(self.wave)),
            "snr {:0.0f}".format(self.snr),
        )
        return "".join(info_str)


class Observation(object):
    """Class to represent a single observation being composed of at least one
    spectral segment, but possiby more depending on spectral orders.

    Parameters
    ----------
    spectra_1d: luciferase.spectra.Spectra1D array
        Array of luciferase.spectra.Spectra1D objects, one for each discrete
        spectral segment making up the exposure (e.g. order, detector)

    t_exp_sec: float
        Duration of the exposure in seconds.

    t_start_str: string
        Start time of observation in human readable string.

    t_start_jd: float
        Observation start time in JD.

    nod_pos: string
        Nodding position for nodding observations.

    object_name: string
        Name of the object.

    grating_setting: string
        Grating setting for the observation.

    min_order, max_order: int
        Minimum and maximum echelle order.

    spectra_1d_blaze_corr, spectra_1d_telluric_corr: TODO, default: None,
        Currently unused.

    n_detectors: int, default: 3
        Number of detectors.
    """
    def __init__(
        self,
        spectra_1d,
        t_exp_sec,
        t_start_str,
        t_start_jd,
        nod_pos,
        object_name,
        grating_setting,
        min_order,
        max_order,
        spectra_1d_blaze_corr=None,
        spectra_1d_telluric_corr=None,
        n_detectors=3,):
        """
        """
        self.spectra_1d = spectra_1d
        self.t_exp_sec = t_exp_sec
        self.t_start_str = t_start_str
        self.t_start_jd = t_start_jd
        self.nod_pos = nod_pos
        self.object_name = object_name
        self.grating_setting = grating_setting
        self.min_order = min_order
        self.max_order = max_order
        #self.spectra_1d_blaze_corr = spectra_1d_blaze_corr
        #self.spectra_1d_telluric_corr = spectra_1d_telluric_corr

        # Calculate t_mid_jd and t_end_jd. TODO: make this automatic.
        self.update_t_mid_and_end()

        # Add the number of orders for ease of use, assuming 3 detectors
        self.n_detectors = n_detectors
        self.n_orders = len(spectra_1d) // n_detectors

    @property
    def spectra_1d(self):
        return self._spectra_1d

    @spectra_1d.setter
    def spectra_1d(self, value):
        # Make sure the result is a list of Spectrum1D objects
        if type(value) != list:
            raise ValueError("spectra_1d should be a list of spectra.")
        
        for spec_i, spectrum in enumerate(value):
            if (type(spectrum) != Spectrum1D 
            and issubclass(type(spectrum), Spectrum1D)):
                raise ValueError((
                    "Spectrum #{:0.0f} in spectrum_1d is not of type "
                    "Spectrum1D or inherited from it.").format(spec_i))

        # Otherwise all good
        self._spectra_1d = value

    @property
    def t_exp_sec(self):
        return self._t_exp_sec

    @t_exp_sec.setter
    def t_exp_sec(self, value):
        self._t_exp_sec = float(value)

    @property
    def t_start_str(self):
        return self._t_start_str

    @t_start_str.setter
    def t_start_str(self, value):
        self._t_start_str = str(value)

    @property
    def t_start_jd(self):
        return self._t_start_jd

    @t_start_jd.setter
    def t_start_jd(self, value):
        self._t_start_jd = float(value)

    @property
    def nod_pos(self):
        return self._nod_pos

    @nod_pos.setter
    def nod_pos(self, value):
        if value not in VALID_NOD_POS:
            raise ValueError(
                "Invalid nod_pos, must be in {}".format(VALID_NOD_POS))
        self._nod_pos = value

    @property
    def object_name(self):
        return self._object_name

    @object_name.setter
    def object_name(self, value):
        self._object_name = str(value)

    @property
    def grating_setting(self):
        return self._grating_setting

    @grating_setting.setter
    def grating_setting(self, value):
        self._grating_setting = str(value)

    @property
    def t_mid_jd(self):
        return self._t_mid_jd

    @t_mid_jd.setter
    def t_mid_jd(self, value):
        self._t_mid_jd = float(value)

    @property
    def t_end_jd(self):
        return self._t_end_jd

    @t_end_jd.setter
    def t_end_jd(self, value):
        self._t_end_jd = float(value)

    @property
    def n_detectors(self):
        return self._n_detectors

    @n_detectors.setter
    def n_detectors(self, value):
        self._n_detectors = int(value)

    @property
    def n_orders(self):
        return self._n_orders

    @n_orders.setter
    def n_orders(self, value):
        self._n_orders = int(value)

    @property
    def min_order(self):
        return self._min_order

    @min_order.setter
    def min_order(self, value):
        self._min_order = int(value)

    @property
    def max_order(self):
        return self._max_order

    @max_order.setter
    def max_order(self, value):
        self._max_order = int(value)


    def update_t_mid_and_end(self,):
        """Called to update the mid and end JD time of the observation."""
        self.t_mid_jd = self.t_start_jd + (self.t_exp_sec / 3600 / 24)/2
        self.t_end_jd = self.t_start_jd + (self.t_exp_sec / 3600 / 24)


    def save_to_fits(self):
        """
        """
        pass

    
    def __str__(self):
        """String representation of Observation details."""
        info_str = (
            "n spectra: {:0.0f}, ".format(len(self.spectra_1d)),
            "grating: {}, ".format(self.grating_setting),
            "object: {}, ".format(self.object_name),
            "date: {}, ".format(self.t_start_str),
            "exp: {} sec, ".format(self.t_exp_sec),
            "nod pos: {}".format(self.nod_pos),
        )

        return "".join(info_str)


    def info(self):
        """Gives an overview of the observation."""
        print(str(self))

        # Print out spectral segments in wavelength order
        wl_starts = [spec.wave[0] for spec in self.spectra_1d]
        wl_indices = np.argsort(wl_starts)

        for seg_i in wl_indices:
            print("\t", str(self.spectra_1d[seg_i]))


    def plot_spectra(
        self,
        do_close_plots=True,
        do_normalise=False,
        fig=None,
        axis=None,):
        """Quickly plot spectra as a function of wavelength for inspection.
        """
        # Plot sequence of spectra
        if do_close_plots:
            plt.close("all")

        # Make a new subplot if we haven't been given a set of axes
        if fig is None and axis is None:
            fig, axis = plt.subplots(figsize=(12,4))

        # Loop over spectra array and plot each spectral segment
        for spectrum in self.spectra_1d:
            # Plot normalised spectra if requested
            if do_normalise:
                norm_fac = np.nanmedian(spectrum.flux)
            else:
                norm_fac = 1

            axis.plot(
                spectrum.wave,
                spectrum.flux / norm_fac,
                linewidth=0.5,)

        axis.set_xlabel(r"Wavelength ($\mu$m)")
        axis.set_ylabel("Flux")

        fig.tight_layout()


def initialise_observation_from_crires_nodding_fits(
    fits_file_nodding_extracted,
    #fits_file_blaze_corrected,
    #fits_file_telluric_corrected,
    fits_ext_names=("CHIP1.INT1", "CHIP2.INT1", "CHIP3.INT1"),
    initialise_empty_bad_px_mask=True,):
    """Takes a CRIRES+ 1D extracted nodding fits file, and initialises an
    Observation object containing a set of Spectra1D objects.

    Parameters
    ----------
    fits_file_nodding_extracted: string
        Filepath to CRIRES+ 1D extracted nodding fits file.

    fits_ext_names: string array, 
        default: ("CHIP1.INT1", "CHIP2.INT1", "CHIP3.INT1")
        HDU names for each chip. Defaults to CRIRES+ standard.

    initialise_empty_bad_px_mask: boolean, default: True
        Whether to initialise empty/uniniformative bad px masks.

    Returns
    -------
    observation: luciferase.spectra.Observation object
        An Observation object containing Spectra1D objects and associated info.
    """
    with fits.open(fits_file_nodding_extracted) as fits_file:
        # Intialise our list of spectra
        spectra_list = []
        
        # Determine the spectral orders to consider. TODO: get this from fits
        # headers in a neater way.
        columns = fits_file[fits_ext_names[0]].data.columns.names
        orders = list(set([int(cc.split("_")[0]) for cc in columns]))
        orders.sort()

        # Create a Spectrum1D object for each spectral order/detector combo
        for det_i, fits_ext in enumerate(fits_ext_names):
            hdu_data = fits_file[fits_ext].data

            for order in orders:
                wave = hdu_data["{:02.0f}_01_WL".format(order)]
                spec = hdu_data["{:02.0f}_01_SPEC".format(order)]
                e_spec = hdu_data["{:02.0f}_01_ERR".format(order)]

                if initialise_empty_bad_px_mask:
                    bad_px_mask = np.full(len(wave), False)
                else:
                    raise NotImplementedError(
                        "Error: currently can only use default bad px mask")

                spec_obj = Spectrum1D(
                    wave=wave,
                    flux=spec,
                    sigma=e_spec,
                    bad_px_mask=bad_px_mask,
                    detector_i=det_i+1,
                    order_i=order,
                )

                spectra_list.append(spec_obj)

        # Sort in wavelength order
        wave_sort_i = np.argsort([spec.wave[0] for spec in spectra_list])
        sorted_spec_list = list(np.array(spectra_list)[wave_sort_i])

        # Extract header keywords. TODO: properly get exptime
        exp_time_sec = fits_file[0].header["HIERARCH ESO DET SEQ1 EXPTIME"]
        t_start_str = fits_file[0].header["DATE-OBS"]
        t_start_jd = fits_file[0].header["MJD-OBS"] + 2400000.5
        object_name = fits_file[0].header["OBJECT"]
        grating_setting = fits_file[0].header["HIERARCH ESO INS WLEN ID"]
        n_detectors = len(fits_file) - 1
        min_order = np.min(orders)
        max_order = np.max(order)

        # Only get nodpos if it exists
        if "HIERARCH ESO SEQ NODPOS" in fits_file[0].header:
            nod_pos = fits_file[0].header["HIERARCH ESO SEQ NODPOS"]
        else:
            nod_pos = None

        # Create Observation object
        observation = Observation(
            spectra_1d=sorted_spec_list,
            t_exp_sec=exp_time_sec,
            t_start_str=t_start_str,
            t_start_jd=t_start_jd,
            nod_pos=nod_pos,
            object_name=object_name,
            grating_setting=grating_setting,
            min_order=min_order,
            max_order=max_order,
            n_detectors=n_detectors,)
            
    return observation


def load_saved_observation_obj():
    """
    """
    pass