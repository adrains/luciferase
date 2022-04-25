"""Objects and functions for working with spectra.
"""
import os
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import luciferase.reduction as lred
from numpy.polynomial.polynomial import Polynomial, polyfit

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

    seeing_arcsec: float
        The measured seeing for this observation ('seeing' here being inclusive
        of AO).
    
    slit_func: float array
        1D slit function across the slit for the observation. For CRIRES 
        observations, it is from this that the seeing is measured.

    continuum_wavelengths: float array
        Wavelength values that have been identified as continuum points for
        doing a continuum normalisation. Note that these wavelength values
        should be in the reference frame of the star.
    """
    def __init__(
        self,
        wave,
        flux,
        sigma,
        bad_px_mask,
        detector_i,
        order_i,
        seeing_arcsec,
        slit_func,
        continuum_wls,
        is_continuum_normalised=False,):
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
        self.seeing_arcsec = seeing_arcsec
        self.slit_func = slit_func
        self.continuum_wls = continuum_wls
        self.is_continuum_normalised = is_continuum_normalised

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

    @property
    def seeing_arcsec(self):
        return self._seeing_arcsec

    @seeing_arcsec.setter
    def seeing_arcsec(self, value):
        self._seeing_arcsec = float(value)

    @property
    def slit_func(self):
        return self._slit_func

    @slit_func.setter
    def slit_func(self, value):
        self._slit_func = np.array(value)

    @property
    def continuum_wls(self):
        return self._continuum_wls

    @continuum_wls.setter
    def continuum_wls(self, value):
        self._continuum_wls = np.array(value)

    @property
    def is_continuum_normalised(self):
        return self._is_continuum_normalised

    @is_continuum_normalised.setter
    def is_continuum_normalised(self, value):
        self._is_continuum_normalised =bool(value)

    @property
    def continuum_poly(self):
        return self._continuum_poly

    @continuum_poly.setter
    def continuum_poly(self, value):
        if type(value) == np.polynomial.polynomial.Polynomial:
            self._continuum_poly = value
        else:
            raise ValueError("Expecting np.polynomial.polynomial.Polynomial.")

    @property
    def continuum_pixels(self):
        return self._continuum_pixels

    @continuum_pixels.setter
    def continuum_pixels(self, value):
        self._continuum_pixels = np.array(value)

    def update_snr(self):
        """Simple function to recompute SNR assuming Poisson uncertainties."""
        try:
            with np.seterr(all="ignore"):
                bad_px_mask = ~np.isfinite(self.flux)
                self.snr = np.nanmedian(
                    (self.flux[bad_px_mask] 
                    / np.sqrt(np.nanmedian(self.flux[bad_px_mask])))
                )

        # In case this fails, just set the SNR to zero
        except:
            self.snr = 0

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


    def do_continuum_normalise(
        self,
        region_width_px=2,
        poly_order=1,
        continuum_region_func="MEDIAN",):
        """Continuum normalise this spectrum (in place) using a polynomial and 
        the continuum points specified. Updates the is_continuum_normalised
        flag afterwards.

        Parameters
        ----------
        region_width_px: int,  default: 2
            A given continuum point is specified as the median value in the 
            +/-region_width_px range either side of the wavelength specified.

        poly_order: int, default: 1
            Order of the polynomial to fit to the continuum. 

        continuum_region_func: string, default: 'median'
            Function to use to determine flux for continuum region, valid 
            options are: [MEDIAN, MEAN, MAX]
        """
        VALID_OPTIONS = ["MEDIAN", "MEAN", "MAX",]

        if continuum_region_func.upper() not in VALID_OPTIONS:
            raise ValueError("Option must be in {}".format(VALID_OPTIONS))
        
        if continuum_region_func.upper() == "MEDIAN":
            cont_func = np.nanmedian
        elif continuum_region_func.upper() == "MEAN":
            cont_func = np.nanmean
        elif continuum_region_func.upper() == "MAX":
            cont_func = np.nanmax

        # Initialise a handy descriptor for this spectrum for printing
        print_label = "(order:{:0.0f}, det:{:0.0f})".format(
            self.order_i, self.detector_i)

        # First check that we haven't already continuum normalised the data
        if self.is_continuum_normalised:
            print("Spectrum ({}) is already continuum normalised!".format(
                print_label))
            return

        # Only do this if we have continuum wavelengths appropriate for our
        # polynomial order
        if len(self.continuum_wls) < (poly_order + 1):
            print("Insufficient continuum points for spectrum ({}).".format(
                print_label))
            return

        # Find the closest pixel for each continuum wavelength, and then find
        # the median flux value within +/-region_width_px of this pixel.
        cont_region_wavelengths = []
        cont_region_fluxes = []
        continuum_pixels = []

        for cont_wl in self.continuum_wls:
            # Find closest pixel
            px_i = np.argmin(np.abs(self.wave - cont_wl))

            # Save continuum pixels for reference
            continuum_pixels.append(px_i)

            # Save the wavelength of this pixel
            cont_region_wavelengths.append(self.wave[px_i])

            # Find median flux for the region around this pixel
            reg_min = px_i - region_width_px
            reg_max = px_i + region_width_px

            cont_px_flux = cont_func(self.flux[reg_min:reg_max])

            cont_region_fluxes.append(cont_px_flux)

        # Now dow polynomial fit to just these flux values
        # TODO: include flux uncertainties
        coef = polyfit(
            x=cont_region_wavelengths,
            y=cont_region_fluxes,
            deg=poly_order,)

        # Save the polynomial and continuum pixels
        self.continuum_poly = Polynomial(coef)
        self.continuum_pixels = continuum_pixels

        # Normalise by continuum and update our flux and uncertainties
        continuum = self.continuum_poly(self.wave)

        self.flux = self.flux / continuum
        self.sigma = self.sigma / continuum

        # Update flag
        self.is_continuum_normalised = True
    

    def plot_spectrum(
        self,
        do_close_plots=True,
        fig=None,
        axis=None,):
        """Quickly plot spectra as a function of wavelength for inspection.

        Parameters
        ----------
        do_close_plots: boolean, default: True
            Whether to close previously open plots. Set to False if passing in
            figure and axes objects.

        fig: matplotlib.figure.Figure, default: None
            Matplotlib figure object.

        axis: matplotlib.axes._subplots.AxesSubplot, default: None
            Matplotlib axis object.
        """
        # Plot sequence of spectra
        if do_close_plots:
            plt.close("all")

        # Make a new subplot if we haven't been given a set of axes
        if fig is None and axis is None:
            fig, axis = plt.subplots(figsize=(12,4))

        # Plot normalised spectra
        axis.plot(
            self.wave,
            self.flux / np.nanmedian(self.flux),
            linewidth=0.5,)

        # Plot segment details
        axis.text(
            x=self.wave[self.n_px//2],
            y=0.1,
            s="Order: {:0.0f}, Detector: {:0.0f}".format(
                self.order_i, self.detector_i),
            horizontalalignment="center",
            fontsize="small",)

        axis.set_xlabel(r"Wavelength ($\mu$m)")
        axis.set_ylabel("Flux")


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
        n_detectors,
        n_orders,
        spectra_1d_blaze_corr=None,
        spectra_1d_telluric_corr=None,):
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
        self.n_detectors = n_detectors
        self.n_orders = n_orders
        #self.spectra_1d_blaze_corr = spectra_1d_blaze_corr
        #self.spectra_1d_telluric_corr = spectra_1d_telluric_corr

        # Calculate t_mid_jd and t_end_jd. TODO: make this automatic.
        self.update_t_mid_and_end()

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
            "n orders: {:0.0f}, ".format(self.n_orders),
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
        axis=None,
        plot_continuum_poly=False,):
        """Quickly plot all spectra in spectra_1d as a function of wavelength
        for inspection.

        Parameters
        ----------
        do_close_plots: boolean, default: True
            Whether to close previously open plots. Set to False if passing in
            figure and axes objects.

        do_normalise: boolean, default: False
            Whether to normalise each spectrum object by its median value 
            before plotting.

        fig: matplotlib.figure.Figure, default: None
            Matplotlib figure object.

        axis: matplotlib.axes._subplots.AxesSubplot, default: None
            Matplotlib axis object.
        
        plot_continuum_poly: boolean, default: False
            Whether to plot our polynomial fit to the spectral continuum.
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

            # If we're plotting the continuum polynomial fit, make sure to
            # undo the continuum normalisation before plotting
            if plot_continuum_poly:
                # Get continuum fit
                continuum = spectrum.continuum_poly(spectrum.wave)

                # Plot spectra
                axis.plot(
                    spectrum.wave,
                    spectrum.flux * continuum,
                    linewidth=0.5,)

                # Plot continuum fit
                axis.plot(
                    spectrum.wave,
                    continuum,
                    linewidth=0.5,
                    color="r",)

                # Plot continuum points used for fit
                # TODO: use actual flux values here.
                axis.plot(
                    spectrum.wave[spectrum.continuum_pixels],
                    continuum[spectrum.continuum_pixels],
                    marker="x",
                    markeredgecolor="k",
                    linestyle="",)

            # Otherwise just plot spectra as is
            else:
                axis.plot(
                    spectrum.wave,
                    spectrum.flux / norm_fac,
                    linewidth=0.5,)

        axis.set_xlabel(r"Wavelength ($\mu$m)")
        axis.set_ylabel("Flux")

        fig.tight_layout()

    
    def continuum_normalise_spectra(
        self,
        region_width_px=2,
        poly_order=1,
        continuum_region_func="MEDIAN",
        do_plot=False,):
        """Continuum normalise all spectra in spectra_1d.

        Parameters
        ----------
        region_width_px: int,  default: 2
            A given continuum point is specified as the median value in the 
            +/-region_width_px range either side of the wavelength specified.

        poly_order: int, default: 1
            Order of the polynomial to fit to the continuum. 

        continuum_region_func: string, default: 'median'
            Function to use to determine flux for continuum region, valid 
            options are: [MEDIAN, MEAN, MAX]

        do_plot: boolean, default: False
            Whether to plot a diagnostic plot after fitting the continuum.
        """
        for spectrum in self.spectra_1d:
            spectrum.do_continuum_normalise(
                region_width_px=region_width_px,
                poly_order=poly_order,
                continuum_region_func=continuum_region_func,)

        # TODO: plot a more detail diagnostic plot
        if do_plot:
            self.plot_spectra(plot_continuum_poly=True)
    

    def plot_continuum_diagnostic(self,):
        """
        """
        pass


    def identify_continuum_regions(
        self,
        n_cont_region_per_spec=7,
        timeout_sec=10,
        return_wls=False):
        """For every spectral segment, take user input for the continuum 
        region locations. We prompt the user for *up to* n_cont_region_per_spec
        continuum points. If there are fewer continuum points than this, the 
        user can simply let timeout_sec pass to use fewer than this and move
        onto the next plot.

        Parameters
        ----------
        n_cont_region_per_spec: int, default: 7
            Maximum number of continuum points to prompt the user for.

        timeout_sec: int, default: 10
            How long to give the user in seconds to select the next point.

        return_wls: bool, default: False
            Whether to return the continuum wavelengths as an array.

        Returns
        -------
        cont_regions: numpy array [optional]
            A sorted 1D array of the selected continuum wavelengths.
        """
        cont_regions = []

        title_text = ("Click to select up to {:0.0f} continuum regions. If "
                      "less than this are to be selected, wait {:0.0f} sec for"
                      " timeout.")
        title_text = title_text.format(n_cont_region_per_spec, timeout_sec)

        # For each spectrum segment, ask for user input for the continuum
        # regions.
        for spec_i, spectrum in enumerate(self.spectra_1d):
            spectrum.plot_spectrum()
            plt.title(title_text)
            user_coords = plt.ginput(
                n=n_cont_region_per_spec,
                timeout=timeout_sec,)

            # We only care about the X values of these coordinates
            continuum_wavelengths = np.array(user_coords)[:,0]

            # Save the continuum regions for the spectrum
            spectrum.continuum_wls = continuum_wavelengths

            cont_regions.append(continuum_wavelengths)

        plt.close()

        if return_wls:
            # Convert to 1D numpy array and sort
            cont_regions = np.concatenate(np.array(cont_regions))
            sorted_i = np.argsort(cont_regions)
            cont_regions = cont_regions[sorted_i]

            return cont_regions


    def save_continuum_wavelengths_to_file(
        self,
        save_path="data/",):
        """Get continuum wavelengths from spectra opjects, combine, and save to
        file. Counterpart to load_continuum_wavelengths_from_file. File is 
        saved as continuum_wavelengths_[date]_[object].txt.

        Parameters
        ----------
        save_path: string, default: 'data/'
            Default relative filepath to save the continuum wavelengths to.
        """
        # Get the stored continuum wavelengths for each spectrum in spectrum_1d
        # and sort.
        continuum_wavelengths = []

        for spectrum in self.spectra_1d:
            continuum_wavelengths.append(spectrum.continuum_wls)

        continuum_wavelengths = np.concatenate(continuum_wavelengths)
        sorted_i = np.argsort(continuum_wavelengths)
        continuum_wavelengths = continuum_wavelengths[sorted_i]
        
        if len(continuum_wavelengths) < 1:
            print("No continuum wavelengths!")
            return
        
        # Save this list to disk
        cont_wavelengths_fn = "continuum_wavelengths_{}_{}.txt".format(
            self.t_start_str.split("T")[0], self.object_name.replace(" ", "_"))
        
        np.savetxt(
            fname=os.path.join(save_path, cont_wavelengths_fn),
            X=continuum_wavelengths,
            delimiter=",",)


    def load_continuum_wavelengths_from_file(
        self,
        load_path="data/",
        cont_wavelengths_fn=None,):
        """Load a previously saved set of continuum wavelengths and save to 
        spectra objects. Counterpart to save_continuum_wavelengths_to_file.

        Parameters
        ----------
        load_path: string, default: 'data/'
            Default relative filepath to load the continuum wavelengths from.

        cont_wavelengths_fn: string, default: None
            Specific filename to import. If None will use:
            continuum_wavelengths_[date]_[object].txt.
        """
        # If we've not been provided a filename construct the standard filename
        # from the object and date
        if cont_wavelengths_fn is None:
            cont_wavelengths_fn = "continuum_wavelengths_{}_{}.txt".format(
                self.t_start_str.split("T")[0], 
                self.object_name.replace(" ", "_"))
            
        cont_wavelengths_path = os.path.join(
                load_path, cont_wavelengths_fn)

        # Load in continuum wavelengths from file
        continuum_wavelengths = np.loadtxt(cont_wavelengths_path)

        # Assign the appropriate wavelengths to each spectral segment
        for spectrum in self.spectra_1d:
            valid_wls = np.logical_and(
                continuum_wavelengths > spectrum.wave.min(),
                continuum_wavelengths < spectrum.wave.max(),)

            spectrum.continuum_wls = continuum_wavelengths[valid_wls]


def initialise_observation_from_crires_nodding_fits(
    fits_file_nodding_extracted,
    fits_file_slit_func=None,
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

    fits_file_slit_func: string, default: None
        Filepath to CRIRES+ slit function for reduced data. Optional.

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
        # Intialise our list of spectra and slit_funcs
        spectra_list = []
        
        # Determine the spectral orders to consider. Note that not all 
        # detectors will necessarily have all orders, so we should pool the 
        # orders from all detectors, and then check before pulling from each.
        # TODO: get this from fits headers in a neater way. 
        columns = []

        for fits_ext_name in fits_ext_names:
            columns += fits_file[fits_ext_name].data.columns.names

        orders = list(set([int(cc.split("_")[0]) for cc in columns]))
        orders.sort()

        # Create a Spectrum1D object for each spectral order/detector combo
        for det_i, fits_ext in enumerate(fits_ext_names):
            hdu_data = fits_file[fits_ext].data

            for order in orders:
                # First check this order exists for this detector
                if ("{:02.0f}_01_WL".format(order) 
                    not in hdu_data.columns.names):
                    continue

                wave = hdu_data["{:02.0f}_01_WL".format(order)]
                spec = hdu_data["{:02.0f}_01_SPEC".format(order)]
                e_spec = hdu_data["{:02.0f}_01_ERR".format(order)]

                if initialise_empty_bad_px_mask:
                    bad_px_mask = np.full(len(wave), False)
                else:
                    raise NotImplementedError(
                        "Error: currently can only use default bad px mask")

                # Check if we have been given the slit function file, and if
                # so use it to compute the seeing for our current order
                if fits_file_slit_func is not None:
                    with fits.open(fits_file_slit_func) as slit_func_fits: 
                        sf_hdu = slit_func_fits[fits_ext].data
                        sf_ext_name = "{:02.0f}_01_SLIT_FUNC".format(order)
                        slit_func = sf_hdu[sf_ext_name]
                    fwhm = lred.fit_seeing_to_slit_func(
                        slit_func=slit_func,)
                else:
                    fwhm = np.nan
                    slit_func = np.nan

                # TODO allow importing continuum points
                continuum_wls = []

                spec_obj = Spectrum1D(
                    wave=wave,
                    flux=spec,
                    sigma=e_spec,
                    bad_px_mask=bad_px_mask,
                    detector_i=det_i+1,
                    order_i=order,
                    seeing_arcsec=fwhm,
                    slit_func=slit_func,
                    continuum_wls=continuum_wls,
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
        max_order = np.max(orders)

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
            n_detectors=n_detectors,
            n_orders=len(orders),)
            
    return observation


def load_saved_observation_obj():
    """
    """
    pass