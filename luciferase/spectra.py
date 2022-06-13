"""Objects and functions for working with spectra.
"""
import os
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
import astropy.constants as const
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import luciferase.reduction as lred
import luciferase.utils as lutils
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
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

        # Initialise hidden variables to keep track of unnormalised flux
        self._unnormalised_flux = np.array([])
        self._unnormalised_sigma =np.array([])

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
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                # Compute the SNR
                bad_px_mask = ~np.isfinite(self.flux)
                self.snr = np.nanmedian(
                    (self.flux[~bad_px_mask] 
                    / np.sqrt(np.nanmedian(self.flux[~bad_px_mask])))
                )

                # Default to 0 in case of error
                if np.isnan(self.snr):
                    self.snr = 0

        # In case this fails, just set the SNR to zero
        except:
            self.snr = 0

    def __str__(self):
        """String representation of Spectrum1D details."""
        info_str = (
            "npx {:0.0f}, ".format(self.n_px),
            "lambda mid {:0.1f} um, ".format(np.nanmean(self.wave)),
            "snr {:0.0f}".format(self.snr),
        )
        return "".join(info_str)


class ObservedSpectrum(Spectrum1D):
    """Class to represent an observed stellar spectrum. Inherits from 
    Spectrum1D.

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
        # Initialise basic parameters with superclass constructor
        super(ObservedSpectrum, self).__init__(
            wave, flux, sigma, bad_px_mask, detector_i, order_i,)
        
        # Now initialise everything else
        self.seeing_arcsec = seeing_arcsec
        self.slit_func = slit_func
        self.continuum_wls = continuum_wls
        self.is_continuum_normalised = is_continuum_normalised

        # Initialise default uniform 1D polynomial (y intercept 1, gradient 0) 
        # for continuum normalisation.
        self.continuum_poly = Polynomial([1,0])

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


    def fit_polynomial_to_continuum_wavelengths(
        self,
        region_width_px=2,
        poly_order=1,
        continuum_region_func="MEDIAN",):
        """Fits a polynomial to the specified continuum points and saves the
        polynomial coefficients.

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


    def optimise_continuum_polynomial_with_telluric_model(
        self,
        telluric_model_spectrum,):
        """Uses a telluric model spectrum from Molecfit to optimise the 
        polynomial coefficients used for continuum normalisation. This is 
        possible as different physical process contribute to the stellar and
        telluric spectra, allowing them to be modelled/treated separately. This
        function should be used in an interative way to converge on the optimal
        continuum normalisation. 

        TODO: mask out strong stellar lines and regions where the telluric
        model has a transmission of 1.0.

        Parameters
        ----------
        telluric_model_spectrum: luciferase.spectra.TelluricSpectrum
            Modelled telluric spectra corresponding to this spectrum. 
        """
        # Start with the initial guess of the coefficients from our 
        # previously fitted linear polynomial
        params_init = self.continuum_poly.coef

        # Mask out nans and infs with default values
        sci_flux = self._unnormalised_flux.copy()
        sci_sigma = self._unnormalised_sigma.copy()

        bad_px_mask = np.logical_or(np.isnan(sci_flux), np.isnan(sci_sigma))

        sci_flux[bad_px_mask] = 1
        sci_sigma[bad_px_mask] = 1E20

        # Setup the list of parameters to pass to our fitting function
        args = (
            self.wave,
            sci_flux,
            sci_sigma,
            telluric_model_spectrum.flux,)

        # Do fit
        ls_fit_dict = least_squares(
            calc_continuum_optimisation_resid, 
            params_init, 
            jac="3-point",
            args=args,)

        # TODO: Diagnostics
        pass

        # Update continuum polynomial with the new best fit coefficients
        self.continuum_poly = Polynomial(ls_fit_dict["x"])


    def do_continuum_normalise(self):
        """Continuum normalise this spectrum using our pre-computed polynomial 
        (initially from fit_polynomial_to_continuum_wavelengths, but later
        also from optimise_continuum_polynomial_with_telluric_model). Updates
        the is_continuum_normalised flag afterwards. To undo, use
        undo_continuum_normalise.
        """
        # Check we haven't already done the continuum normalisation
        if self.is_continuum_normalised:
            print("Already continuum normalised!")
            return

        # Store the unnormalised flux and uncertainty
        self._unnormalised_flux = self.flux.copy()
        self._unnormalised_sigma = self.sigma.copy()

        # Normalise by continuum and update our flux and uncertainties
        continuum = self.continuum_poly(self.wave)

        self.flux = self.flux / continuum
        self.sigma = self.sigma / continuum

        # Set infs to nan
        inf_mask = np.logical_or(np.isinf(self.flux), np.isinf(self.sigma))
        self.flux[inf_mask] = np.nan
        self.sigma[inf_mask] = np.nan

        # Update flag
        self.is_continuum_normalised = True


    def undo_continuum_normalise(self):
        """Counterpart function to do_continuum_normalise to restore the 
        spectrum to its unnormalised state.
        """
        # Only proceed if our spectra is continuum normalised
        if self.is_continuum_normalised:
            self.flux = self._unnormalised_flux.copy()
            self.sigma = self._unnormalised_sigma.copy()
            self.is_continuum_normalised = False
        else:
            print("Spectrum not yet normalised!")


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
            fig, axis = plt.subplots(figsize=(16,5))
            plt.subplots_adjust(left=0.05, right=0.95,)

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

        axis.set_xlim(self.wave[0], self.wave[-1])
        axis.set_xlabel(r"Wavelength ($\mu$m)")
        axis.set_ylabel("Flux")


class TelluricSpectrum(Spectrum1D):
    """Class to hold a modelled telluric spectrum. Inherits from Spectrum1D.

    The molecfit file used to prepare this object has the following columns:
     - chip:    science spectral segment #
     - lambda:  science wavelength scale
     - flux:    science fluxes
     - weight:  science flux weights. Per src/mf_readspec.c (in the molecfit
                source code), the weights are simply the inverse variance 
                (i.e. 1/sigma).
     - mrange:  model spectral segment #
     - mlambda: model wavelength scale
     - mscal:   model continuum scaling factor
     - mflux:   best fit model telluric correction
     - mweight: model flux weights (i.e. the inverse variance)
     - dev:     weighted difference between model and observed spectrum (per
                the description in the header of mf_molecfit_writefile in
                src/mf_molecfit.c)
     - mtrans:  model transmission curve (for telluric features in absorption).
                note that this *should* be equal to mflux in the absence of
                molecfit performing its own continuum fit

    Of these, we take mlambda --> wave, mtrans --> flux, 1/mweight --> sigma, 
    dev --> model_weighted_residuals, and mscal --> model_continuum_scaling.

    Parameters
    ----------
    wave, flux, sigma: float array
        Wavelength, flux, and uncertainty vectors for the spectrum.

    bad_px_mask: boolean array
        Bad pixel mask for the spectrum. True corresponds to bad pixels.

    detector_i, order_i: int
        Integer label for the detector and order assocated with the spectrum.

    model_weighted_residuals: float array
        Molecfit calculated weighted residuals on telluric fit.

    model_continuum_scaling: float array
        Scaling array to convert between atmospheric transmission (normalised
        to 1) of telluric features and the fitted spectra. Note that this will
        be an array of 1s if Molecfit did not do its own continuum fitting. 
    """
    def __init__(
        self,
        wave,
        flux,
        sigma,
        bad_px_mask,
        detector_i,
        order_i,
        model_weighted_residuals,
        model_continuum_scaling,):
        """
        """
        # Initialise basic parameters with superclass constructor
        super(TelluricSpectrum, self).__init__(
            wave, flux, sigma, bad_px_mask, detector_i, order_i,)
        
        # Now initialise everything else
        self.model_weighted_residuals = model_weighted_residuals
        self.model_continuum_scaling = model_continuum_scaling
        
    @property
    def model_weighted_residuals(self):
        return self._model_weighted_residuals

    @model_weighted_residuals.setter
    def model_weighted_residuals(self, value):
        # Check dimensions
        if len(value) != self.n_px:
            raise ValueError("Error, length of array != n_px.")
        else:
            self._model_weighted_residuals = np.array(value)

    @property
    def model_continuum_scaling(self):
        return self._model_continuum_scaling

    @model_continuum_scaling.setter
    def model_continuum_scaling(self, value):
        # Check dimensions
        if len(value) != self.n_px:
            raise ValueError("Error, length of array != n_px.")
        else:
            self._model_continuum_scaling = np.array(value)


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

    n_detectors: int
        Number of detectors.

    n_orders: int
        Number of spectral orders. Note that not all orders necessarily need to
        have flux on all detectors.
    
    fits_file: string
        Source fits file for the data.

    bcor: float
        Barycentric velocity in km/s.

    rv, e_rv: float
        Radial velocity and radial velocity uncertainty of the star in km/s.

    rv_template_interpolator: scipy.interpolate.interpolate.interp1d object
        Interpolation object used to determine RV.

    spectra_1d_telluric_model: luciferase.spectra.TelluricSpectrum array
        Modelled telluric spectra corresponding to spectrum_1d. Defaults to []
        until assigned with initialise_molecfit_best_fit_model.
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
        fits_file,
        bcor,
        rv,
        e_rv,
        spectra_1d_telluric_model=[],
        rv_template_interpolator=None,
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
        self.fits_file = fits_file
        self.bcor = bcor
        self.rv = rv
        self.e_rv = e_rv
        self.spectra_1d_telluric_model = spectra_1d_telluric_model
        self.rv_template_interpolator = rv_template_interpolator
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
            and not issubclass(type(spectrum), Spectrum1D)):
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

    @property
    def fits_file(self):
        return self._fits_file

    @fits_file.setter
    def fits_file(self, value):
        if os.path.isfile(value):
            self._fits_file = str(value)

    @property
    def bcor(self):
        return self._bcor

    @bcor.setter
    def bcor(self, value):
        self._bcor = float(value)

    @property
    def rv(self):
        return self._rv

    @rv.setter
    def rv(self, value):
        self._rv = float(value)

    @property
    def e_rv(self):
        return self._e_rv

    @e_rv.setter
    def e_rv(self, value):
        self._e_rv = float(value)

    @property
    def spectra_1d_telluric_model(self):
        return self._spectra_1d_telluric_model

    @spectra_1d_telluric_model.setter
    def spectra_1d_telluric_model(self, value):
        # Make sure the result is a list of Spectrum1D objects
        if type(value) != list:
            raise ValueError(
                "spectra_1d_telluric_model should be a list of spectra.")
        
        # Make sure the list of model telluric spectra is either empty or has
        # the same length as spectrum_1d
        if len(value) != 0 and len(value) != len(self.spectra_1d):
            raise ValueError(("Length of spectra_1d_telluric_model should be "
                "the same as that of spectrum_1d."))
        
        # Finally check to make sure each object in the list is either a 
        # TelluricSpectrum object, or is derived from it.
        for spec_i, spectrum in enumerate(value):
            if (type(spectrum) != TelluricSpectrum 
            and not issubclass(type(spectrum), TelluricSpectrum)):
                raise ValueError((
                    "Telluric Spectrum #{:0.0f} in spectra_1d_telluric_model "
                    "is not of type TelluricSpectrum or inherited from it."
                    ).format(spec_i))

        # Otherwise all good
        self._spectra_1d_telluric_model = value
    
    @property
    def rv_template_interpolator(self):
        return self._rv_template_interpolator

    @rv_template_interpolator.setter
    def rv_template_interpolator(self, value):
        if type(value) == interp1d or value is None:
            self._rv_template_interpolator = value
        else:
            raise ValueError("Invalid interpolator!")
    
    @property
    def wave_all(self):
        """Convenience function to access 1D array of wavelengths."""
        return np.hstack([spec.wave for spec in self.spectra_1d])

    @property
    def flux_all(self):
        """Convenience function to access 1D array of fluxes."""
        return np.hstack([spec.flux for spec in self.spectra_1d])

    @property
    def sigma_all(self):
        """Convenience function to access 1D array of sigmas."""
        return np.hstack([spec.sigma for spec in self.spectra_1d])


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
        plot_continuum_poly=False,
        line_list=None,
        line_depth_threshold=0.2,
        plot_molecfit_model=False,
        y_lim_median_fac=1.5,
        figsize=(16,4),
        do_save=False,
        save_folder="plots/",
        line_annotation_fontsize="xx-small",
        alternate_annotation_height=True,
        linewidth=0.4,):
        """Quickly plot all spectra in spectra_1d as a function of wavelength
        for inspection. Optionally can be saved as a pdf.

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

        line_list: pandas.DataFrame, default: None
            VALD line list, imported using the read_vald_linelist function. If 
            None, no lines are plotted.

        line_depth_threshold: float, default: 0.2
            Minimum strength of lines to plot. Smaller values are weaker lines.

        plot_molecfit_model: boolean, default: False
            Whether to overplot an imported best fit Molecfit model.

        y_lim_median_fac: float, default: 1.5
            Used to avoid edge pixels distorting the y scale--a median is 
            computed of all flux pixels, and the upper limit is this median
            multiplied by y_lim_median_fac.

        figsize: float tuple, default: (16,4)
            Size of the figure in inches, has format (x, y).

        do_save: boolean, default: False
            Whether to save the resulting plot as a pdf.

        save_folder: string, default: 'plots/'
            Directory to save output plots to as spectra_<objname>.pdf.

        line_annotation_fontsize: string, default: 'xx-small'
            Font size for atomic line annotations

        alternate_annotation_height: boolean, default: True
            Whether to alternate the heights of atomic line annotations (i.e. 
            short arrow, tall arrow, short arrow, tall arrow, etc) to better
            avoid overlapping text.

        linewidth: float, default: 0.4
            Linewidth for plotted spectra.
        """
        # Plot sequence of spectra
        if do_close_plots:
            plt.close("all")

        # Make a new subplot if we haven't been given a set of axes
        if fig is None and axis is None:
            fig, axis = plt.subplots(figsize=figsize)

        # Set y limits based on median flux
        fluxes = []
        continua = []

        # Loop over spectra array and plot each spectral segment
        for spec_i, spectrum in enumerate(self.spectra_1d):
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
                    spectrum._unnormalised_flux,
                    linewidth=linewidth,)

                # Plot continuum fit
                axis.plot(
                    spectrum.wave,
                    continuum,
                    linewidth=linewidth,
                    color="r",)

                # Plot continuum points used for fit
                # TODO: use actual flux values here.
                axis.plot(
                    spectrum.wave[spectrum.continuum_pixels],
                    (spectrum._unnormalised_flux)[spectrum.continuum_pixels],
                    marker="x",
                    markeredgecolor="k",
                    linestyle="",)

                continua.append(continuum)

            # Otherwise just plot spectra as is
            else:
                axis.plot(
                    spectrum.wave,
                    spectrum.flux / norm_fac,
                    linewidth=linewidth,)

            # Append to our fluxes array
            fluxes.append(spectrum.flux / norm_fac)

            # Plot the best fit molecfit model if a) we've been asked to, and
            # b) we actually have it imported.
            if len(self.spectra_1d_telluric_model) > 0 and plot_molecfit_model:
                axis.plot(
                    self.spectra_1d_telluric_model[spec_i].wave,
                    self.spectra_1d_telluric_model[spec_i].flux / norm_fac,
                    linewidth=linewidth,
                    alpha=0.9,
                    color="black",
                    linestyle="--",)

            # Now annote the line list if we've been asked to
            if line_list is not None:
                # Warn user if either RV or bcor is nan
                if np.isnan(self.rv) or np.isnan(self.bcor):
                    raise Warning("Warning: either bcor or RV is NaN.")
                # Shift the line to the rest frame of the star
                shift_fac = (1-(self.rv-self.bcor)/(const.c.si.value/1000))
                wl_new = line_list["WL_vac(nm)"].values * shift_fac

                # Plot only those lines above the depth threshold *and* within
                # the wavelength region we're considering
                depth_mask = line_list["depth"] > line_depth_threshold
                wl_mask = np.logical_and(
                    wl_new > np.nanmin(spectrum.wave),
                    line_list["WL_vac(nm)"] < np.nanmax(spectrum.wave),)
                line_mask = np.logical_and(depth_mask, wl_mask)

                point_height = 1.1 * np.nanmedian(spectrum.flux)
                text_height = 1.2 * point_height

                for species_i, (species, wl) in enumerate(zip(
                    line_list["SpecIon"][line_mask].values, 
                    wl_new[line_mask])):

                    if alternate_annotation_height and species_i % 2 == 0:
                        offset = (text_height - point_height)/4
                    else:
                        offset = 0

                    # Plot text and arrow for each line
                    axis.annotate(
                        text=species,
                        xy=(wl, point_height),
                        xytext=(wl, text_height + offset),
                        horizontalalignment="center",
                        fontsize=line_annotation_fontsize,
                        arrowprops=dict(arrowstyle='->',lw=0.2),)

        # Set height based on median fluxes
        if not plot_continuum_poly:
            fluxes = np.concatenate(fluxes)
            axis.set_ylim(-0.1, np.nanmedian(fluxes)*y_lim_median_fac)
        
        # Or set height based on continuum
        else:
            continua = np.concatenate(continua)
            axis.set_ylim([-0.1, np.nanmedian(continua)*y_lim_median_fac])

        axis.xaxis.set_minor_locator(plticker.MultipleLocator(base=25))
        axis.xaxis.set_major_locator(plticker.MultipleLocator(base=50))

        axis.set_xlabel(r"Wavelength (nm)")
        axis.set_ylabel("Flux")

        fig.tight_layout()

        # Save if asked
        if do_save:
            fig_name = "spectra_{}.pdf".format(
                self.object_name.replace(" ", ""))
            plt.savefig(os.path.join(save_folder, fig_name,))

    
    def fit_poly_to_spectra_continuum_wavelengths(
        self,
        region_width_px=2,
        poly_order=1,
        continuum_region_func="MEDIAN",):
        """Fits and saves a polynomial to the continuum wavelengths previously
        identified.

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
        for spectrum in self.spectra_1d:
            spectrum.fit_polynomial_to_continuum_wavelengths(
                region_width_px=region_width_px,
                poly_order=poly_order,
                continuum_region_func=continuum_region_func,)


    def continuum_normalise_spectra(
        self,
        do_plot=False,):
        """Continuum normalise all spectra in spectra_1d. Note that this 
        should be done *after* suitable polynomials have been fit.

        Parameters
        ----------
        do_plot: boolean, default: False
            Whether to plot a diagnostic plot after fitting the continuum.
        """
        for spectrum in self.spectra_1d:
            spectrum.do_continuum_normalise()

        # TODO: plot a more detail diagnostic plot
        if do_plot:
            self.plot_spectra(plot_continuum_poly=True)
    

    def undo_continuum_normalise_spectra(self,):
        """Counterpart to continuum_normalise_spectra to undo continuum 
        normalisation.
        """
        for spectrum in self.spectra_1d:
            spectrum.undo_continuum_normalise()


    def identify_continuum_regions(
        self,
        n_cont_region_per_spec=10,
        timeout_sec=20,
        return_wls=False,):
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

            # Set the Y limit based on the median to avoid edge pixel spikes
            #plt.ylim(0, 4)

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


    def optimise_continuum_fit_using_telluric_model(self,):
        """Optimises the continuum fit for all spectra in spectra_1d using the
        corresponding model telluric absorption spectrum.
        """
        # Only proceed if we have a telluric model and we've already done a 
        # manual continuum normalisation
        if len(self.spectra_1d_telluric_model) == 0:
            raise Exception("No telluric model!")

        # For every spectral segment, find best set of linear polynomial 
        # coefficients (i.e. gradient and Y offset) to divide science spectrum
        # by to match the telluric model
        for spec_i, (sci_spec, tell_spec) in enumerate(
            zip(self.spectra_1d, self.spectra_1d_telluric_model)):
            # Update the continuum polynomial
            sci_spec.optimise_continuum_polynomial_with_telluric_model(
                tell_spec)

            # Undo the previous continuum fit
            sci_spec.undo_continuum_normalise()

            # Now update the continuum fit itself
            sci_spec.do_continuum_normalise()


    def save_molecfit_fits_files(
        self,
        molecfit_path,
        only_one_spectral_segment=False,
        science_fits="SCIENCE.fits",
        wavelength_fits="WAVE_INCLUDE.fits",
        atmosphere_fits='MAPPING_ATMOSPHERIC.fits',
        pixel_fits="PIXEL_EXCLUDE.fits",
        edge_px_to_exlude=40,):
        """Function to write spectra and associated files in a molecfit 
        compatible format so that we can telluric correct our features.

        Note 1: if performing continuum normalisation here, the FIT_CONTINUUM
        parameter in model.rc should be set to 0.

        Note 2: this function only generates a basic pixel mask which masks out
        detector edge pixels. To also mask out strong science lines or produce
        diagnostic plots, run save_molecfit_pixel_exclude separately.

        TODO: split this function up so that there are separate functions for
        writing science files, wavelength files, atmosphere files, and pixel
        exclusion files (the latter of which is already done).

        Molecfit has the following steps, each of which has an associated SOF
        and rc file.
        1) Model
            SOF file (model.sof) takes as input:
                WAVE_INCLUDE.fits               WAVE_INCLUDE
                SCIENCE.fits                    SCIENCE
                PIXEL_EXCLUDE.fits              PIXEL_EXCLUDE
                WAVE_EXCLUDE.fits               WAVE_EXCLUDE

        2) Calctrans
            SOF file (calctrans.sof) takes as input:
                MODEL/MODEL_MOLECULES.fits      MODEL_MOLECULES
                MODEL/ATM_PARAMETERS.fits       ATM_PARAMETERS
                MODEL/BEST_FIT_PARAMETERS.fits  BEST_FIT_PARAMETERS
                MAPPING_ATMOSPHERIC.fits        MAPPING_ATMOSPHERIC
                SCIENCE.fits                    SCIENCE

        3) Correct
            SOF file (correct.sof) takes as input:
                CALCTRANS/TELLURIC_CORR.fits    TELLURIC_CORR
                SCIENCE.fits                    SCIENCE

        ESO documentation:
            http://www.eso.org/sci/software/pipelines/skytools/molecfit

        A&A Publication:
            https://ui.adsabs.harvard.edu/abs/2015A%26A...576A..77S/abstract
        
        Parameters
        ----------
        molecfit_path: string
            Filepath where we will save the output fits files.

        only_one_spectral_segment: boolean, default: False
            Currently unused.

        science_fits: string, default: 'SCIENCE.fits'
            Filename for output fits file containing science spectra.

        wavelength_fits: string, default: 'WAVE_INCLUDE.fits'
            Filename for output fits file containing wavelengths to include.

        atmosphere_fits: string, default: 'MAPPING_ATMOSPHERIC.fits'
            Filename for output fits file mapping 'SCIENCE' - 'ATM_PARAMETERS'.

        pixel_fits: string, default: 'PIXEL_EXCLUDE.fits'
            Filename for output fits file containing pixels to exclude.

        edge_px_to_exlude: int, default: 40
            Number of pixels on each side of each spectral segment to mask out
            when fitting molecfit.

        Output Files
        ------------
        1) SCIENCE.fits
            Science spectra consisting of primary HDU and N fits table HDUs 
            with columns ['WAVE', 'SPEC' 'ERR']. By default N is equal to the
            number of spectral segments, but N=1 if only_one_spectral_segment
            is set to True.

        2) WAVE_INCLUDE.fits
            Wavelengths to include in Molecfit modelling, plus other input
            parameters. Fits file consists of
            an empty primary HDU, and a single fits table HDU with columns
            ['LOWER_LIMIT', 'UPPER_LIMIT', 'MAPPED_TO_CHIP', 'CONT_FIT_FLAG',
             'CONT_POLY_ORDER', 'WLC_FIT_FLAG']. 
             
            The first two

        3) ATM_PARAMETERS_EXT.fits
            ATM_PARAMETERS_EXT file for Molecfit. File consists of an empty
            primary HDU, and then a single fits table HDU with the column 
            ['ATM_PARAMETERS_EXT'].

        4) PIXEL_EXCLUDE.fits
            Pixels to exclude in Molecfit modelling. Fits file consists of
            an empty primary HDU, and a single fits table HDU with columns
            ['LOWER_LIMIT', 'UPPER_LIMIT',].
        """
        # Setup save paths
        science_save_path = os.path.join(molecfit_path, science_fits)
        wavelength_save_path = os.path.join(molecfit_path, wavelength_fits)
        atmosphere_save_path = os.path.join(molecfit_path, atmosphere_fits)

        # Initialise HDU lists for each of our output fits files, assigning
        # the each primary HDU from our input science file.
        with fits.open(self.fits_file) as fits_file:
            primary_hdu = fits_file[0].copy()

        hdul_output = fits.HDUList([primary_hdu])
        hdul_winc = fits.HDUList([primary_hdu])
        hdul_atm =  fits.HDUList([primary_hdu])

        # Now initialise other arrays
        wmin, wmax  = [], []
        map2chip = []
        atm_parms_ext = []

        # Loop over every spectral segment
        for spec_i, spec in enumerate(self.spectra_1d):
            # Get the chip and spectral order indices
            chip_i = spec.detector_i
            order_i = spec.order_i

            # Create fits columns for WAVE, SPEC, and ERR
            col1 = fits.Column(name='WAVE', format='D', array=spec.wave)
            col2 = fits.Column(name='SPEC', format='D', array=spec.flux)
            col3 = fits.Column(name='ERR', format='D', array=spec.sigma)

            # Create fits table HDU with WAVE SPEC ERR
            table_hdu = fits.BinTableHDU.from_columns([col1, col2, col3])

            # Append table HDU to output HDU list
            hdul_output.append(table_hdu)

            # Append wmin for given order to wmin array and convert to micron
            wmin.append(np.min(spec.wave)*0.001)

            # Append wmax for giver order to wmax array and convert to micron
            wmax.append(np.max(spec.wave)*0.001)
            
            hdul_output[spec_i+1].header.append(
                ("EXTNAME",
                "det{:02.0f}_{:02.0f}_01".format(chip_i, order_i),
                "order/detector"),
                end=True)

            # Append order counter to map2chip
            map2chip.append(spec_i+1)

            # Update atmospheric parameter mapping: 1 for non-primary fits HDUs
            if spec_i+1 == 1:
                atm_parms_ext.append(0)
            else:
                atm_parms_ext.append(1)

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

        # Write to the science output file
        hdul_output.writeto(science_save_path, overwrite=True)

        # Write to WAVE_INCLUDE file
        hdul_winc.writeto(wavelength_save_path, overwrite=True)
        
        # Write atmospheric mapping output file
        col1_atm = fits.Column(
            name="ATM_PARAMETERS_EXT",
            format='I',
            array=atm_parms_ext)
        table_hdu_atm = fits.BinTableHDU.from_columns([col1_atm])
        hdul_atm.append(table_hdu_atm)
        hdul_atm.writeto(atmosphere_save_path, overwrite=True)

        # And finally save the basic pixel exclude file with only edge pixels.
        # To also mask out strong science lines, run this separately.
        self.save_molecfit_pixel_exclude(
            molecfit_path,
            pixel_fits=pixel_fits,
            edge_px_to_exlude=edge_px_to_exlude,
            do_science_line_masking=True,)


    def save_molecfit_pixel_exclude(
        self,
        molecfit_path,
        pixel_fits="PIXEL_EXCLUDE.fits",
        edge_px_to_exlude=40,
        do_science_line_masking=True,
        px_absorption_threshold=0.9,
        do_plot_exclusion_diagnostic=False,):
        """Writes the pixel exclude fits file for input to Molecfit. If we have
        a template spectrum interpolator, we will exclude strong science lines
        as well as the detector edges, otherwise just the latter.

        Parameters
        ----------
        molecfit_path: string
            Directory to save fits file.

        pixel_fits: string, default: 'PIXEL_EXCLUDE.fits'
            Name of the output pixel exclude fits file.

        edge_px_to_exclude: int, default: 40
            Pixels to exclude from each edge of the detector.

        do_science_line_masking: boolean, default: True
            If True, we will also exclude strong science lines in addition to
            excluding the detector edges.

        px_absorption_threshold: float, default: 0.9
            Continuum normalised pixels with flux values below this will be
            considered to belong to strong science lines and will be masked
            out during the Molecfit fit.

        do_plot_exclusion_diagnostic: boolean, default: False
            Whether to plot a diagnostic plot.

        Output Files
        ------------
        PIXEL_EXCLUDE.fits
            Pixels to exclude in Molecfit modelling. Fits file consists of
            an empty primary HDU, and a single fits table HDU with columns
            ['LOWER_LIMIT', 'UPPER_LIMIT',].
        """
        # Initialise HDU lists for our output fits file, assigning the primary 
        # HDU from our input science file.
        with fits.open(self.fits_file) as fits_file:
            primary_hdu = fits_file[0].copy()

        hdul_excl =  fits.HDUList([primary_hdu])

        # Initialise exclusion arrays
        px_excl_min, px_excl_max = [], []

        wave = self.wave_all
        flux = self.flux_all

        # Initialise the pixel exclusion array by first excluding all edge
        # pixels. While we could do this in a vectorised way, we'll use a loop
        # so that we don't have to assume that all segments are the same length
        px_exclude_mask_all = []

        for spec_i, spec in enumerate(self.spectra_1d):
            px_exclude_mask = np.full(spec.wave.shape, False)
            px_exclude_mask[:edge_px_to_exlude] = True
            px_exclude_mask[-edge_px_to_exlude:] = True

            px_exclude_mask_all.append(px_exclude_mask)

        px_exclude_mask_all = np.hstack(px_exclude_mask_all)

        # Now we want to also exclude all pixels with model absorption greater
        # than our threshold. The intent here is to mask out strong lines that
        # might confluse molecfit when it is fitting for telluric species.
        if (do_science_line_masking
            and self.rv_template_interpolator is not None):
            model_flux = self.rv_template_interpolator(
                wave * (1-(self.rv-self.bcor)/(const.c.si.value/1000)))
            px_exclude_mask_all = np.logical_or(
                px_exclude_mask_all,
                model_flux < px_absorption_threshold,)

        # Our mask is now complete, and we need to convert this to the molecfit
        # pixel exclusion format, which is a matched set of arrays 
        # corresponding to the lower and upper limits of the exclusion region.
        prev_px_excluded = False

        for px_i, px_exclude in enumerate(px_exclude_mask_all):
            # We've gotten to the end of the excluded region
            if (not px_exclude and prev_px_excluded
                or px_i+1 == len(px_exclude_mask_all) and prev_px_excluded):
                px_excl_min.append(px_i_min)
                px_excl_max.append(px_i_max)

                prev_px_excluded = False

            # Start of excluded region, but hold off adding to our arrays
            # Note: current assumption is that to mask a single pixel the max
            # and min values should be the same.
            elif px_exclude and not prev_px_excluded:
                px_i_min = px_i
                px_i_max = px_i 
                prev_px_excluded = True
            
            # Continue to storing, but update max
            elif px_exclude and prev_px_excluded:
                # Set max
                px_i_max = px_i
                continue
        # reset
        else:
            prev_px_excluded = False

        # Save the fits file
        pmin_excl = fits.Column(
            name='LOWER_LIMIT', format='K', array=px_excl_min)
        pmax_excl = fits.Column(
            name='UPPER_LIMIT', format='K', array=px_excl_max)

        pixel_save_path = os.path.join(molecfit_path, pixel_fits)

        table_hdu_pexcl = fits.BinTableHDU.from_columns([pmin_excl, pmax_excl])
        hdul_excl.append(table_hdu_pexcl)
        hdul_excl.writeto(pixel_save_path, overwrite=True)

        # Optionally plot a diagnostic
        if do_plot_exclusion_diagnostic:
            self.plot_spectra()

            # TODO: include this all in plotting function
            plt.plot(
                wave,
                model_flux,
                linewidth=0.5,
                color="black",
                linestyle="--",)

            for px_min, px_max in zip(px_excl_min, px_excl_max):
                plt.axvspan(
                    wave[px_min],
                    wave[px_max],
                    ymin=0, 
                    ymax=2,
                    alpha=0.4,
                    color="r",)


    def initialise_molecfit_best_fit_model(
        self,
        molecfit_model_fits_file,
        convert_um_to_nm=True,):
        """Imports and saves the best fit molecfit spectrum for each spectrum
        in specta_1d. Note that this assumes that the model molecfit spectrum 
        is sorted the same as the observed spectra (as we use it as a reference
        for order/detector numbering). 

        Parameters
        ----------
        molecfit_model_fits_file: string
            Filepath to the molecfit model fits file.

        convert_um_to_nm: boolean, default: True
            Whether the molecfit model wavelengths are in microns, and if so to
            convert them back to nm.
        """
        with fits.open(molecfit_model_fits_file) as model:
            # For convenience grab molecfit data
            model_chip_num = model[1].data["chip"]
            chip_num_unique = list(set(model_chip_num))
            chip_num_unique.sort()

            model_wave = model[1].data["mlambda"]

            # Convert wavelengths if necessary
            if convert_um_to_nm:
                model_wave *= 1000

            model_flux = model[1].data["mflux"]

            # Convert 1/var to sigma
            model_sigma = 1 / model[1].data["mweight"]**0.5

            model_weighted_residuals = model[1].data["dev"]
            model_cont_scaling = model[1].data["mscal"]

            # Initialise our output list of TelluricSpectra objects
            telluric_spectra_list = []

            # Loop over each chip number
            for chip_i, chip_num in enumerate(chip_num_unique):
                # Create each of our vectors, grabbing only the data for this 
                # particular order/detector segment
                wave = model_wave[model_chip_num == chip_num]
                flux = model_flux[model_chip_num == chip_num]
                sigma = model_sigma[model_chip_num == chip_num]
                resid = model_weighted_residuals[model_chip_num == chip_num]
                cont_scaling = model_cont_scaling[model_chip_num == chip_num]
                detector_i = self.spectra_1d[chip_i].detector_i
                order_i = self.spectra_1d[chip_i].order_i

                # Initialise empty bad_px_mask
                # TODO: handle this better. Does the base class need a 
                # # bad_px_mask?
                bad_px_mask = np.full(wave.shape, False)

                telluric_spec_obj = TelluricSpectrum(
                    wave=wave,
                    flux=flux,
                    sigma=sigma,
                    bad_px_mask=bad_px_mask,
                    detector_i=detector_i,
                    order_i=order_i,
                    model_weighted_residuals=resid,
                    model_continuum_scaling=cont_scaling,
                )

                # Append
                telluric_spectra_list.append(telluric_spec_obj)

            # Now attach to our observation object
            self.spectra_1d_telluric_model = telluric_spectra_list


    def load_molecfit_corrected_spectra(self,):
        """
        """
        pass


    def fit_rv(
        self,
        template_spectrum_fits,
        segment_contamination_threshold=0.95,
        ignore_segments=[],
        px_absorption_threshold=0.9,
        ls_diff_step=(0.1),
        rv_init_guess=0,
        rv_min=-200,
        rv_max=200,
        delta_rv=1,
        fit_method="CC",
        do_diagnostic_plots=False,
        verbose=False,
        figsize=(16,4),):
        """Function to fit the radial velocity globally for the observation
        object by doing a simultaneous fit using all spectral segments. Two
        different methods are implemented: a cross correlation and a least
        squares fit.

        TODO: Ideally there'd be a 'prepare for fitting' function, then 
        separate fitting functions (rather than a 'master' fitting function)
        which does the prep and then delegates out to subfunctions.

        Parameters
        ----------
        template_spectrum_fits: string
            Filepath to template spectrum to import for RV fitting.

        segment_contamination_threshold: float, default: 0.95
            If the median flux value for the model telluric transmission of a
            spectral segment is below this we consider the *entire segment* too
            contaminated for use when RV fitting. A value of 1 in this case 
            would be entirely continuum (and be the most restrictive), and a 
            value of 0 would be the most permissible.

        ignore_segments: int list, default: []
            List of spectral segments to ignore, where 0 is the first segment.

        px_absorption_threshold: float, default: 0.9
            Similar to segment_contamination_threshold, but it operates on a
            *per pixel* level on the model telluric spectrum rather than a per
            segment level.
            
        ls_diff_step: float tuple, default: (0.1)
            Input for least squares fitting, sets the step size in km/s.

        rv_init_guess: float, default: 0
            Initial guess for RV in km/s for use when least squares fitting.
            
        rv_min: float, default: -200
            Lower RV bound in km/s to apply to both cross correlation and least
            squares fitting.

        rv_max: float, default: +200
            Upper RV bound in km/s to apply to both cross correlation and least
            squares fitting.

        delta_rv: float, default 1
            RV step size for cross correlation scan in km/s.

        fit_method: string, default 'CC'
            Fit method, either 'CC' for cross correlation, or 'LS' for least
            squares fitting.

        do_diagnostic_plots: boolean, default: False
            Whether to plot diagnostic plots at the conclusion of the fit.

        verbose: boolean, default: False
            Whether to print updates on fitting.

        figsize: float tuple, default: (16,4)
            Figure size of diagnostic plot.
        """
        # Check the requested fitting method is valid
        fit_method = fit_method.upper()
        VALID_METHODS = ["CC", "LS"]

        if fit_method not in VALID_METHODS:
            raise ValueError("Invalid method. Must be in {}".format(
                VALID_METHODS))

        # Set the detault bad pixel value--cross correlation is happy with NaNs
        # but the least squares fitting breaks if it sees them.
        if fit_method == "CC":
            bad_px_value = np.nan
        elif fit_method == "LS":
            bad_px_value = 1

        # Intiialise pixel exclude mask for *all* spectral pixels. This will
        # be separated by segment (i.e. 2D rather than 1D), but we'll later
        # make a single continuous 1D array.
        pixel_exclude_mask = []

        # If we don't have a set of telluric model spectra, we're limited to
        # fitting without masking out the telluric features.
        if len(self.spectra_1d_telluric_model) != len(self._spectra_1d):
            print("No telluric model to use for masking, doing basic fit.")

            for seg_i, spec in enumerate(self.spectra_1d):
                # Mask out NaN, inf, or nonphysical pixels
                sci_nonfinite_mask = np.logical_or(
                        ~np.isfinite(spec.flux),
                        ~np.isfinite(spec.sigma))

                with np.errstate(invalid='ignore'):
                    sci_nonphysical_mask = np.logical_or(
                        spec.flux > 1.05,
                        spec.flux < 0,)

                exclude_mask = np.logical_or(
                    sci_nonphysical_mask, 
                    sci_nonfinite_mask)

                pixel_exclude_mask.append(exclude_mask)

                assert np.sum(np.isnan(spec.flux[~exclude_mask])) == 0

        # Otherise we can proceed with the smarter RV fit where we mask out any
        # segments that we're ignoring due to excessive telluric absorption. 
        # We'll do this by taking a median of the telluric model flux, and if 
        # this median is below our contamination threshold then we'll consider
        # the segment too contaminated to be useful for radial velocity fitting
        else:
            for seg_i, tspec in enumerate(self.spectra_1d_telluric_model):
                # If a segment is too contaminated to use, mask it out entirely
                if (np.nanmedian(tspec.flux) < segment_contamination_threshold
                    or seg_i in ignore_segments):
                    pixel_exclude_mask.append(np.full(tspec.flux.shape, True))

                # Otherwise only mask out nan pixels and those below the
                # telluric absorption threshold
                else:
                    # While we aren't excluding this entire segment, we'll 
                    # still exclude any pixels with *telluric* absorption below
                    # px_absorption_threshold or *science* pixels with nans,
                    # infs, or nonphysical spikes (e.g. detector edge effects)
                    telluric_absorb_mask = tspec.flux < px_absorption_threshold

                    sci_nonfinite_mask = np.logical_or(
                        ~np.isfinite(self.spectra_1d[seg_i].flux),
                        ~np.isfinite(self.spectra_1d[seg_i].sigma))

                    with np.errstate(invalid='ignore'):
                        sci_nonphysical_mask = np.logical_or(
                            self.spectra_1d[seg_i].flux > 1.05,
                            self.spectra_1d[seg_i].flux < 0,)

                    exclude_mask = np.logical_or(
                        telluric_absorb_mask,
                        np.logical_or(
                            sci_nonphysical_mask, 
                            sci_nonfinite_mask))

                    pixel_exclude_mask.append(exclude_mask)

                    assert np.sum(
                        np.isnan(
                            self.spectra_1d[seg_i].flux[~exclude_mask])) == 0

        pixel_exclude_mask = np.hstack(pixel_exclude_mask)

        # Stitch wavelengths, spectra, and sigmas together
        sci_wave = np.hstack([spec.wave for spec in self.spectra_1d])
        sci_flux = np.hstack([spec.flux for spec in self.spectra_1d])
        sci_sigma = np.hstack([spec.sigma for spec in self.spectra_1d])

        # Now mask out regions we're not considering
        sci_flux[pixel_exclude_mask] = bad_px_value
        sci_sigma[pixel_exclude_mask] = np.inf

        # Load in our template spectrum and the associated interpolator
        temp_wave, temp_flux = lutils.load_plumage_template_spectrum(
            template_spectrum_fits, do_convert_air_to_vacuum_wl=True,)
        
        calc_template_flux = interp1d(
            x=temp_wave,
            y=temp_flux,
            kind="linear",
            bounds_error=False,
            assume_sorted=True)

        # Store the interpolator
        self.rv_template_interpolator = calc_template_flux

        # Intialise output for misc fitting parameters
        fit_dict = {}

        # Fit using cross-correlation--best for scanning a wide range
        if fit_method == "CC":
            rv, rv_steps, cross_corrs = fit_rv_cross_corr(
                sci_wave=sci_wave,
                sci_flux=sci_flux,
                calc_template_flux=calc_template_flux,
                bcor=self.bcor,
                rv_min=rv_min,
                rv_max=rv_max,
                delta_rv=delta_rv,
                do_diagnostic_plots=do_diagnostic_plots,
                figsize=figsize)

            e_rv = np.nan
            fit_dict["rv_steps"] = rv_steps
            fit_dict["cross_corrs"] = cross_corrs

        # Fit using least squares--best once the local maximum has been 
        # identified and you want to fit considering flux uncertainties.
        elif fit_method == "LS":
            rv, e_rv, fit_dict = fit_rv_least_squares(
                sci_wave=sci_wave,
                sci_flux=sci_flux,
                sci_sigma=sci_sigma,
                calc_template_flux=calc_template_flux,
                bcor=self.bcor,
                rv_init_guess=rv_init_guess,
                rv_min=rv_min,
                rv_max=rv_max,
                ls_diff_step=ls_diff_step,
                do_diagnostic_plots=do_diagnostic_plots,
                verbose=verbose,
                figsize=figsize)

        if verbose:
            print(rv, "km/s")

        # Save our fitted RV, return fitting dictionary
        self.rv = rv
        self.e_rv = e_rv

        return fit_dict


def initialise_observation_from_crires_fits(
    fits_file_extracted,
    fits_file_slit_func=None,
    fits_ext_names=("CHIP1.INT1", "CHIP2.INT1", "CHIP3.INT1"),
    slit_pos_spec_num=1,
    initialise_empty_bad_px_mask=True,
    file_format="CRIRES",
    site="Paranal",
    rv=np.nan,
    e_rv=np.nan,
    drop_empty_orders=True,):
    """Takes a CRIRES+ 1D extracted nodding fits file, and initialises an
    Observation object containing a set of Spectra1D objects.

    Parameters
    ----------
    fits_file_extracted: string
        Filepath to CRIRES+ 1D extracted nodding fits file.

    fits_file_slit_func: string, default: None
        Filepath to CRIRES+ slit function for reduced data. Optional.

    fits_ext_names: string array, 
        default: ("CHIP1.INT1", "CHIP2.INT1", "CHIP3.INT1")
        HDU names for each chip. Defaults to CRIRES+ standard.

    slit_pos_spec_num: int, default: 1
        Spectrum number for when extracting multiple spectra from different
        positions along the slit. Currently an Observation object only holds
        a single spectrum, so this function should be called N times for 
        extended objects where N is the number of spectra extracted.

    initialise_empty_bad_px_mask: boolean, default: True
        Whether to initialise empty/uniniformative bad px masks.

    file_format: string, default: 'CRIRES'
        File format of input fits files. Revevant if different headers are
        used when extracting coordinate and time info out for computation of
        the barycentric velocity.

    site: string, default: 'Paranal'
        Observatory location for computation of the barycentric velocity.

    rv, e_rv: float, default: np.nan
        RV and associated uncertainty of the star (if known) in km/s.

    drop_empty_orders: boolean, default: True
        Whether to drop all nan orders.

    Returns
    -------
    observation: luciferase.spectra.Observation object
        An Observation object containing Spectra1D objects and associated info.
    """
    # List of valid file formats
    VALID_FILE_FORMATS = ["CRIRES",]

    with fits.open(fits_file_extracted) as fits_file:
        # Check file format is valid, and if so get the barycentric correction
        if file_format == "CRIRES":
            # Polarisation observations cannot have multiple spectra extracted
            # from along the slit, and thus have a different format to nodding
            # frames.
            if fits_file[0].header["HIERARCH ESO INS1 OPTI1 ID"] == "SPU":
                obs_type = "POL"
                wl_col_suffix = "_WL"
                spec_col_suffix = "_INTENS"
                sigma_col_suffix = "_INTENS_ERR"
            # Otherwise assume the observations are in the nodding format, and
            # use the provided slit_pos_spec_num.
            else:
                obs_type = "NOD"
                wl_col_suffix = "_{:02.0f}_WL".format(slit_pos_spec_num)
                spec_col_suffix = "_{:02.0f}_SPEC".format(slit_pos_spec_num)
                sigma_col_suffix = "_{:02.0f}_ERR".format(slit_pos_spec_num)

            # Calculcate midpoint
            exp_time_sec = fits_file[0].header["HIERARCH ESO DET SEQ1 EXPTIME"]
            t_mid = fits_file[0].header["MJD-OBS"] + exp_time_sec / 86400 / 2

            # Calculate barycentric correction. 
            # TODO fix 'dubious year' warnings thrown by astropy
            bcor = lutils.compute_barycentric_correction(
                ra=fits_file[0].header["RA"],
                dec=fits_file[0].header["DEC"],
                time_mid=t_mid,
                site=site,)

        else:
            raise Exception("Invalid file format, must be in {}".format(
                VALID_FILE_FORMATS))
        
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
                if ("{:02.0f}{}".format(order, wl_col_suffix) 
                    not in hdu_data.columns.names):
                    continue

                wave = hdu_data["{:02.0f}{}".format(order, wl_col_suffix)]
                spec = hdu_data["{:02.0f}{}".format(order, spec_col_suffix)]
                e_spec = hdu_data["{:02.0f}{}".format(order, sigma_col_suffix)]

                if initialise_empty_bad_px_mask:
                    bad_px_mask = np.full(len(wave), False)
                else:
                    raise NotImplementedError(
                        "Error: currently can only use default bad px mask")

                # Check if we have been given the slit function file, and if
                # so use it to compute the seeing for our current order
                # TODO: unclear how this interacts with polarimetric 
                # observations
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

                spec_obj = ObservedSpectrum(
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

                # If we have all nan arrays (SNR set to 0 by default), it means
                # the orders are empty and should be skipped.
                if drop_empty_orders and spec_obj.snr == 0:
                    continue
                else:
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
            n_orders=len(orders),
            fits_file= os.path.abspath(fits_file_extracted),
            bcor=bcor,
            rv=rv,
            e_rv=e_rv,)
            
    return observation


def read_vald_linelist(filepath,):
    """Load in a VALD sourced line list into a pandas dataframe format. Note 
    that the first two lines (everything before the column names) and
    everything after the last row of line data should be commented out with #,
    the column names should have commas inserted between them, and that VALD by
    default returns air (rather than vacuum) wavelengths.

    Parameters
    ----------
    filepath: string
        Filepath to the VALD file.

    Returns
    -------
    line_list: pd.DataFrame
        DataFrame of line list data.
    """
    # Load in linelist
    line_list = pd.read_csv(
        filepath, 
        sep=",",
        comment="#")

    # Drop last column
    line_list.drop(columns=line_list.columns[-1], inplace=True)

    # Remove extra spaces in column names
    line_list.columns = line_list.columns.str.replace(" ", "")

    # Remove quotes in species column
    line_list["SpecIon"] = line_list["SpecIon"].str.replace("'", "")

    # Remove quotes from references column
    line_list["Reference"] = line_list["Reference"].str.replace("'", "")

    # Remove leading and trailihg spaces in reference column
    line_list["Reference"] = line_list["Reference"].str.lstrip()
    line_list["Reference"] = line_list["Reference"].str.rstrip()

    # Convert air to vacuum wavelengths
    line_list["WL_vac(A)"] = lutils.convert_air_to_vacuum_wl(
        line_list["WL_air(A)"])

    # Convert both these to nm
    line_list["WL_air(nm)"] = line_list["WL_air(A)"] / 10
    line_list["WL_vac(nm)"] = line_list["WL_vac(A)"] / 10

    # Reorder columns so wavelength columns are together
    new_col_order = ["SpecIon", "WL_air(A)", "WL_vac(A)", "WL_air(nm)", 
        "WL_vac(nm)", "Excit(eV)", "Vmic", "loggf*", "Rad.", "Stark", "Waals",
        "factor", "depth", "Reference",]

    line_list = line_list[new_col_order]

    return line_list


def load_saved_observation_obj():
    """
    """
    pass


def fit_rv_cross_corr(
    sci_wave,
    sci_flux,
    calc_template_flux,
    bcor,
    rv_min,
    rv_max,
    delta_rv,
    do_diagnostic_plots,
    figsize,):
    """Computes the cross correlation of pre-masked observed spectra against
    a template spectrum from rv_min to rv_max in steps of delta_rv. Returns
    the best fit RV, as well as the RV steps, and cross correlation values.

    Parameters
    ----------
    sci_wave: float array
        Science wavelength array corresponding to sci_flux.
    
    sci_flux: float array
        Science flux array corresponding to sci_wave. Note that pixels not to
        be included in the fit should be pre-masked and set to some default 
        value (ideally NaN).

    calc_template_flux: scipy.interpolate.interpolate.interp1d
        Interpolation function for RV template to cross correlate with.

    bcor: float
        Barcycentric correction in km/s.

    rv_min: float
        Lower RV bound in km/s to apply to both cross correlation and least
        squares fitting.

    rv_max: float
        Upper RV bound in km/s to apply to both cross correlation and least
        squares fitting.

    delta_rv: float
        RV step size for cross correlation scan in km/s.

    do_diagnostic_plots: boolean
        Whether to plot diagnostic plots at the conclusion of the fit.

    figsize: float tuple
        Figure size of diagnostic plot.

    Returns
    -------
    rv: float
        Best fit RV in km/s.

    rv_steps: float array
        RV values in km/s the cross correlation was evaluated at.

    cross_corrs: float array
        Cross correlation values corresponding to rv_steps.
    """
    # Compute the RV steps to evaluate the cross correlation at
    rv_steps = np.arange(rv_min, rv_max, delta_rv)

    # Initialise our output vector of cross correlations
    cross_corrs = np.full(rv_steps.shape, 0)

    # Run cross correlation for each RV step
    for rv_i, rv in enumerate(rv_steps):
        # Shift the template spectrum
        template_flux = calc_template_flux(
            sci_wave * (1-(rv-bcor)/(const.c.si.value/1000)))

        # Compute the cross correlation
        cross_corrs[rv_i] = np.nansum(sci_flux * template_flux)

    # Determine closest RV
    best_fit_rv =  rv_steps[np.argmax(cross_corrs)]

    if do_diagnostic_plots:
        plt.close("all")
        fig, (cc_axis, spec_axis) = plt.subplots(2,1, figsize=figsize)
        plt.subplots_adjust(hspace=0.4,)

        # Plot cross correlation fit
        cc_axis.plot(rv_steps, cross_corrs, linewidth=0.2)
        cc_axis.set_xlabel("RV (km/s)")
        cc_axis.set_ylabel("Cross Correlation")

        # Compute best fit template and plot spectral fit
        template_flux = calc_template_flux(
            sci_wave * (1-(best_fit_rv-bcor)/(const.c.si.value/1000)))

        spec_axis.plot(
            sci_wave, sci_flux, linewidth=0.2, label="science")

        spec_axis.plot(
            sci_wave, template_flux, linewidth=0.2, label="template")

        spec_axis.legend()
        spec_axis.set_xlabel("Wavelength (nm)")
        spec_axis.set_ylabel("Flux (cont norm)")
        plt.tight_layout()
    
    return best_fit_rv, rv_steps, cross_corrs


def fit_rv_least_squares(
    sci_wave,
    sci_flux,
    sci_sigma,
    calc_template_flux,
    bcor,
    rv_init_guess,
    rv_min,
    rv_max,
    ls_diff_step,
    do_diagnostic_plots,
    verbose,
    figsize,):
    """Performs least squares fitting of pre-masked observed spectra against
    a template spectrum from rv_min to rv_max. Returns the best fit RV, the
    statistical uncertainty, as well as the returned fitting dictionary.

    Parameters
    ----------
    sci_wave: float array
        Science wavelength array corresponding to sci_flux and sci_sigma.
    
    sci_flux: float array
        Science flux array corresponding to sci_wave and sci_sigma. Note that 
        pixels not to be included in the fit should be pre-masked and set to 
        some default value (ideally 1, as NaN breaks the least squares fit).

    sci_sigma: float array
        Science sigma array corresponding to sci_wave and sci_flux. Note that 
        pixels not to be included in the fit should be pre-masked and set to 
        some default value (ideally inf, as NaN breaks the least squares fit).

    calc_template_flux: scipy.interpolate.interpolate.interp1d
        Interpolation function for RV template to cross correlate with.

    bcor: float
        Barcycentric correction in km/s.

    rv_init_guess: float
        Initial guess for the RV in km/s.

    rv_min: float
        Lower RV bound in km/s to apply to both cross correlation and least
        squares fitting.

    rv_max: float
        Upper RV bound in km/s to apply to both cross correlation and least
        squares fitting.

    ls_diff_step: float tuple, default: (0.1)
        Input for least squares fitting, sets the step size in km/s.

    do_diagnostic_plots: boolean
        Whether to plot diagnostic plots at the conclusion of the fit.

    verbose: boolean
        Whether to print updates on fitting.

    figsize: float tuple
        Figure size of diagnostic plot.

    Returns
    -------
    rv, std: float
        Best fit RV and corresponding statistical uncertainty.

    ls_fit_dict: dict
        Dictionary overview of least squares fit as outputted from 
        scipy.optimize.least_squares.
    """
    # Prepare list of arguments for least squares fitting
    params_init = [rv_init_guess]
    args = (sci_wave, sci_flux, sci_sigma, calc_template_flux, bcor, verbose)

    # Do fit
    ls_fit_dict = least_squares(
        calc_rv_shift_residual, 
        params_init, 
        jac="3-point",
        bounds=((rv_min),(rv_max)),
        diff_step=ls_diff_step,
        args=args,)

    # Calculate uncertainty
    residuals = ls_fit_dict["fun"]

    # Calculate RMS to scale uncertainties by
    rms = np.sqrt(np.sum(residuals**2)/np.sum(residuals != 0))

    # Calculate uncertainties
    jac = ls_fit_dict["jac"]
    cov = np.linalg.inv(jac.T.dot(jac))
    std = np.sqrt(np.diagonal(cov)) * rms
    ls_fit_dict["std"] = std

    rv = float(ls_fit_dict["x"])

    if do_diagnostic_plots:
        plt.close("all")
        fig, spec_axis = plt.subplots(1,1, figsize=figsize)

        # Compute best fit template and plot spectral fit
        template_flux = calc_template_flux(
            sci_wave * (1-(rv-bcor)/(const.c.si.value/1000)))

        spec_axis.plot(
            sci_wave, sci_flux, linewidth=0.2, label="science")

        spec_axis.plot(
            sci_wave, template_flux, linewidth=0.2, label="template")

        spec_axis.legend()
        spec_axis.set_xlabel("Wavelength (nm)")
        spec_axis.set_ylabel("Flux (cont norm)")
        plt.tight_layout()

    return rv, std, ls_fit_dict


def calc_rv_shift_residual(
    params,
    sci_wave,
    sci_flux,
    sci_sigma,
    template_flux_interp,
    bcor,
    verbose):
    """Loss function for least squares fitting.

    Parameters
    ----------
    params: float array
        params[0] is the radial velocity in km/s.
    
    wave: float array
        Wavelength scale of the science spectra.

    flux: float array
        Fluxes for the science spectra.

    e_flux: float array
        Uncertainties on the science fluxes.

    template_flux_interp: scipy.interpolate.interpolate.interp1d object
        Interpolation function to take wavelengths and compute template
        spectrum.

    bcor: float
        Barycentric velocity in km/s.

    verbose: boolean
        Whether to print updates on fitting.

    Returns
    -------
    resid_vect: float array
        Error weighted loss.
    """
    # Shift the template spectrum
    template_flux = template_flux_interp(
        sci_wave * (1-(params[0]-bcor)/(const.c.si.value/1000)))

    # Return loss
    resid_vect = (sci_flux - template_flux) / sci_sigma

    rchi2 = np.sum(resid_vect**2) / (len(resid_vect)-len(params))

    if verbose:
        print("RV = {:+0.5f}, rchi^2 = {:0.5f}".format(params[0],rchi2))

    return resid_vect


def calc_continuum_optimisation_resid(
    params,
    sci_wave,
    sci_flux,
    sci_sigma,
    telluric_model_flux,):
    """Calculates the residuals between a continuum normalised observed science
    spectrum and the best fit molecfit model. Called to optimise over the 
    polynomial coefficients in params to get the best possible continuum
    normalisation.

    Parameters
    ----------
    params: float array
        Contains the polynomial coefficients for use with
        numpy.polynomial.polynomial.Polynomial.

    sci_wave, sci_flux, sci_sigma: float array
        Wavelength scale, fluxes, and uncertainties for the science spectrum.

    telluric_model_flux: float array
        Model telluric absorption normalised to 1.0.

    Returns
    -------
    resid_vect: float array
        Uncertainty weighted residuals vector.
    """
    # Calculate the continuum
    continuum_poly = Polynomial(params)
    continuum = continuum_poly(sci_wave)

    # Continuum normalise our science flux
    norm_sci_flux = sci_flux / continuum

    # Compute residuals
    resid_vect = (norm_sci_flux - telluric_model_flux) / sci_sigma

    return resid_vect