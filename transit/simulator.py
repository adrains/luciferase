"""Module containing functions involved with preparing simulated exoplanet
transits from model stellar, telluric, and planetary spectra.
"""
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.io import fits
import astropy.constants as const
from scipy.interpolate import interp1d
from scipy.integrate import simps
from PyAstronomy.pyasl import instrBroadGaussFast

# -----------------------------------------------------------------------------
# Load/save spectra
# -----------------------------------------------------------------------------
def read_marcs_spectrum(
    filepath,
    n_max_n_mu=55,
    header_length=36,
    wave_min=None,
    wave_max=None,):
    """Reads in a MARCS .itf file to extract wavelengths, weights, mu values,
    and radiant fluxes in units of ergs/cm^2/s/Å/µ.

    TODO: there might be a factor of pi missing in this definition.
    
    Note that the amount of mu points sampled varies for each wavelength point
    and ranges from 16 through 55. For speed we pre-allocate our mu and
    intensity arrays with shape [n_wave, n_max_n_mu], meaning that for any
    wavelength with < 55 samples the array will be 'padded' with nans. This
    does mean that one cannot take slices through this array in the mu
    direction prior to interpolation/resampling, but it is both
    computatationally expensive and unnecessary to do that for the entire MARCS
    spectral range, and we leave that until later when one is considering a
    more limited wavelength range.

    Parameters
    ----------
    filepath: str
        Filepath to MARCS .itf file.

    n_max_n_mu: int, default: 55
        The maximum number of mu samples across the disc we expect.

    header_length: int, default:36
        The number of characters in a header row.

    wave_min, wave_max: float or None, default: None
        Minimum and maximum wavelengths to return in Angstrom.

    Returns
    -------
    wavelengths: 1D float array
        Array of wavelengths of shape [n_wave].
    
    weights: 1D float array
        Array of weights of shape [n_wave].
    
    n_samples: 1D float array
        Array of n_mu_samples of shape [n_wave].
    
    mus: 2D float array
        Array of mus of shape [n_wave, n_max_n_mu].
    
    intensities: 2D float array
        Array of intensities of shape [n_wave, n_max_n_mu].
    """
    # Initialise booleans
    have_read_header = False
    have_read_mus = False
    have_read_intensities = False

    with open(filepath) as marcs_itf:
        # Read in all lines
        all_lines = marcs_itf.readlines()

        # Count the number of wavelengths so we can preallocate an array
        line_lengths = np.array([len(row) for row in all_lines])
        n_wave = np.sum(line_lengths == header_length)

        # Preallocate arrays
        wavelengths = np.zeros(n_wave)
        weights = np.zeros(n_wave)
        n_samples = np.zeros(n_wave)

        # Finally, initialise per-wavelength vectors
        mus = np.full((n_wave, n_max_n_mu), np.nan)
        intensities = np.full((n_wave, n_max_n_mu), np.nan)

        wave_i = 0
        mu_min = 0
        intens_min = 0

        if wave_min is not None and wave_max is not None:
            desc = "Reading MARCS File ({} < lambda < {} A)".format(
                wave_min, wave_max)
        else:
            desc = "Reading MARCS File"

        for line_str in tqdm(all_lines, desc=desc, leave=False):
            # Split on whitespace
            line_values = line_str.split()

            # -----------------------------------------------------------------
            # Parse line info
            # -----------------------------------------------------------------
            # Read header
            if not have_read_header:
                assert len(line_values) == 3

                wavelengths[wave_i] = float(line_values[0])
                weights[wave_i] = float(line_values[1])

                n_samp = int(line_values[2])
                n_samples[wave_i] = n_samp

            # Read mu values
            elif not have_read_mus:
                # Update mu values
                mu_max = mu_min + len(line_values)

                mus[wave_i, mu_min:mu_max] = \
                    np.array(line_values).astype(float)

                # Update min
                mu_min = mu_max

            # Read intensity values
            elif not have_read_intensities:
                # Update intensity values
                intens_max = intens_min + len(line_values)

                intensities[wave_i, intens_min:intens_max] = \
                    np.array(line_values).astype(float)

                # Update min
                intens_min = intens_max

            # -----------------------------------------------------------------
            # Checks to change state (i.e. update booleans and counters)
            # -----------------------------------------------------------------
            # Have read header
            if not have_read_header:
                have_read_header = True

            # Have read header and finished reading mus
            elif have_read_header and not have_read_mus and mu_min == n_samp:
                have_read_mus = True
            
            # Have read header, mus, and intensities
            elif have_read_header and have_read_mus and intens_min == n_samp:
                # Reset counters
                mu_min = 0
                intens_min = 0

                # Reset booleans
                have_read_header = False
                have_read_mus = False
                have_read_intensities = False

                # Index wavelength
                wave_i += 1

    # Determine mask if we've been given both upper and lower bounds
    if wave_min is not None and wave_max is not None:
        mask = np.logical_and(wavelengths > wave_min, wavelengths < wave_max)

    # Only given a lower bound
    elif wave_min is not None and wave_max is None:
        mask = wavelengths > wave_min

    # Only given an upper bound
    elif wave_min is None and wave_max is not None:
        mask = wavelengths < wave_max

    # No bounds
    else:
        mask = np.full_like(wavelengths, True).astype(bool)

    # Mask arrays and return
    wavelengths = wavelengths[mask]
    weights = weights[mask]
    n_samples = n_samples[mask]
    mus = np.abs(mus[mask, :])          # Ensure mus are positive
    intensities = intensities[mask, :]

    return wavelengths, weights, n_samples, mus, intensities


def load_crires_throughput_json(json_path, convert_um_to_angstrom=True):
    """Imports a JSON of the various components of CRIRES+'s instrumental
    transfer function into DataFrame format. A JSON can be downloaded for a
    particular grating setting (e.g. K2148) from the CRIRES+ Exposure Time
    Calculate (ETC, https://etc.eso.org/observing/etc/crires2).

    Parameters
    ----------
    json_path: string
        Filepath to the CRIRES+ JSON file.

    convert_um_to_angstrom: boolean, default: True
        Whether to convert the wavelength scale in um to Angstrom.

    Returns
    -------
    throughput_df: pandas DataFrame
        DataFrame with the measured instrumental throughputs for the variuous 
        parts of the system for a particular grating setting (e.g. K2148). This
        DataFrame the columns:
        ['wavelengths', 'atm.transmission', 'grating eff. incl. blaze',
         'eff.telescope', 'enslitted energy fraction', 'total throughput',
         'eff.detector.', eff.instrument]
    """
    # Open JSON
    with open(json_path) as json_file:
        data = json.load(json_file)

    # Intitalise output dict
    throughput_dict = {}

    # Extract wavelength scales
    wavelengths = []

    for detector in data["xaxis"]["detectors"]:
        wavelengths += detector["data"]

    wavelengths = np.array(wavelengths)

    if convert_um_to_angstrom:
        wavelengths *= 10

    throughput_dict["wavelengths"] = wavelengths

    # Extract throughput data for each kind
    for throughput in data["ypanels"][1]["seriesArray"]:
        label = throughput["label"]

        eta = []

        # Extract throughput data for each detector
        for detector in throughput["detectors"]:
            eta += detector["data"]

        throughput_dict[label] = eta

    # Convert to DataFrame
    throughput_df = pd.DataFrame.from_dict(throughput_dict)
    throughput_df.sort_values("wavelengths", inplace=True)

    return throughput_df


def load_planet_spectrum(wave_file, spec_file, convert_cm_to_angstrom=True,):
    """Load in the wavelength and flux vectors from a petitRADTRANS model
    planet spectrum and convert the wavelength scale from cm to Angstrom.

    Parameters
    ----------
    wave_file, spec_file: str
        Filepaths for petitRADTRANS wavelength and flux fits files respectively

    convert_cm_to_angstrom: boolean, default: True
        Whether to convert the wavelength scale in cm to Angstrom.

    Returns
    -------
    wave, spec: 1D float array
        Imported wavelength and spectra arrays.
    """
    with fits.open(wave_file) as wave_fits:
        wave = wave_fits[0].data

    with fits.open(spec_file) as spec_fits:
        spec = spec_fits[0].data

    if convert_cm_to_angstrom:
        wave /= (10**-8)

    return wave, spec


def save_planet_spectrum(wave_file, spec_file, wave, spec):
    """Save the wavelength and flux vectors from a petitRADTRANS model
    planet spectrum.

    Parameters
    ----------
    wave_file, spec_file: str
        Filepaths for petitRADTRANS wavelength and flux fits files respectively

    wave, spec: 1D float array
        Imported wavelength and spectra arrays.
    """
    hdu_wave = fits.HDUList()
    wave_img =  fits.PrimaryHDU(wave)
    wave_img.header["EXTNAME"] = ("WAVES", "Wavelength scale")
    hdu_wave.append(wave_img)
    hdu_wave.writeto(wave_file, overwrite=True)

    hdu_flux = fits.HDUList()
    flux_img =  fits.PrimaryHDU(spec)
    flux_img.header["EXTNAME"] = ("FLUX", "Planet fluxes")
    hdu_flux.append(flux_img)
    hdu_flux.writeto(spec_file, overwrite=True)


def load_telluric_spectrum(
    molecfit_fits,
    tau_fill_value=0,
    convert_to_angstrom=True,):
    """Load in the atmospheric *transmittance* from a MOLECFIT best fit file,
    and convert to optical depth as Tau = -ln (T). 

    The format of the molecfit BEST_FIT_MODEL.fits fike is one table HDU with
    the following columns:
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

    Of these, we take mlambda and mtrans.

    Parameters
    ----------
    molecfit_file: string
        Filepath to molecfit fitted telluric spectrum.

    tau_fill_value: float, default: 0
        Default value for missing values in the molecfit fitted spectrum.

    convert_um_to_angstrom: boolean, default: True
        Whether to convert the wavelength scale in um to Angstrom.

    Returns
    -------
    telluric_wave, telluric_tau: float array
        Loaded wavelength and tau (optical depth) arrays.
    
    calc_telluric_tau: scipy interp1d object
        Interpolator for tau.
    """
    with fits.open(molecfit_fits) as telluric_fits:
        # Extract data
        data = telluric_fits[1].data

        telluric_wave = data["mlambda"]
        telluric_trans = data["mtrans"]

        # Ensure these are sorted
        sorted_i = np.argsort(telluric_wave)
        telluric_wave = telluric_wave[sorted_i]
        telluric_trans = telluric_trans[sorted_i]

        # Calculate optical depth
        telluric_tau = -np.log(telluric_trans)

        # Construct an interpolator for the optical depth
        calc_telluric_tau = interp1d(
            x=telluric_wave,
            y=telluric_tau,
            fill_value=tau_fill_value,
            assume_sorted=True,)

    if convert_to_angstrom:
        telluric_wave *= 1E4

    return telluric_wave, telluric_tau, calc_telluric_tau


# -----------------------------------------------------------------------------
# General Functions
# -----------------------------------------------------------------------------
def interpolate_marcs_spectrum_at_mu(
    fluxes_mu,
    mus,
    mu_selected,):
    """Interpolate a MARCS spectrum initially in units of ergs/cm^2/s/Å/µ to a
    specific µ value.

    Parameters
    ----------
    fluxes_mu: 2D float array
        Array of fluxes of shape [n_wave, n_max_n_mu].

    mus: 2D float array
        Array of mus of shape [n_wave, n_max_n_mu].
    
    mu_selected: float
        New mu value to interpolate fluxes for.

    Returns
    -------
    fluxes_new_mu: 1D float array
        Array of fluxes at mu_selected of shape [n_wave].
    """
    # Basic error checking
    if mu_selected > 1 or mu_selected < 0:
        raise ValueError("Invalid mu value, must be 0 <= mu <= 1")
    elif mu_selected == 0:
        raise Warning("mu=0. May encounter edge-effects.")
    
    fluxes_new_mu = np.zeros(fluxes_mu.shape[0])

    # For each wavelength point, interpolate to the new mu value given our
    # existing flux(mu) scale output from MARCS. Make sure to avoid nans.
    desc = "Interpolating to mu value"

    for wave_i in tqdm(range(fluxes_mu.shape[0]), desc=desc, leave=False):
        mask = ~np.isnan(mus[wave_i])

        interp_flux = interp1d(
            x=mus[wave_i][mask],
            y=fluxes_mu[wave_i][mask],)
        
        fluxes_new_mu[wave_i] = interp_flux(mu_selected)

    return fluxes_new_mu


def integrate_marcs_spectrum(
    wavelengths,
    mus,
    intensities,):
    """Given F(λ,µ), integrate the flux from the entire stellar disc as
    representing an observed spectrum of an unresolved star.

    We formulate this problem as finding the volume of a solid of revolution,
    where the height is our intensity in units of energy/s/area, and we 
    integrate this across the stellar disc as:

    F(λ) = ∫ π f(µ)^2 dx

    Parameters
    ----------
    wavelengths: 1D float array
        Array of wavelengths of shape [n_wave].
    
    mus: 2D float array
        Array of mus of shape [n_wave, n_max_n_mu].
    
    intensities: 2D float array
        Array of intensities of shape [n_wave, n_max_n_mu].
    
    Returns
    -------
    integrated_flux: 1D float array
        Disc integrated flux of the star of shape [n_wave].
    """
    # Initialise output flux vector
    flux_integrated = np.zeros_like(wavelengths)

    desc = "Integrating"

    # For every wavelength
    for wave_i, wave in enumerate(tqdm(wavelengths, desc=desc, leave=False)):
        # Invert mus to get x spacing
        x = np.sqrt(1 - mus[wave_i]**2)
        mask = ~np.isnan(x)
        x = x[mask]

        # Integrate the flux at this wavelength
        yy = np.pi * intensities[wave_i, mask]**2
        flux_integrated[wave_i] = simps(y=yy, x=x)

    return flux_integrated


def physically_scale_stellar_flux(
    wavelengths,
    fluxes_mu,
    fluxes_disk,
    r_star_r_sun,
    dist_pc,
    r_tel_prim,
    r_tel_cen_ob,):
    """Physically scale a MARCS synthetic spectrum to the flux collected by a
    circular telescope (with circular central obstruction) of a star of certain
    radius and distance assuming *no* atmospheric or instrumental losses (these
    will be applied later). We are specifically interested in scaling the
    *disc integrated* fluxes (i.e. a spatially unresolved spectrum of the star,
    integrated over all mu points) to the light received at Earth. 

    There are some geometry considerations in this problem, since the disc
    integrated spectrum is a 2D integration (circle), but the total flux of the
    star involves 3D geometry (sphere). We make the following assumptions:
     - The 3D integration can be done independently of mu by assuming that we
       integrate only the flux emitted normal (mu=1) to the stellar surface.
     - Given an arbitrary sized collecting area collecting X% of total stellar
       flux, over a sufficiently broad wavelength band the total total flux
       (integrated over all wavelengths) should be the same for a spectrum with
       mu=1 and the disc integrated spectrum taking into account limb darkening

    The units of a MARCS spectrum are ergs/cm^2/s/Å measured at the surface of
    the star. Our goal is to directly scale our fluxes using the stellar
    radius, distance, and telescope size to those received at Earth. We
    formulate the problem as follows:
     - Find total flux by integrating a sphere of radius r_star
     - Fraction of this light collected on Earth is the ratio between the
       telescope collecting area and the *surface area* of a sphere of radius 
       dist_star.

    Using equations:
    A_1 = 4*pi*dist_star^2
    A_2 = pi*r_tel_prim^2 - pi*r_tel_cen_ob^2
        = pi*(r_tel_prim^2 - r_tel_cen_ob^2)

    scale_fac = A_2 / A_1
              = (r_tel_prim^2 - r_tel_cen_ob^2) / (4*dist_star^2)
    
    Steps:
    1) Integrate total stellar flux given stellar radius and spectrum using
       mu=0 spectrum.
    2) Using stellar distance and telescope collecting area, calculate how much
       of this flux reaches Earth (in a particular band).
    3) Scale disk integrated spectrum in a given band to match the fractional
       total received in that band at Earth.

    Parameters
    ----------
    wavelengths: 1D float array
        Array of wavelengths of shape [n_wave].
    
    fluxes_marcs: 2D float array
        Array of intensities of shape [n_wave, n_max_n_mu].

    fluxes_disk: 1D float array.
        Disc integrated flux of the star of shape [n_wave].

    r_star_r_sun: float
        Radius of the star in units of R_sun.

    dist_pc: float
        Distance to the star in units of pc.

    r_tel_prim: float
        Radius of the telescope primary mirror in metres.

    r_tel_cen_ob: float
        Radius of the telescope central obstruction in metres.

    Returns
    -------
    fluxes_disk_scaled: 1D float array
        Spectral fluxes in units of ergs/s/Å scaled to the light received at
        Earth.

    flux_total_integrated: float
        The *total* integrated flux of the star in ergs/s within our given
        wavelength band.
    """
    assert fluxes_mu.shape[0] == len(fluxes_disk)

    # Grab constants for convenience
    pc_2_m = const.pc.value
    r_sun_2_cm = const.R_sun.cgs.value

    # Calculate the scale factor for the total amount of light received by a
    # telescope on Earth. This is the fraction of total light collected by our
    # telescope (absent any throughput losses which are considered later).
    flux_frac = (r_tel_prim**2 - r_tel_cen_ob**2) / (4*(dist_pc*pc_2_m)**2)

    # Calculate total stellar flux by integrating over a sphere representing 
    # our stellar surface area. This new flux array has units ergs/s/Å. We use
    # only the normal component of our flux array with mu=0 for this.
    fluxes_total = fluxes_mu[:,0]*4*np.pi * (r_star_r_sun*r_sun_2_cm)**2

    # Scale our flux array to be the light received by our telescope.
    fluxes_scaled =  fluxes_total * flux_frac

    # Determine the radiant flux (erg/sec) for both the total and disk flux by
    # integrating over all wavelengths in our current bandpass.
    flux_total_integrated = simps(y=fluxes_scaled, x=wavelengths)
    flux_disk_integrated = simps(y=fluxes_disk, x=wavelengths)

    # Making the assumption that the flux within a (sufficiently broad) band
    # should be constant, we can scale our disk integrated spectrum to match
    # the total light received at Earth by our telescope.
    scale_frac = flux_total_integrated / flux_disk_integrated

    fluxes_disk_scaled = fluxes_disk * scale_frac

    return fluxes_disk_scaled, flux_total_integrated


def apply_instrumental_transfer_function(
    wave,
    flux,
    throughput_json_path,
    fill_throughput_value=0,):
    """Apply the CRIRES+ instrumental transfer function for the telescope, 
    enslitted fraction, instrument throughput, grating + blaze efficiency, and 
    detector efficiency. Note that we do *not* account for the atmospheric
    throughput here.

    Parameters
    ----------
    wave, flux: 1D float array
        Wavelength and flux arrays of shape [n_wave].

    throughput_json_path: string
        Filepath to the CRIRES+ JSON file.

    fill_throughput_value: float, default: 0
        Default througput value to fill undefined wavelength points. Default is
        0, meaning that we assume no transmission for undefined values.

    Returns
    -------
    flux_scaled: 1D float array
        Scaled flux array of shape [n_wave].
    """
    # Load throughput JSON
    throughput_df = load_crires_throughput_json(throughput_json_path,)

    # Multiply throughputs (all except atmosphere)
    total_throughput = (throughput_df["grating eff. incl. blaze"]
        * throughput_df["eff.telescope"]
        * throughput_df["enslitted energy fraction"]
        * throughput_df["eff.detector"]
        * throughput_df["eff.instrument"])
    
    # Interpolate throughput
    calc_throughput = interp1d(
        x=throughput_df["wavelengths"].values,
        y=total_throughput.values,
        fill_value=fill_throughput_value,)
    
    throughput = calc_throughput(wave)

    # Scale flux by throughput
    flux_scaled = flux * throughput

    return flux_scaled


def convert_flux_to_counts(wave, flux, gain,):
    """Convert flux in erg/sec to units of counts/sec. 

    TODO: currently we assume a uniform gain for all spectral segments. 
    However, the CRIRES+ data reduction pipeline uses a different gain for each
    of its 3 detectors, which should be implemented here:
        CR2RES_GAIN_CHIP1   2.15
        CR2RES_GAIN_CHIP2   2.19
        CR2RES_GAIN_CHIP3   2.00

    Parameters
    ----------
    wave, flux: 1D float array
        Wavelength and flux arrays of shape [n_wave].

    gain: float
        Gain in e- / ADU
    Returns
    -------
    flux_counts_per_sec: 1D float array
        Flux array in units of counts/sec with shape [n_wave].
    """
    # Convert from ergs/s to photons/sec
    photon_energy_per_wl = \
        const.c.cgs.value * const.h.cgs.value / (wave * 10E-10)

    flux_photons_per_sec = flux / photon_energy_per_wl

    # Convert from photons/sec to counts/sec
    flux_counts_per_sec = flux_photons_per_sec / gain

    return flux_counts_per_sec


def smear_spectrum():
    """TODO. For now we're using the broadening function from PyAstronomy and
    assuming that delta RV across the exposure is small. It is, however, more
    rigorous to combine the 'observed' planet exposure from a number of sub-
    exposures calculating the orbital velocities at each step.
    """
    pass


def interpolate_spectrum(
    wave_new,
    wave_old,
    flux_old,
    fill_value=np.nan,):
    """Interpolate spectrum to a new wavelength scale using linear 
    interpolation. This function assumes that wave_old is a single 1D array,
    but wave_new might be multple different wavelength segments (e.g. different
    detectors).
    N_order x N_detector spectral segments.

    Parameters
    ----------
    wave_new: 1D or 2D float array
        New wavelength scale, can be either 1D (a single wavelength scale) or
        2D (multiple separate scales).

    wave_old: 1D float array
        Original wavelengths scale for the data, has shape [n_wave].

    flux_old: 1D float array
        Original flux scale for the data, has shape [n_wave].

    fill value: float, default: np.nan
        Fill value for missing data when interpolating.

    Returns
    -------
    flux_new: 1D or 2D float array
        New flux array with shape matching that of wave_new.
    """
    # Simple case is that we only have a single spectral segment, so we only
    # need to interpolate a single time.
    if len(wave_new.shape) == 1:
        flux_interp = interp1d(
            x=wave_old,
            y=flux_old,
            bounds_error=False,
            fill_value=fill_value,)
        
        flux_new = flux_interp(wave_new)

    # More complex case of multiple segments (e.g. multiple detectors) means
    # multiple different interpolations.
    elif len(wave_new.shape) > 1:
        flux_new = np.full_like(wave_new, np.nan)

        for spec_i in range(wave_new.shape[0]):
            flux_interp = interp1d(
                x=wave_old,
                y=flux_old,
                bounds_error=False,
                fill_value=fill_value,)

            flux_new[spec_i] = flux_interp(wave_new[spec_i])
    
    # Note sure if we can get here, but error check anyway
    else:
        raise Exception("Invalid dimensions for wave_new.")
    
    # Error checking on output--throw an exception if our new flux array is all
    # zeroes/nans (and our input array is nonzero) which might indicate
    # different units on our wavelength scales.
    if np.nansum(flux_old) != 0 and np.nansum(flux_new) == 0:
        raise Exception("Interpolated flux == 0")

    return flux_new


def calc_model_flux(
    stellar_flux,
    telluric_tau,
    planet_trans,
    shadow_flux_opaque,
    shadow_flux_atmo,
    airmass,
    scale,):
    """Compute the model observed flux.

    The Aronson method combines spectra as:

    M^j_λ = [F_λ(1+γ_j) - I^j_λ(1+β_j) * P^j_λ(1+δ_j)] * e^(-tau_λ * z_j) * S_j

    Where:
     - M^j_λ is the "observed" model spectrum at orbital phase j and 
       wavelength λ
     - F_λ(1+γ_j) is the total stellar flux in the stellar with doppler
       shift γ_j
     - I^j_λ(1+β_j) is the planet flux at orbital phase j with doppler 
       shift β_j
     - P^j_λ(1+δ_j) is the planet "shadow" at phase j with doppler shift δ
     - tau_λ is the optical depth of telluric absorption in Earth's atmosphere
     - z_j is the airmass at phase j
     - S_j is a scaling factor which accounts for observational conditions.

    Where the velocity components are defined as:
     - γ_j = (v_bary + v_star) / c
     - β_j = (v_bary + v_star + vsini * x_planet) / c
     - δ_j = (v_bary + v_star + v^i_planet) / c

    Velocities are precomputed for each phase in transit_info.

    However, for this simulation we want to be more explicit in modelling the
    transparent and non-transparent components of the planet. To do this, we
    replace the content of the square brackets with:

    S = F_λ(1+γ_j) - I_P^j_λ(1+β_j) * A + I_P^j_λ(1+β_j) * P^j_λ(1+δ_j) * A

    Where we first remove flux assuming an entirely opaque planet, then add 
    back in the flux corresponding to the planet atmosphere. In reality this
    corresponds to the planet scale height (with different absorption occuring
    at different altitudes), but for simplicity to begin with we'll just adopt
    an atmospheric thickness as a fraction of the planet radius and apply the
    planet transmission to the amount of flux travelling through this.

    Parameters
    ----------
    stellar_flux: 1D float array
        Stellar fluxes of shape [n_wave].

    telluric_tau: 1D float array
        Optical depth array of shape [n_wave].

    planet_trans: 1D float array
        Planet atmosphere transmittance array of shape [n_wave].

    shadow_flux_opaque: 1D float array
        *Blocked* stellar flux array (assuming the planet is completed opaque)
        of shape [n_wave]. This should be equal to zero when the planet is not
        transitting.

    shadow_flux_atmo: 1D float array
        *Transmitted* stellar flux array (light passing through a transparent
        annulus representing the atmosphere of the planet) of shape [n_wave].
        This should be equal to zero when the planet is not transitting.

    airmass: float
        Airmass at the current epoch.

    scale: float
        Flux scale factor for the current epoch, used in the Aronson method to
        account for e.g. variable seeing.

    Returns
    -------
    flux_model_ob: 1D float array
        Simulated model fluxes of shape [n_wave].
    """
    flux_model_ob = (
        (stellar_flux - shadow_flux_opaque + shadow_flux_atmo * planet_trans)
        * np.exp(-telluric_tau * airmass) * scale)
    
    return flux_model_ob

# -----------------------------------------------------------------------------
# Simulate
# -----------------------------------------------------------------------------
def simulate_transit_single_epoch(
    wave_observed,
    syst_info,
    transit_epoch,
    wave_marcs,
    fluxes_marcs,
    mus_marcs,
    wave_planet,
    trans_planet,
    telluric_wave,
    telluric_tau,
    instr_resolving_power,
    r_tel_prim,
    r_tel_cen_ob,
    throughput_json_path,
    do_equidistant_lambda_sampling_before_broadening=True,
    fill_throughput_value=0,
    planet_transmission_boost_fac=1,):
    """Simulate a single epoch of a planet during transit by modelling the
    stellar, planetary, and telluric components. This function takes into
    account the different velocities of each component, instrumental/velocity
    broadening of spectra, and the instrumental transfer function to return
    a modelled 'observed' spectrum in units of counts.

    We use a slightly different formalism to the Aronson Method here, in that
    we model:
     - The stellar flux (doppler shift gamma)
     - The stellar flux blocked assuming an entirely opaque planet (beta)
     - The stellar flux transmitted through the planet atmosphere (beta)
     - The planet transmittance (delta)
     - The optical depth of telluric absorption
    with each component shifted to the appropriate radial velocity listed in
    brackets. Note that when simulating a transit, we are shifting from the
    barcentric/rest-frame to the stellar frame, so we use the convention
    wave_stellar = (1 - doppler_shift) * wave_rest.

    Note that we use the broadening functions from PyAstronomy:
     - https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/broad.html
    
    TODO: do a consistency check, as it'd be nice to expect that convolving a
    spectrum to a resolution higher than the input doesn't change the input,
    but in practice (likely due to the need to regrid things beforehand) we
    don't end up with the same spectrum out. Probably not a concern for CRIRES
    since we'll be dropping the resolution by ~half, but might be worth
    considering resampling to finer wavelength scale before doing any
    broadening.

    Parameters
    ----------
    wave_observed: 1D or 2D float array
        Observed/instrumental wavelength scale of shape [n_wave] for a single 
        spectral segment, or [n_spec, n_wave] in the case of multiple orders or
        detectors.

    syst_info: pandas DataFrame
        DataFrame containing planet/star/system properties. The data frame has
        columns ['value', 'sigma', 'reference', 'comment'] and indices:
        ['m_star_msun', 'r_star_rsun', 'k_star_mps', 'dist_pc', 'vsini',
         'rv_star', 'ldc_init_a1', 'ldc_init_a2', 'ldc_init_a3', 'ldc_init_a4',
         'a_planet_au', 'e_planet', 'i_planet_deg', 'omega_planet_deg',
         'w_planet_deg', 'k_star_mps', 'transit_dur_hours', 'jd0_days', 
         'period_planet_days', 'm_planet_mearth', 'r_planet_rearth', 
         'r_planet_atmo_r_earth'].

    transit_epoch: pandas Series
        Single row of transit_info DataFrame representing a single epoch. This
        series has columns: 

        ['mjd_start', 'mjd_mid', 'mjd_end', 'jd_start', 'jd_mid', 'jd_end',
         'airmass', 'bcor', 'hcor', 'ra', 'dec', 'exptime_sec', 'nod_pos',
         'raw_file', 'phase_start', 'is_in_transit_start', 'r_x_start',
         'r_y_start', 'r_z_start', 'v_x_start', 'v_y_start', 'v_z_start',
         's_projected_start', 'scl_start', 'mu_start',
         'planet_area_frac_start', 'phase_mid', 'is_in_transit_mid',
         'r_x_mid', 'r_y_mid', 'r_z_mid', 'v_x_mid', 'v_y_mid', 'v_z_mid',
         's_projected_mid', 'scl_mid', 'mu_mid', 'planet_area_frac_mid',
         'phase_end', 'is_in_transit_end', 'r_x_end', 'r_y_end', 'r_z_end',
         'v_x_end', 'v_y_end', 'v_z_end', 's_projected_end', 'scl_end',
         'mu_end', 'planet_area_frac_end', 'gamma', 'beta', 'delta']

    wave_marcs: 1D float array
        Wavelength grid of MARCS spectrum of shape [n_wave_marcs].
    
    fluxes_marcs, mus_marcs: 2D float array
       MARCS flux and mu arrays of shape [n_wave_marcs, n_max_n_mu].

    wave_planet, trans_planet: 1D float array
        Wavelength and transmittance arrays for the planet spectrum of shape
        [n_wave_planet].

    telluric_wave, telluric_tau: 1D float array
        Wavelength and tau arrays for the telluric spectrum of shape 
        [n_wave_telluric].

    instr_resolving_power: float
        Instrumental resolving power to simulate.

    r_tel_prim, r_tel_cen_ob: float
        Radii of telescope primary mirror and central obstruction in metres.

    throughput_json_path: string
        Filepath to the CRIRES+ JSON file containing instrument transfer
        functions as a function of wavelength.

    do_equidistant_lambda_sampling_before_broadening: boolean, default: True
        Whether to resample wavelength and flux arrays onto a new wavelength
        scale with equidistant lambda sampling prior to broadening. This is
        required for using instrBroadGaussFast.
    
    fill_throughput_value: float, default: 0
        Default througput value to fill undefined wavelength points. Default is
        0, meaning that we assume no transmission for undefined values.

    planet_transmission_boost_fac: float, default: 1
        Whether to artificially boost the strength of planetary absorption by
        some factor for testing. By default set to 1 (i.e. disabled).

    Returns
    -------
    flux_counts, snr
        Modelled spectra flux in counts and the associated SNR.
    """
    # -------------------------------------------------------------------------
    # Integrated stellar flux
    # -------------------------------------------------------------------------
    print("\tSimulating stellar flux...")
    # Integrate stellar flux across entire disc so that we have a single 1D
    # spectrum that takes into account the effect of limb darkening (that is we
    # integrate over the μ dimension giving ergs/cm^2/s/Å).
    fluxes_disk = integrate_marcs_spectrum(wave_marcs, mus_marcs, fluxes_marcs)

    # Scale this 1D spectrum to match the light received at Earth (ergs/s/Å)
    fluxes_disk_scaled, flux_total_integrated = physically_scale_stellar_flux(
        wavelengths=wave_marcs,
        fluxes_mu=fluxes_marcs,
        fluxes_disk=fluxes_disk,
        r_star_r_sun=syst_info.loc["r_star_rsun", "value"],
        dist_pc=syst_info.loc["dist_pc", "value"],
        r_tel_prim=r_tel_prim,
        r_tel_cen_ob=r_tel_cen_ob,)

    # Doppler shift stellar flux to velocity γ_j
    stellar_flux_rv_shift = interpolate_spectrum(
        wave_new=wave_marcs*(1-transit_epoch["gamma"]),
        wave_old=wave_marcs,
        flux_old=fluxes_disk_scaled,
        fill_value=1.0,)

    # Broaden stellar flux to instrumental resolution
    stellar_flux_instr = instrBroadGaussFast(
        wvl=wave_marcs,
        flux=stellar_flux_rv_shift,
        resolution=instr_resolving_power,
        equid=do_equidistant_lambda_sampling_before_broadening,)

    # Interpolate stellar flux to instrumental wavelength scale
    stellar_flux_obs = interpolate_spectrum(
        wave_new=wave_observed,
        wave_old=wave_marcs,
        flux_old=stellar_flux_instr,)

    # -------------------------------------------------------------------------
    # Planet *is* transiting
    # -------------------------------------------------------------------------
    # Only do next two steps if planet is actually in-transit at the mid-point
    if transit_epoch["is_in_transit_mid"]:
        # ---------------------------------------------------------------------
        # Planet flux
        # ---------------------------------------------------------------------
        print("\tSimulating planet flux...")
        # Boost the strength of the planet spectrum for testing [optional]
        trans_planet = 1 - planet_transmission_boost_fac*(1-trans_planet)

        # Doppler shift planet flux
        planet_flux_rv_shift = interpolate_spectrum(
            wave_new=wave_planet*(1-transit_epoch["delta"]),
            wave_old=wave_planet,
            flux_old=trans_planet,
            fill_value=1.0,)

        # Smear planet flux. The physically realistic way to do this is create
        # and combine flux from number of 'sub-exposures' where each is shifted
        # to the appropriate RV based on the planet orbit. In practice though,
        # the difference in velocity over an exposure is generally below that
        # of the instrumental resolution, so this level of precision should not
        # be required. As such, for now we simply calculate the resolving power
        # R based on a spectral resolution delta_rv, and smear with a Gaussian
        # as before.
        delta_rv = \
            np.abs(transit_epoch["v_y_start"] - transit_epoch["v_y_end"])
        smear_R = const.c.value / (delta_rv*1000)

        if smear_R < instr_resolving_power:
            print("Warning--smeared R is < instrument R.")

        planet_flux_smeared = instrBroadGaussFast(
            wvl=wave_planet,
            flux=planet_flux_rv_shift,
            resolution=smear_R,
            equid=do_equidistant_lambda_sampling_before_broadening,)

        # Convolve planet flux to instrumental resolution
        planet_flux_instr = instrBroadGaussFast(
            wvl=wave_planet,
            flux=planet_flux_smeared,
            resolution=instr_resolving_power,
            equid=do_equidistant_lambda_sampling_before_broadening,)

        # Interpolate planet flux to instrumental wavelength scale
        planet_spec_obs = interpolate_spectrum(
            wave_new=wave_observed,
            wave_old=wave_planet,
            flux_old=planet_flux_instr,)

        # ---------------------------------------------------------------------
        # Blocked stellar flux (assuming an entirely opaque planet)
        # ---------------------------------------------------------------------
        print("\tSimulating blocked flux...")
        # Interpolate flux to the mu value of the planet centre
        # TODO: at small planet sizes this is a valid approximation, but the
        # better solution would be to integrate the flux of an arbitrary circle
        # with radius r_planet and central coordinates x and y within the 
        # stellar disk.
        flux_at_mu = interpolate_marcs_spectrum_at_mu(
            fluxes_mu=fluxes_marcs,
            mus=mus_marcs,
            mu_selected=transit_epoch["mu_mid"],)
        
        # Scale this flux such that it represents the total flux blocked by the
        # planet assuming it were entirely opaque
        r_p = syst_info.loc["r_planet_rearth", "value"] * const.R_earth.value
        r_s = syst_info.loc["r_star_rsun", "value"] * const.R_sun.value

        flux_at_mu_integrated = simps(x=wave_marcs, y=flux_at_mu)
        area_ratio = (transit_epoch["planet_area_frac_mid"] * r_p**2 / r_s**2)
        scale_fac = area_ratio * flux_total_integrated / flux_at_mu_integrated
        flux_shadow_opaque = flux_at_mu * scale_fac

        # Doppler shift stellar flux to velocity β_j
        flux_shadow_opaque_rv_shift = interpolate_spectrum(
            wave_new=wave_marcs*(1-transit_epoch["beta"]),
            wave_old=wave_marcs,
            flux_old=flux_shadow_opaque,
            fill_value=1.0,)

        # Broaden stellar flux to instrumental resolution
        flux_shadow_opaque_instr = instrBroadGaussFast(
            wvl=wave_marcs,
            flux=flux_shadow_opaque_rv_shift,
            resolution=instr_resolving_power,
            equid=do_equidistant_lambda_sampling_before_broadening,)

        # Interpolate stellar flux to instrumental wavelength scale
        flux_shadow_opaque_obs = interpolate_spectrum(
            wave_new=wave_observed,
            wave_old=wave_marcs,
            flux_old=flux_shadow_opaque_instr,)

        # ---------------------------------------------------------------------
        # Transmitted stellar flux (through planet atmosphere)
        # ---------------------------------------------------------------------
        print("\tSimulating transmitted flux...")
        # Here we use the same assumed mu value as before, but instead scale
        # thee spectrum by the adopted thickness of the planet atmosphere.
        planet_atmo_frac = (syst_info.loc["r_planet_atmo_r_earth", "value"]**2
            / syst_info.loc["r_planet_rearth", "value"]**2 )
        
        flux_shadow_atmo = flux_shadow_opaque * planet_atmo_frac

        # Doppler shift stellar flux to velocity β_j
        flux_shadow_atmo_rv_shift = interpolate_spectrum(
            wave_new=wave_marcs*(1-transit_epoch["beta"]),
            wave_old=wave_marcs,
            flux_old=flux_shadow_atmo,
            fill_value=1.0,)

        # Broaden stellar flux to instrumental resolution
        flux_shadow_atmo_instr = instrBroadGaussFast(
            wvl=wave_marcs,
            flux=flux_shadow_atmo_rv_shift,
            resolution=instr_resolving_power,
            equid=do_equidistant_lambda_sampling_before_broadening,)

        # Interpolate stellar flux to instrumental wavelength scale
        flux_shadow_atmo_obs = interpolate_spectrum(
            wave_new=wave_observed,
            wave_old=wave_marcs,
            flux_old=flux_shadow_atmo_instr,)

    # -------------------------------------------------------------------------
    # Planet *not* transiting
    # -------------------------------------------------------------------------
    else:
        print("\tPlanet not transiting")
        planet_spec_obs = 0
        flux_shadow_opaque_obs = 0
        flux_shadow_atmo_obs = 0

    # -------------------------------------------------------------------------
    # Tellurics
    # -------------------------------------------------------------------------
    print("\tSimulating tellurics absorption...")
    # Interpolate telluric spectrum - TODO: doppler shift needed?
    tau_obs = interpolate_spectrum(
        wave_new=wave_observed,
        wave_old=telluric_wave,
        flux_old=telluric_tau,)

    # -------------------------------------------------------------------------
    # Combine components
    # -------------------------------------------------------------------------
    print("\tCombining spectra...")
    # Apply planet to stellar spectrum
    flux_model_ob = calc_model_flux(
        stellar_flux=stellar_flux_obs,
        telluric_tau=tau_obs,
        planet_trans=planet_spec_obs,
        shadow_flux_opaque=flux_shadow_opaque_obs,
        shadow_flux_atmo=flux_shadow_atmo_obs,
        airmass=transit_epoch["airmass"],
        scale=1,)                                   # TODO: do properly
    
    # -------------------------------------------------------------------------
    # Instrumental transfer function, convert to counts
    # -------------------------------------------------------------------------
    print("\tConverting to instrumental scale...")
    # Scale by the observational transfer function (i.e. telescope, instrument,
    # grating including blaze, and detector throughputs)
    flux_instrumental = apply_instrumental_transfer_function(
        wave=wave_observed,
        flux=flux_model_ob,
        throughput_json_path=throughput_json_path,
        fill_throughput_value=fill_throughput_value,)

    # Convert ergs/s to counts/sec
    flux_counts_per_sec = convert_flux_to_counts(
        wave=wave_observed,
        flux=flux_instrumental,
        gain=2,)                                   # TODO: do properly
    
    # Convert counts/sec to counts
    flux_counts = flux_counts_per_sec * transit_epoch["exptime_sec"]

    # Compute SNR per spectral segment
    snr = flux_counts / flux_counts**0.5

    # -------------------------------------------------------------------------
    # Add noise [optional, TODO]
    # -------------------------------------------------------------------------
    pass

    # All done
    return flux_counts, snr


def simulate_transit_multiple_epochs(
    wave_observed,
    syst_info,
    transit_info,
    marcs_fits,
    planet_wave_fits,
    planet_spec_fits,
    molecfit_fits,
    throughput_json_path,
    wl_min,
    wl_max,
    instr_resolving_power,
    r_tel_prim,
    r_tel_cen_ob,
    do_equidistant_lambda_sampling_before_broadening,
    fill_throughput_value,
    tau_fill_value,
    planet_transmission_boost_fac,
    do_use_uniform_stellar_spec,
    do_use_uniform_telluric_spec,
    do_use_uniform_planet_spec,):
    """Simulates an entire transit with many epochs using multiple calls to
    simulate_transit_single_epoch. See docstrings of calc_model_flux and
    simulate_transit_single_epoch for more detail.

    Parameters
    ----------
    wave_observed: 1D or 2D float array
        Observed/instrumental wavelength scale of shape [n_wave] for a single 
        spectral segment, or [n_spec, n_wave] in the case of multiple orders or
        detectors.

    syst_info: pandas DataFrame
        DataFrame containing planet/star/system properties. The data frame has
        columns ['value', 'sigma', 'reference', 'comment'] and indices:
        ['m_star_msun', 'r_star_rsun', 'k_star_mps', 'dist_pc', 'vsini',
         'rv_star', 'ldc_init_a1', 'ldc_init_a2', 'ldc_init_a3', 'ldc_init_a4',
         'a_planet_au', 'e_planet', 'i_planet_deg', 'omega_planet_deg',
         'w_planet_deg', 'k_star_mps', 'transit_dur_hours', 'jd0_days', 
         'period_planet_days', 'm_planet_mearth', 'r_planet_rearth', 
         'r_planet_atmo_r_earth'].

    transit_info: pandas DataFrame
        DataFrame containing information associated with each transit time
        step. This DataFrame has columns:

        ['mjd_start', 'mjd_mid', 'mjd_end', 'jd_start', 'jd_mid', 'jd_end',
         'airmass', 'bcor', 'hcor', 'ra', 'dec', 'exptime_sec', 'nod_pos',
         'raw_file', 'phase_start', 'is_in_transit_start', 'r_x_start',
         'r_y_start', 'r_z_start', 'v_x_start', 'v_y_start', 'v_z_start',
         's_projected_start', 'scl_start', 'mu_start',
         'planet_area_frac_start', 'phase_mid', 'is_in_transit_mid',
         'r_x_mid', 'r_y_mid', 'r_z_mid', 'v_x_mid', 'v_y_mid', 'v_z_mid',
         's_projected_mid', 'scl_mid', 'mu_mid', 'planet_area_frac_mid',
         'phase_end', 'is_in_transit_end', 'r_x_end', 'r_y_end', 'r_z_end',
         'v_x_end', 'v_y_end', 'v_z_end', 's_projected_end', 'scl_end',
         'mu_end', 'planet_area_frac_end', 'gamma', 'beta', 'delta']

    marcs_fits, planet_wave_fits, planet_spec_fits, molecfit_fits,
    throughput_json_path: string
        Filepaths for MARCS spectrum, planet wave, planet spectra, molecfit
        best fit model, and CRIRES+ throughput files.

    wave_min, wave_max: float or None, default: None
        Minimum and maximum wavelengths in Angstrom to use when loading MARCS
        spectrum.

    instr_resolving_power: float
        Instrumental resolving power to simulate.

    r_tel_prim, r_tel_cen_ob: float
        Radii of telescope primary mirror and central obstruction in metres.

    do_equidistant_lambda_sampling_before_broadening: boolean, default: True
        Whether to resample wavelength and flux arrays onto a new wavelength
        scale with equidistant lambda sampling prior to broadening. This is
        required for using instrBroadGaussFast.
    
    fill_throughput_value: float, default: 0
        Default througput value to fill undefined wavelength points. Default is
        0, meaning that we assume no transmission for undefined values.
    
    tau_fill_value: float, default: 0
        Default value for missing values in the molecfit fitted spectrum.

    planet_transmission_boost_fac: float, default: 1
        Whether to artificially boost the strength of planetary absorption by
        some factor for testing. By default set to 1 (i.e. disabled).

    do_use_uniform_stellar_spec: boolean
        If True, the imported MARCS stellar spectrum is set to its median
        value. Note that this will cause limb darkening to cease to be
        meaningful, so only use this for testing.
    
    do_use_uniform_telluric_spec: boolean
        If True, the imported telluric spectrum is set to have 100% 
        transmittance. Note that this will remove any airmass dependence, so
        only use this for testing.

    do_use_uniform_planet_spec: boolean
        If True, the imported planet spectrum is set to have 100% 
        transmittance.

    Returns
    -------
    fluxes_model, snr: float array
        Modelled flux and SNR arrays corresponding to wavelengths of shape 
        [n_phase, n_spec, n_wave].
    """
    # Input checking
    if wl_min > np.min(wave_observed) or wl_max < np.max(wave_observed):
        raise ValueError("MARCS wl bounds do not cover observed wl scale.")
    
    # Import MARCS fluxes (ergs/cm^2/s/Å/μ)
    wave_marcs, _, _, mus_marcs, fluxes_marcs = read_marcs_spectrum(
        filepath=marcs_fits,
        wave_min=wl_min,
        wave_max=wl_max,)
    
    if do_use_uniform_stellar_spec:
        fluxes_marcs = np.ones_like(fluxes_marcs) * np.nanmedian(fluxes_marcs)
    
    # Import planet flux
    wave_planet, trans_planet = load_planet_spectrum(
        wave_file=planet_wave_fits,
        spec_file=planet_spec_fits,)
    
    if do_use_uniform_planet_spec:
        trans_planet = np.ones_like(trans_planet)
    
    # Load telluric spectrum
    telluric_wave, telluric_tau, _ = load_telluric_spectrum(
        molecfit_fits=molecfit_fits,
        tau_fill_value=tau_fill_value,)
    
    if do_use_uniform_telluric_spec:
        telluric_tau = np.zeros_like(telluric_tau)

    # Initialise output flux array
    shape = (len(transit_info), wave_observed.shape[0], wave_observed.shape[1])
    fluxes_model_all = np.zeros(shape)
    snr_model_all = np.zeros(shape)
    
    # Loop over all phases and determine fluxes
    for epoch_i, transit_epoch in transit_info.iterrows():
        print("Simulating epoch {}...".format(epoch_i))
        flux_counts, snr = simulate_transit_single_epoch(
            wave_observed=wave_observed,
            wave_marcs=wave_marcs,
            fluxes_marcs=fluxes_marcs,
            mus_marcs=mus_marcs,
            wave_planet=wave_planet,
            trans_planet=trans_planet,
            telluric_wave=telluric_wave,
            telluric_tau=telluric_tau,
            syst_info=syst_info,
            transit_epoch=transit_epoch,
            instr_resolving_power=instr_resolving_power,
            r_tel_prim=r_tel_prim,
            r_tel_cen_ob=r_tel_cen_ob,
            do_equidistant_lambda_sampling_before_broadening=\
                do_equidistant_lambda_sampling_before_broadening,
            throughput_json_path=throughput_json_path,
            fill_throughput_value=fill_throughput_value,
            planet_transmission_boost_fac=planet_transmission_boost_fac,)
        
        fluxes_model_all[epoch_i] = flux_counts
        snr_model_all[epoch_i] = snr

    return fluxes_model_all, snr_model_all