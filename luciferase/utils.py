"""Utilities functions.
"""
import numpy as np
import glob
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation

def sort_spectra_by_time(spectra_path_with_wildcard):
    """Imports time series spectra from globbed fits files given a filepath 
    with wildcard, and sorts by time, returning the sorted list of files.
    """
    # Grab filenames and sort
    spec_seq_files = glob.glob(spectra_path_with_wildcard)

    times = []

    for spec_file in spec_seq_files:
        times.append(fits.getval(spec_file, "DATE-OBS"))

    times = np.array(times)
    sorted_i = np.argsort(times)

    spec_seq_files = np.array(spec_seq_files)[sorted_i]

    return spec_seq_files, times


def compute_barycentric_correction(
    ra,
    dec,
    time_mid,
    site="Paranal", 
    disable_auto_max_age=False,
    overrid_iers=False):
    """Compute the barycentric corrections for a set of stars

    In late 2019 issues were encountered accessing online files related to the
    International Earth Rotation and Reference Systems Service. This is 
    required to calculate barycentric corrections for the data. This astopy 
    issue may prove a useful resource again if the issue reoccurs:
    - https://github.com/astropy/astropy/issues/8981

    Parameters
    ----------
    ra: string
        Right ascension in degrees.
    
    dec: string
        Declination in degrees.

    time_mid: float
        Mid-point of observation in MJD format.

    site: string
        The site name to look up its coordinates.

    disable_auto_max_age: boolean, defaults to False
        Useful only when IERS server is not working.

    Returns
    -------
    bcor: float 
        Barycentric correction in km/s.
    """
    # Get the location
    loc = EarthLocation.of_site(site)

    if disable_auto_max_age:
        #from astropy.utils.iers import IERS_A_URL
        IERS_A_URL = 'ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals.all'
        #from astropy.utils.iers import conf
        #conf.auto_max_age = None
    
    # Override the IERS server if the mirror is down or not up to date
    if overrid_iers:
        from astropy.utils import iers
        from astropy.utils.iers import conf as iers_conf
        url = "https://datacenter.iers.org/data/9/finals2000A.all"
        iers_conf.iers_auto_url = url
        iers_conf.reload()

    # Get the onsky coordinates
    sc = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))

    # Compute the barycentric correction for the star
    time = Time(float(time_mid), format="mjd")
    barycorr = sc.radial_velocity_correction(obstime=time, location=loc)  
    bcor = barycorr.to(u.km/u.s).value

    return bcor

def load_plumage_template_spectrum(
    template_fits,
    do_convert_angstroms_to_nm=True,
    do_convert_air_to_vacuum_wl=False,):
    """Imports a plumage sourced synthetic template spectrum in the form of
    wavelength and spectrum vectors.

    Parameters
    ----------
    template_fits: string
        Filepath of the fits file to load. It should have three extensions:
        [WAVE, SPEC, and PARAMS]
    
    convert_angstroms_to_nm: boolean, default: True
        If True, assumes that the template spectrum wavelength scale is in
        Angstroms and divides by 10 to convert it to nm.

    do_convert_air_to_vacuum_wl: boolean, default: False
        Whether to convert air to vacuum wavelengths on import.

    Returns
    -------
    wave, spec: float array
        Template wavelength and spectrum arrays.
    """
    with fits.open(template_fits) as tfits:
        wave = tfits["WAVE"].data
        spec = tfits["SPEC"].data[0]    # By default this array is 2D

        if do_convert_air_to_vacuum_wl:
            wave = convert_air_to_vacuum_wl(wave)

        if do_convert_angstroms_to_nm:
            wave /= 10
        
    return wave, spec


def convert_air_to_vacuum_wl(wavelengths_air,):
    """Converts provided air wavelengths to vacuum wavelengths using the 
    formalism described here: 
        https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

    Parameters
    ----------
    wavelengths_air: float array
        Array of air wavelength values to convert.

    Returns
    -------
    wavelengths_vac: float array
        Corresponding array of vacuum wavelengths
    """
    # Calculate the refractive index for every wavelength
    ss = 10**4 / wavelengths_air
    n_ref = (1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - ss)
         + 0.0001599740894897 / (38.92568793293 - ss))

    # Calculate vacuum wavelengths
    wavelengths_vac = wavelengths_air * n_ref

    return wavelengths_vac