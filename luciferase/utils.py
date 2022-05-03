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