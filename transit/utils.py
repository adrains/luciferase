"""Utilities functions associated with transit modelling.

Originally written in IDL by Nikolai Piskunov, ported to Python by Adam Rains.
"""
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
from astropy.table import Table
from astropy import constants as const
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.stats import sigma_clip
from scipy.interpolate import interp1d

# -----------------------------------------------------------------------------
# Kepler's laws
# -----------------------------------------------------------------------------
def compute_kepler_eccentric_anomaly(M, e, err, max_count=15):
    """Returns the solution x=eccentric anomaly to Kepler's elliptic equation
    f(x) = x - e*sin(x) - M = 0 via Halley's method to an accuracy of 
    |f(x)|<err. I suspect this code came from Danby's textbook.

    Based on https://github.com/joehahn/el2xv/blob/master/kepler.pro, which is
    the IDL code Nik Piskunov used for the IDL implementation.
    
    Parameters
    ----------
    M: float
        Mean anomaly in radians.
    
    e: float
        Orbital eccentricity, 0 < e < 1.

    err: float
        Allowed error such that |f(x)| < err.

    Returns
    -------
    x: float array
        Eccentric anomaly
    """
    # Initial guess
    twopi = 2*np.pi
    M = M - twopi*int(M/twopi)

    if np.sin(M) < 0:
        s = -1
    else:
        s = 1
    
    #s = np.ones_like(M)
    #sinM = np.sin(M)
    #j = np.where(sinM < 0)
    #if j[0] != -1:
    #    s[j] = -1

    x = M + s*(0.85)*e
    f = 1

    count = 0

    while (np.abs(f) > err ) and (count < max_count):
        es = e*np.sin(x)
        ec = e*np.cos(x)
        f = x - es - M
        df = 1 - ec
        ddf = es
        dddf = ec
        d1 = -f/df
        d2 = -f/(df + d1*ddf/2)
        d3 = -f/(df + d2*ddf/2 + d2*d2*dddf/6)
        x = x + d3
        count = count + 1

    if (count > max_count):
        print("Kepler failed to converge")

    return x

def compute_orbit_cartesian_pos_and_vel(
    M_star,
    a,
    e,
    I,
    O,
    w,
    M,
    M_planet=0,
    error=1E-8):
    """Convert orbit elements a,e,I,O,w,M to cartesian coordinates and 
    velocities x,y,z,vx,vy,vx. This routine also calls kepler to solve Kepler's
    equation, and the optional error parameter is the tolerance in that
    solution and defaults to error=1d-8. ;Formulas used here probably come from
    textbook by Murray & Dermott or Danby.

    Based on https://github.com/joehahn/el2xv/blob/master/el2xv.pro, which is
    the IDL code Nik Piskunov used for the IDL implementation.

    Parameters
    ----------
    M_star: float
        Mass of central body in M_sol.

    a: float
        Semimajor axis in AU.

    e: float
        Orbital eccentricity, 0 < e < 1.

    I: float
        Orbital inclination in radians.

    O: float
        Longitude of ascending node in radians.
    
    w: float
        Argument of periapse in radians.

    M: float
        Mean anomaly in radians.

    M_planet: float, default: 0
        Mass of the orbiting body in M_sol.

    error: float, default: 1E-8
        Convergence tolerance for computation of the eccentric anomaly.

    Returns
    -------
    r_xyz: float array
        Cartesian position vector in units of AU.

    v_xyz: float array
        Cartesian velocity vector in units of AU/(2*pi*year).
    """
    # Calculate the rotation matrices
    so = np.sin(O)
    co = np.cos(O)
    sp = np.sin(w)
    cp = np.cos(w)
    si = np.sin(I)
    ci = np.cos(I)
    d11 =  cp*co - sp*so*ci
    d12 =  cp*so + sp*co*ci
    d13 =  sp*si
    d21 = -sp*co - cp*so*ci
    d22 = -sp*so + cp*co*ci
    d23 =  cp*si

    # Storage
    #xx = np.zeros_like(a)
    #yy = np.zeros_like(a)
    #vxx = np.zeros_like(a)
    #vyy = np.zeros_like(a)

    # Choose appropriate GM
    GM = M_star + M_planet

    # for elliptic elements
    if (a > 0):
        # Solve kepler's eqn.
        EA = compute_kepler_eccentric_anomaly( M, e, error)
        cE = np.cos(EA)
        sE = np.sin(EA)

        # coordinates in orbit frame
        e_root = np.sqrt( 1 - e**2 )
        xx = a*( cE - e )
        yy = a*e_root*sE 
        r = a*( 1 - e*cE )
        na2 = np.sqrt( GM*a )
        vxx = -na2*sE/r 
        vyy = na2*e_root*cE/r 

    else:
        raise ValueError("Semi-major axis should be positive.")

    """
    # for hyperbolic elements
    if (a[j] < 0):

        # solve kepler's eqn.
        F = hyper_kepler( M[j], e[j], error)
        chF = np.cosh(F)
        shF = np.sinh(F)

        # coordinates in orbit frame
        sqe = np.sqrt( e[j]^2 - 1d ) 
        sqgma = np.sqrt( -GM*a[j] )
        ri = -1d/( a[j]*( e[j]*chF - 1d ) )
        xx[j] = -a[j]*( e[j] - chF )
        yy[j] = -a[j]*sqe*shF 
        vxx[j] = -ri*sqgma*shF 
        vyy[j] = ri*sqgma*sqe*chF
    """

    # Rotate to reference frame
    r_xyz  = np.array([
        d11*xx + d21*yy,
        d12*xx + d22*yy,
        d13*xx + d23*yy,])

    v_xyz = np.array([
        d11*vxx + d21*vyy,
        d12*vxx + d22*vyy,
        d13*vxx + d23*vyy,])

    return r_xyz, v_xyz

# -----------------------------------------------------------------------------
# General Functions
# -----------------------------------------------------------------------------
def bezier_init(x, y,):
    """
    Computes the y derivative necessary to compute Bezier splines. Companion
    function to bezier_interp.

    Note: this function requires that the wavelength array is monotonically 
    *increasing* and not decreasing.

    Maths
    -----
    If we define for points x_a and x_b along a ray:
            u = (x - x_a)/(x_b - x_a)
    then any function can be fit with a Bezier spline as
            f(u) = f(x_a)*(1 - u)^3 + 3*c_0*u*(1-u)^2 + 3*c_1*u^2*(1-u) 
                   + f(x_b)*u^3
    where c_0 and c_1 are the local control parameters.

    Control parameters for interval [x_a, x_b] are computed as:
            c_0 = f(x_a) + delta/3*D'_a
    and
            c_1 = f(x_b) - delta/3*D'_b

            If D(b-1/2)*D(b+1/2) > 0 then
            D'_b    = D(b-1/2)*D(b+1/2) / (alpha*D(b+1/2) + (1-alpha)*D(b-1/2))
            Else
            D'_b    = 0

            D(b-1/2) = [f(x_b) - f(x_a)] / delta
            D(b+1/2) = [f(x_c) - f(x_b)] / delta'
            alpha        = [1 + delta'/(delta + delta')]/3
            delta        = x_b - x_a
            delta'     = x_c - x_b

    For the first and the last step we assume D(b-1/2)=D(b+1/2) and, therefore,
    D'_b = D(b+1/2) for the first point and
    D'_b = D(b-1/2) for the last point

    The actual interpolation is split in two parts. This function computes the
    derivative D'_b, wheras bezier_interp performs the actual interpolation.

    Parameters
    ----------
    x, y: float array
        x and y values to compute the derivative of.

    Returns
    -------
    y2: float array
        dy/dx associated with x and y.
    """
    # Perform check on whether our input array is monotonically increasing
    if not np.all(np.diff(x) > 0):
        raise ValueError("x_a array should be monotonically increasing.")

    # Ensure arrays have the same shape
    if len(x) != len(y):
        raise ValueError("Input X and Y arrays must have same lengths!")

    n_x = len(x)

    y2 = np.zeros(n_x)
    h_2 = x[1] - x[0]
    deriv_2 = (y[1]-y[0])/h_2
    y2[0] = deriv_2

    # This should exclude the two end points
    for x_i in range(1, n_x-1):
        h_1 = h_2
        deriv_1 = deriv_2
        h_2 = (x[x_i+1]-x[x_i])
        deriv_2 = (y[x_i+1] - y[x_i]) / h_2
        alpha = (1 + h_2/(h_1+h_2))/3

        if (deriv_1*deriv_2 > 0.0):
            y2[x_i] = deriv_1*deriv_2/(alpha*deriv_2+(1.0-alpha)*deriv_1)
        else:
            y2[x_i] = 0.0

    y2[n_x-1] = deriv_2

    return y2


def bezier_interp(x_a, y_a, y2_a, x_interp,):
    """
    Performs cubic Bezier spline interpolation.

    Note: this function requires that the wavelength array is monotonically 
    *increasing* and not decreasing.

    Maths
    -----
    If we define for points x_a and x_b along a ray:
            u = (x - x_a)/(x_b - x_a)
    then any function can be fit with a Bezier spline as
            f(u) = f(x_a)*(1 - u)^3 + 3*c_0*u*(1-u)^2 + 3*c_1*u^2*(1-u) 
                   + f(x_b)*u^3
    where c_0 and c_1 are the local control parameters.

    Control parameters for interval [x_a, x_b] are computed as:
            c_0 = f(x_a) + delta/3*D'_a
    and
            c_1 = f(x_b) - delta/3*D'_b

            If D(b-1/2)*D(b+1/2) > 0 then
            D'_b    = D(b-1/2)*D(b+1/2) / (alpha*D(b+1/2) + (1-alpha)*D(b-1/2))
            Else
            D'_b    = 0

            D(b-1/2) = [f(x_b) - f(x_a)] / delta
            D(b+1/2) = [f(x_c) - f(x_b)] / delta'
            alpha        = [1 + delta'/(delta + delta')]/3
            delta        = x_b - x_a
            delta'     = x_c - x_b

    For the first and the last step we assume D(b-1/2)=D(b+1/2) and, therefore,
    D'_b = D(b+1/2) for the first point and
    D'_b = D(b-1/2) for the last point

    The actual interpolation is split in two parts. This function performs the
    interpolation, and bezier_init calculates the derivative D'_b.

    Parameters
    ----------
    x_a, y_a, y2_a: float array
        Original x, y, and dy/dx arrays to interpolate. y2_a should be from the
        function bezier_init.

    x_interp: float array
        New x array to interpolate onto.

    Returns
    -------
    y_interp: float array
        Interpolated y values corresponding to x_interp.
    """
    # Perform check on whether our input array is monotonically increasing
    if not np.all(np.diff(x_a) > 0):
        raise ValueError("x_a array should be monotonically increasing.")

    # Ensure arrays have the same shape
    if len(x_a) != len(y_a) and len(y_a) != len(y2_a):
        raise ValueError("Input X, Y, and Y2 arrays must have same lengths!")
    
    # Initialise output array for interpolated Y values
    y_interp = np.zeros_like(x_interp)
    
    # Ensure interpolation is possible: new x array must have values within
    # our original x array. If this is not the case, the interpolation fails
    # and we will return an array of zeroes.
    x_in_range = np.where(
        np.logical_and(
            x_interp >= np.min(x_a),
            x_interp <= np.max(x_a)))

    if np.sum(x_in_range) == 0:
        print("Interpolation failed--no valid wavelengths provided.")
        return y_interp

    # Locate the *lower* indices bounding the values in x_interp. Unlike the
    # original IDL function value_locate, np.searchsorted does not distinguish
    # values below the bounds of x_a by inserting a -1. However, the original
    # IDL code was k_low = ((value_locate(x_a, x_interp))<(n_x-2L))>0 which
    # serves to clip it to 0 and the len(x_a)-1.
    k_low = np.searchsorted(x_a, x_interp)

    # Restrict the maximum value of k_low to be len(x_a) - 1
    k_low[k_low == len(x_a)] = len(x_a) - 1

    # For the *upper* indices we simply index this by 1
    k_high = k_low + 1

    # Compute the x separation between each of these upper and lower bounds
    h_sep = x_a[k_high] - x_a[k_low]

    # Locate the y values associated with these bounds
    y_1 = y_a[k_low]
    y_2 = y_a[k_high]
    
    # Compute fractional x distances between each bound and the central point
    # to interpolate
    aa = (x_a[k_high] - x_interp) / h_sep
    bb = (x_interp - x_a[k_low]) / h_sep

    # Compute control parameters for each interval
    c_0 = y_1 + h_sep/3*y2_a[k_low]
    c_1 = y_2 - h_sep/3*y2_a[k_high]

    # Perform the interpolation
    y_interp = (aa**3)*y_1 + 3*(aa**2)*bb*c_0 + 3*aa*(bb**2)*c_1 + (bb**3)*y_2
    
    return y_interp


def cross_correlate_nodded_spectra():
    """
    """
    pass


def doppler_shift(x, y, gamma, y2):
    """Doppler shift a provided flux array. Note that this function currently
    mirrors the ends for pixels not in the original array.

    TODO: replace mirroring with NaNs using standard scipy interpolation.

    Parameters
    ----------
    x: 1D float array
        Wavelength vector of shape [n_wave].

    y: 1D float array
        Flux vector of shape [n_wave].

    gamma: float
       Unitless Doppler shift, where a positive value of gamma means red shift.

    y2: 1D float array
        Flux derivative of shape [n_wave].

    Returns
    -------
    flux_rv_shifted: 1D float array
        Doppler shifted flux vector of shape [n_wave].
    """
    # Positive gamma means red shift
    n = len(x)

    # TODO: Nik's original function allowed for y2 to not have the same length
    # as x, but it seems that would never be the case? This assertion is here
    # to check this.
    assert len(x) == len(y) and len(y) == len(y2)

    # Compute steps in wavelength and padding in pixels
    dx1 = x[1] - x[0]
    pad1 = np.max(np.ceil(-gamma*x[0]/dx1), 0)

    dx2 = x[-1] - x[-2]
    pad2 = np.max(np.ceil(gamma*x[-1]/dx2), 0)

    # Pad arrays to avoid extrapolation
    # For the spectrum mirror points relative to the ends
    if pad1 == 0 and pad2 == 0:
        xx = x
        yy = y
        yy2 = y2

    # Mirror the *end* of the array
    elif pad1 == 0 and pad2 > 0:
        x_pad = x[-1] + (np.arange(pad2)+1)*dx2
        xx = np.concatenate((x, x_pad))

        y_pad = y[n-np.arange(pad2)-2]
        yy = np.concatenat((y, y_pad))
        
        yy2_pad = -y2[-np.arange(pad2)-2]
        yy2 = np.concatenate((y2, yy2_pad))

    # Mirror the *start* of the array
    elif pad1 > 0 and pad2 == 0:
        x_pad = x[0] - (pad1-np.arange(pad1))*dx1
        xx = np.concatenate((x_pad, x))

        y_pad = y[pad1-np.arange(pad1)]
        yy = np.arange((y_pad, y))

        yy2_pad = -y2[pad1-np.arange(pad1)]
        yy2 = np.concatenate((yy2_pad, y2))

    # Mirror *both* ends of the array
    elif pad1 > 0 and pad2 > 0:
        x_pad_start = x[0]-(pad1-np.arange(pad1))*dx1
        x_pad_end = x[n-1]+(np.arange(pad2)+1)*dx2
        xx = np.concatenate((x_pad_start, x, x_pad_end))

        y_pad_start = y[pad1-np.arange(pad1)]
        y_pad_end = y[n-np.arange(pad2)-2]
        yy = np.concatenate((y_pad_start, y, y_pad_end))

        yy2_pad_start = -y2[pad1-np.arange(pad1)]
        yy2_pad_end = -y2[n-np.arange(pad2)-2]
        yy2 = np.concatenate((yy2_pad_start, y2, yy2_pad_end))

    # Bezier interpolation
    flux_rv_shifted = bezier_interp(xx, yy, yy2, x*(1.0+gamma))

    return flux_rv_shifted

    """
    if len(y2) == n:
    
    else:
        # Compute steps in wavelength and padding in pixels
        dx1 = x[1]-x[0]
        pad1 = fix(gamma*x[0]/dx1)<0           # TODO <>
        dx2 = x[n-1]-x[n-2]
        pad2 = fix(gamma*x[n-1]/dx2)>0         # TODO <>

        # Pad arrays to avoid extrapolation
        # For the spectrum mirror points relative to the ends
        if pad1 == 0 and pad2 == 0:
            xx=x
            yy=y
        elif pad1 == 0 and pad2 > 0:
            xx=[x,x[n-1]+(np.arange(pad2)+1)*dx2]
            yy=[y,y[n-np.arange(pad2)-2]]

        elif pad1 > 0 and pad2 == 0:
            xx=[x[0 ]-(pad1-np.arange(pad1))*dx1,x]
            yy=[y[pad1-np.arange(pad1)],y]

        elif pad1 > 0 and pad2 > 0:
            xx=[x[0 ]-(pad1-np.arange(pad1))*dx1,x,x[n-1]+(np.arange(pad2)+1)*dx2]
            yy=[y[pad1-np.arange(pad1)],y,y[n-np.arange(pad2)-2]]

        # Bezier interpolation
        yy2 = bezier_init(xx,yy,)
        rr = bezier_interp(xx, yy, yy2, x*(1.0+gamma))
    """

# -----------------------------------------------------------------------------
# Preparation/Import Functions
# -----------------------------------------------------------------------------
def load_planet_properties(planet_properties_file):
    """Load in the planet properties from a text file and return a pandas 
    dataframe. The file should be a whitespace delimited data file with five
    columns [parameter, value, sigma, reference, comment] and the following 
    rows:
        m_star_msun
        r_star_rsun
        k_star_mps
        a_planet_au
        e_planet
        i_planet_deg
        omega_lan_planet_deg
        w_ap_planet_deg
        k_planet_mps
        jd0_days
        period_planet_days
        transit_dur_hours
        m_planet_mearth
        r_planet_rearth

    Strings with whitespace should be commented with double quote characters,
    and comment lines are those starting with #.

    Parameters
    ----------
    planet_properties_file: string
        File path for the planet data file to load.
        
    Returns
    -------
    syst_info: pandas DataFrame
        Pandas dataframe containing the planet properties.
    """
    # Check the file exists
    if not os.path.isfile(planet_properties_file):
        raise FileNotFoundError("Planet properties file does not exist!")

    # Load in planet properties
    syst_info = pd.read_csv(
        filepath_or_buffer=planet_properties_file,
        delim_whitespace=True,
        comment="#",
        quotechar='"',
        index_col="parameter",
        dtype={"value":float, "sigma":float})

    return syst_info


def extract_single_nodding_frame(reduced_fits, nod_pos, slit_pos_spec_num=1):
    """Extract information for a single nodding exposure and compute mus and 
    velocities. Information extracted:
     - A/B spectra (wavelengths, fluxes, sigmas)
     - Transit/pointing info (times, airmasses, phases)

    Since the CRIRES pipeline defaults to using the first header in a SOF 
    file for all reduced files, we need to go back to the raw A/B frames to 
    get the correct time/airmass pointing information.

    Parameters
    ----------
    reduced_fits: string
        Filepath of the raw file associated with the reduced frame.

    nod_pos: string
        Nodding position of this frame, either A or B.

    slit_pos_spec_num: int, default: 1
        Slit position index, only relevant for extended objects or multi-object
        spectroscopy where multiple positions in the slit have been extracted
        rather than just a single position.

    Returns
    -------
    nod_pos_dict: dict
        Dictionary containing the following key/value pairs for a single
        nodding position:
            'waves': float array of shape [n_phase, n_spec, n_px]
            'fluxes': float array of shape [n_phase, n_spec, n_px]
            'sigmas': float array of shape [n_phase, n_spec, n_px]
            'det_nums': float array of shape [n_spec]
            'order_nums': float array of shape [n_spec]
            'ra': float
            'dec': float
            'mjd': float
            'jd': float
            'airmass': float
            'phase': float
            'bary_corr': float
            'hel_corr': float
            'gamma': float
            'nod_pos': string
            'raw_file': string
    """
    if nod_pos not in ["A", "B"]:
        raise Exception("Invalid nodding position, must be either 'A' or 'B'.")
    
    # Get the filenames for the two raw science frames. Note that these are
    # *just* the filenames, not the paths. Our assumption, however, is that
    # all the raw files are simply in the directory above.
    raw_fits_1 = fits.getval(reduced_fits, "HIERARCH ESO PRO REC1 RAW1 NAME")
    raw_fits_2 = fits.getval(reduced_fits, "HIERARCH ESO PRO REC1 RAW2 NAME")

    # Inspect each of the raw frames, and continue working with whichever frame
    # matches our specified nod_pos
    raw_fits_root = os.path.join(os.path.split(reduced_fits)[0], "..")
    raw_fits_path_1 = os.path.join(raw_fits_root, raw_fits_1)
    raw_fits_path_2 = os.path.join(raw_fits_root, raw_fits_2)

    nod_pos_1 = fits.getval(raw_fits_path_1, "HIERARCH ESO SEQ NODPOS")
    nod_pos_2 = fits.getval(raw_fits_path_2, "HIERARCH ESO SEQ NODPOS")

    if nod_pos_1 == nod_pos:
        raw_fits_path = raw_fits_path_1

    elif nod_pos_2 == nod_pos:
        raw_fits_path = raw_fits_path_2

    # Something went wrong
    else:
        raise Exception("Nod positions don't match.")

    # Now that we have our raw fits file in the correct nodding position, grab
    # all of the header information we need
    with fits.open(raw_fits_path) as fits_file:
        # Extract and compute times at start, mid, and end points
        exptime = fits_file[0].header["HIERARCH ESO DET SEQ1 EXPTIME"]
        mjd_start = fits_file[0].header["MJD-OBS"]
        mjd_mid = mjd_start + (exptime / 3600 / 24)/2
        mjd_end = mjd_start + (exptime / 3600 / 24)

        jd_start = 2400000.5 + mjd_start
        jd_mid = 2400000.5 + mjd_mid
        jd_end = 2400000.5 + mjd_end

        # Get the start and end airmasses, + compute the midpoint
        airm_start = fits_file[0].header["HIERARCH ESO TEL AIRM START"]
        airm_end = fits_file[0].header["HIERARCH ESO TEL AIRM END"]
        airmass_mid = (airm_start+airm_end)*0.5

        ra2000 = fits_file[0].header["RA"]
        de2000 = fits_file[0].header["DEC"]
        equinox = "J{}".format(fits_file[0].header["EQUINOX"])
        obs_alt = fits_file[0].header["HIERARCH ESO TEL GEOELEV"]   # metres

        # [deg] Tel geo latitute (+=North)
        obs_lat = fits_file[0].header["HIERARCH ESO TEL GEOLAT"]

        # [deg] Tel geo longitude (+=East)
        obs_lon = fits_file[0].header["HIERARCH ESO TEL GEOLON"]

        # Initialise astropy coordinate objects
        vlt = EarthLocation.from_geodetic(
            lat=obs_lat*u.deg,
            lon=obs_lon*u.deg,
            height=obs_alt*u.m,)

        sc = SkyCoord(ra=ra2000*u.deg, dec=de2000*u.deg, equinox=equinox,)

        # Calculate the barycentric correction
        # Note that the astropy implementation is allegedly consistent with the
        # Wright and Eastman implementation:
        #  - https://ui.adsabs.harvard.edu/abs/2014PASP..126..838W/abstract
        #  - https://docs.astropy.org/en/stable/coordinates/velocities.html
        bary_corr = sc.radial_velocity_correction(
            kind="barycentric",
            obstime=Time(mjd_mid, format="mjd"),
            location=vlt,)

        # Note that we just store the barycentric correction for now, and deal
        # with the +/- sign convention later when constructing doppler shifts.
        bary_corr = bary_corr.to(u.km/u.s)

        # Calculate the heliocentric correction
        hel_corr = sc.radial_velocity_correction(
            kind="heliocentric",
            obstime=Time(mjd_mid, format="mjd"),
            location=vlt,)

        hel_corr = hel_corr.to(u.km/u.s)

    # Now that we've grabbed all the header information, go back to the 
    # redued file for the actual spectra
    with fits.open(reduced_fits) as fits_file:
        # Initialise arrays
        fits_ext_names = ["CHIP1.INT1", "CHIP2.INT1", "CHIP3.INT1"]
        wl_col_suffix = "_{:02.0f}_WL".format(slit_pos_spec_num)
        spec_col_suffix = "_{:02.0f}_SPEC".format(slit_pos_spec_num)
        sigma_col_suffix = "_{:02.0f}_ERR".format(slit_pos_spec_num)

        # Initialise arrays
        waves = []
        fluxes = []
        sigmas = []
        det_nums = []
        order_nums = []
        
        # Determine the spectral orders to consider. Note that not all 
        # detectors will necessarily have all orders, so we should pool the 
        # orders from all detectors, and then check before pulling from each.
        # TODO: get this from fits headers in a neater way. 
        columns = []

        for fits_ext_name in fits_ext_names:
            columns += fits_file[fits_ext_name].data.columns.names

        orders = list(set([int(cc.split("_")[0]) for cc in columns]))
        orders.sort()

        # Loop over each detector and order and extract data
        for det_i, fits_ext in enumerate(fits_ext_names):
            hdu_data = fits_file[fits_ext].data

            for order in orders:
                # First check this order exists for this detector
                if ("{:02.0f}{}".format(order, wl_col_suffix) 
                    not in hdu_data.columns.names):
                    print("Det {}, Order {} missing, skipping".format(
                        det_i+1, order))
                    continue

                wave = hdu_data["{:02.0f}{}".format(order, wl_col_suffix)]
                flux = hdu_data["{:02.0f}{}".format(order, spec_col_suffix)]
                sigma = hdu_data["{:02.0f}{}".format(order, sigma_col_suffix)]

                # Store
                waves.append(wave)
                fluxes.append(flux)
                sigmas.append(sigma)
                det_nums.append(det_i + 1)
                order_nums.append(order)

        # Convert to numpy format
        waves = np.vstack(waves)
        fluxes = np.vstack(fluxes)
        sigmas = np.vstack(sigmas)
        det_nums = np.array(det_nums)
        order_nums = np.array(order_nums)

    # Now we're all done, return everything in a dictionary
    nod_pos_dict = {
        "waves":waves,
        "fluxes":fluxes,
        "sigmas":sigmas,
        "det_nums":det_nums,
        "order_nums":order_nums,
        "ra":ra2000,
        "dec":de2000,
        "exptime_sec":exptime,
        "mjd_start":mjd_start,
        "mjd_mid":mjd_mid,
        "mjd_end":mjd_end,
        "jd_start":jd_start,
        "jd_mid":jd_mid,
        "jd_end":jd_end,
        "airmass":airmass_mid,
        "bcor":bary_corr.value,
        "hcor":hel_corr.value,
        "nod_pos":nod_pos,
        "raw_file":raw_fits_path,
    }

    return nod_pos_dict


def extract_nodding_time_series(
    root_dir,
    sub_dir_wildcard="1xAB_*",
    nod_ab_wildcard="cr2res_obs_nodding_extracted[AB].fits",
    is_verbose=True,):
    """Extract spectra and associated header/computed information for a series 
    of nodding observations assocated with a transit observation. This is done
    through repeated calls to extract_single_nodding_frame, and the end result
    is one array each for waves, fluxes, sigmas, detectors, and orders, as well
    as a single pandas DataFrame containing information about each timestep.

    Parameters
    ----------
    root_dir: string
        Base directory containing subfolders for each nodding pair.
    
    sub_dir_wildcard: string, default: '1xAB_*'
        Format of the subdirectories for each nodding pair with glob wildcard.

    nod_ab_wildcard: string, default: 'cr2res_obs_nodding_extracted[AB].fits'
        Format of A/B nodding frames with glob wildcard.

    Returns
    -------
    waves: float array
        Wavelength scales for each timestep of shape [n_phase, n_spec, n_px].
    
    fluxes: float array
        Flux array for each timestep of shape [n_phase, n_spec, n_px].
    
    sigmas: float array
        Uncertainty array for each timestep of shape [n_phase, n_spec, n_px].
    
    detectors: int array
        Array associating spectral segments to CRIRES+ detector number of shape 
        [n_spec].
    
    orders: int array
        Array associating spectral segments to CRIRES+ order number of shape 
        [n_spec].
    
    transit_df: pandas DataFrame
        Pandas DataFrame with header/computed information about each timestep.
        Has columns: [mjd, jd, phase, airmass, bcor, hcor, gamma, ra, dec, 
        nod_pos, raw_file] and is of length [n_phase].
    """
    # Find all subdirectories
    path_ab = os.path.join(root_dir, sub_dir_wildcard, nod_ab_wildcard)
    all_nodding_files = glob.glob(path_ab)
    all_nodding_files.sort()

    # Only continue if we've found an even and nonzero number of folders
    n_files = len(all_nodding_files)

    if n_files == 0:
        raise Exception("No folders found!")
    elif n_files % 2 != 0:
        raise Exception("Unmatched number of nodding pairs!")
    else:
        print("{} nodding pairs found.".format(n_files))

    # Initialise arrays (as lists). These will need to be sorted later.
    waves = []      # [n_phase, n_spec, n_px]
    fluxes = []     # [n_phase, n_spec, n_px]
    sigmas = []     # [n_phase, n_spec, n_px]
    detectors = []  # [n_spec]
    orders = []     # [n_spec]

    # Initialise pandas data frame for time series results
    df_cols = ["mjd_start", "mjd_mid", "mjd_end", "jd_start", "jd_mid", 
        "jd_end", "airmass", "bcor", "hcor", "ra", "dec", "exptime_sec", 
        "nod_pos", "raw_file"]

    transit_df = pd.DataFrame(
        data=np.full((n_files, len(df_cols)), np.nan),
        columns=df_cols,)

    # Loop over all folders
    for file_i, nodding_file in enumerate(
        tqdm(all_nodding_files, desc="Importing spectra", leave=False,)):
        # Determine which nodding position this file is
        if nod_ab_wildcard.replace("[AB]", "A") in nodding_file:
            nod_pos = "A"
        elif nod_ab_wildcard.replace("[AB]", "B") in nodding_file:
            nod_pos = "B"
        else:
            raise Exception("No A/B nod position found-check wildcard format.")

        # Extract data
        nod_pos_dict = extract_single_nodding_frame(nodding_file, nod_pos,)
        
        # Update arrays
        waves.append(nod_pos_dict["waves"])
        fluxes.append(nod_pos_dict["fluxes"])
        sigmas.append(nod_pos_dict["sigmas"])
        detectors.append(nod_pos_dict["det_nums"])
        orders.append(nod_pos_dict["order_nums"])

        # Update dataframe
        for col in df_cols:
            transit_df.loc[file_i, col] = nod_pos_dict[col]

    # All done, convert everything to numpy format
    waves = np.stack(waves)
    fluxes = np.stack(fluxes)
    sigmas = np.stack(sigmas)
    detectors = np.vstack(detectors)
    orders = np.vstack(orders)

    # Sort arrays on MJDs
    sorted_i = np.argsort(transit_df["mjd_mid"].values)

    waves = waves[sorted_i]
    fluxes = fluxes[sorted_i]
    sigmas = sigmas[sorted_i]
    detectors = detectors[sorted_i]
    orders = orders[sorted_i]

    # Sort dataframe
    transit_df.sort_values("mjd_mid", inplace=True)

    # Reassign indices so they are in chronological order
    transit_df.reset_index(drop=True, inplace=True)

    # All done! Return!
    return waves, fluxes, sigmas, detectors, orders, transit_df


def save_transit_info_to_fits(
    waves,
    obs_spec,
    sigmas,
    detectors,
    orders,
    transit_info,
    syst_info,
    fits_save_dir,
    label,):
    """Saves prepared wavelength, spectra, sigma, detector, order, transit, and
    planet information in a single multi-extension fits file ready for use for
    modelling with the Aronson method.

    Fits structure:
        HDU 0 (image): 'WAVES' [n_phase, n_spec, n_wave]
        HDU 1 (image): 'OBS_SPEC' [n_phase, n_spec, n_wave]
        HDU 2 (image): 'SIGMAS' [n_phase, n_spec, n_wave]
        HDU 3 (image): 'DETECTORS' [n_spec]
        HDU 4 (image): 'ORDERS' [n_spec]
        HDU 5 (table): 'TRANSIT_INFO' [n_phase]
        HDU 6 (table): 'SYST_INFO'

    Parameters
    ----------
    waves: float array
        Wavelength scales for each timestep of shape [n_phase, n_spec, n_px].
    
    obs_spec: float array
        Observed spectra array for each timestep of shape 
        [n_phase, n_spec, n_px].
    
    sigmas: float array
        Uncertainty array for each timestep of shape [n_phase, n_spec, n_px].
    
    detectors: int array
        Array associating spectral segments to CRIRES+ detector number of shape 
        [n_spec].
    
    orders: int array
        Array associating spectral segments to CRIRES+ order number of shape 
        [n_spec].
    
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

    syst_info: pandas DataFrame
        DataFrame containing planet/star/system properties. The data frame has
        columns ['value', 'sigma', 'reference', 'comment'] and indices:
        ['m_star_msun', 'r_star_rsun', 'k_star_mps', 'dist_pc', 'vsini',
         'rv_star', 'ldc_init_a1', 'ldc_init_a2', 'ldc_init_a3', 'ldc_init_a4',
         'a_planet_au', 'e_planet', 'i_planet_deg', 'omega_planet_deg',
         'w_planet_deg', 'k_star_mps', 'transit_dur_hours', 'jd0_days', 
         'period_planet_days', 'm_planet_mearth', 'r_planet_rearth', 
         'r_planet_atmo_r_earth'].

    fits_save_dir: string
        Directory to save the resulting fits file to.

    label: string
        Label to be included in the filename of format
        transit_data_{}.fits where {} is the label.
    """
    # Intialise HDU List
    hdu = fits.HDUList()

    # Intput checking (?) TODO
    pass

    # HDU 0: wavelength scale
    wave_img =  fits.PrimaryHDU(waves)
    wave_img.header["EXTNAME"] = ("WAVES", "Wavelength scale")
    hdu.append(wave_img)

    # HDU 1: observed spectra
    spec_img =  fits.PrimaryHDU(obs_spec)
    spec_img.header["EXTNAME"] = ("OBS_SPEC", "Observed spectra")
    hdu.append(spec_img)

    # HDU 2: observed spectra uncertainties
    sigmas_img =  fits.PrimaryHDU(sigmas)
    sigmas_img.header["EXTNAME"] = ("SIGMAS", "Observed spectra uncertainties")
    hdu.append(sigmas_img)

    # HDU 3: detectors
    detector_img =  fits.PrimaryHDU(detectors)
    detector_img.header["EXTNAME"] = (
        "DETECTORS", "CRIRES+ detector associated with each spectal segment")
    hdu.append(detector_img)

    # HDU 4: orders
    order_img =  fits.PrimaryHDU(orders)
    order_img.header["EXTNAME"] = (
        "ORDERS", "CRIRES+ grating order associated with each spectal segment")
    hdu.append(order_img)

    # HDU 5: table of observational information for each phase
    transit_tab = fits.BinTableHDU(Table.from_pandas(transit_info))
    transit_tab.header["EXTNAME"] = ("TRANSIT_INFO", "Observation info table")
    hdu.append(transit_tab)
    
    # HDU 6: table of planet system information
    planet_tab = fits.BinTableHDU(Table.from_pandas(syst_info.reset_index()))
    planet_tab.header["EXTNAME"] = ("SYST_INFO", "Table of planet info")
    hdu.append(planet_tab)

    # Done, save
    label = label.replace(" ", "").replace("-", "")
    fits_file = os.path.join(
        fits_save_dir, "transit_data_{}.fits".format(label))
    hdu.writeto(fits_file, overwrite=True)


def load_transit_info_from_fits(fits_load_dir, star_name,):
    """Loads wavelength, spectra, sigma, detector, order, transit, and planet
    information from a single multi-extension fits file ready for use for
    modelling with the Aronson method.
    
    Fits structure:
        HDU 0 (image): 'WAVES' [n_phase, n_spec, n_wave]
        HDU 1 (image): 'OBS_SPEC' [n_phase, n_spec, n_wave]
        HDU 2 (image): 'SIGMAS' [n_phase, n_spec, n_wave]
        HDU 3 (image): 'DETECTORS' [n_spec]
        HDU 4 (image): 'ORDERS' [n_spec]
        HDU 5 (table): 'TRANSIT_INFO' [n_phase]
        HDU 6 (table): 'SYST_INFO'

    Parameters
    ----------
    fits_load_dir: string
        Directory to load the fits file from.

    star_name: string
        Name of the star to be included in the filename of format
        transit_data_{}.fits where {} is the star name.

    Returns
    -------
    waves: float array
        Wavelength scales for each timestep of shape [n_phase, n_spec, n_px].
    
    obs_spec: float array
        Observed spectra array for each timestep of shape 
        [n_phase, n_spec, n_px].
    
    sigmas: float array
        Uncertainty array for each timestep of shape [n_phase, n_spec, n_px].
    
    detectors: int array
        Array associating spectral segments to CRIRES+ detector number of shape 
        [n_spec].
    
    orders: int array
        Array associating spectral segments to CRIRES+ order number of shape 
        [n_spec].
    
    transit_df: pandas DataFrame
        Pandas DataFrame with header/computed information about each timestep.
        Has columns: [mjd, jd, phase, airmass, bcor, hcor, gamma, ra, dec, 
        nod_pos, raw_file] and is of length [n_phase].

    planet_df: pandas DataFrame
        TODO
    """
    # Load in the fits file
    fits_file = os.path.join(
        fits_load_dir, "transit_data_{}.fits".format(star_name))

    with fits.open(fits_file, mode="readonly") as fits_file:
        # Load all image data
        waves = fits_file["WAVES"].data.astype(float)
        obs_spec = fits_file["OBS_SPEC"].data.astype(float)
        sigmas = fits_file["SIGMAS"].data.astype(float)
        detectors = fits_file["DETECTORS"].data.astype(float)
        orders = fits_file["ORDERS"].data.astype(float)

        # Load table data
        transit_info = Table(fits_file["TRANSIT_INFO"].data).to_pandas()
        syst_info = Table(fits_file["SYST_INFO"].data).to_pandas()

        # Set header for syst_info
        syst_info.set_index("parameter", inplace=True)

    # All done, return
    return waves, obs_spec, sigmas, detectors, orders, transit_info, \
        syst_info


def save_transit_model_results_to_fits():
    """
    """
    pass


def calculate_transit_timestep_info(transit_info, syst_info,):
    """Iterate over each time step and compute planet phase, planet XYZ
    position, planet XYZ velocities, and associated mu value at the start, mid-
    point, and end of each exposure. These values are then added to each row in
    our transit_info DataFrame.

    Parameters
    ----------
    transit_info: pandas DataFrame
        Pandas DataFrame with header/computed information about each timestep.
        Has columns:
        ['mjd_start', 'mjd_mid', 'mjd_end', 'jd_start', 'jd_mid', 'jd_end',
         'airmass', 'bcor', 'hcor', 'ra', 'dec', 'exptime_sec', 'nod_pos',
         'raw_file',] and is of length [n_phase].

    syst_info: pandas DataFrame
        DataFrame containing planet/star/system properties. The data frame has
        columns ['value', 'sigma', 'reference', 'comment'] and indices:
        ['m_star_msun', 'r_star_rsun', 'k_star_mps', 'dist_pc', 'vsini',
         'rv_star', 'ldc_init_a1', 'ldc_init_a2', 'ldc_init_a3', 'ldc_init_a4',
         'a_planet_au', 'e_planet', 'i_planet_deg', 'omega_planet_deg',
         'w_planet_deg', 'k_star_mps', 'transit_dur_hours', 'jd0_days', 
         'period_planet_days', 'm_planet_mearth', 'r_planet_rearth', 
         'r_planet_atmo_r_earth'].

    Updates
    -------
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
    """
    # -------------------------------------------------------------------------
    # Error Checking
    # -------------------------------------------------------------------------
    pass

    # -------------------------------------------------------------------------
    # Units and setup
    # -------------------------------------------------------------------------
    # Grab dimensions
    n_phase = len(transit_info)

    # Define constants for convenience
    AU = const.au.cgs.value
    C_LIGHT = const.c.cgs.value
    R_SUN = const.R_sun.cgs.value
    R_EARTH = const.R_earth.cgs.value
    M_SUN = const.M_sun.cgs.value
    M_EARTH = const.M_earth.cgs.value
    
    # Convert degrees to radians
    w_ap_planet_rad = np.deg2rad(syst_info.loc["w_planet_deg", "value"])
    omega_lan_planet_rad = \
        np.deg2rad(syst_info.loc["omega_planet_deg", "value"])
    i_planet_rad = np.deg2rad(90-syst_info.loc["i_planet_deg", "value"])

    # Convert planet mass to solar units
    m_planet_msun = syst_info.loc["m_planet_mearth", "value"] * M_EARTH/M_SUN

    # Calculate transit duration as a fraction of the orbital period
    #transit = syst_info["transit_dur_hours"]/syst_info["transit_dur_hours"]/24

    # Calculate velocities in m/s for the planet
    # TODO: do this properly
    k_planet = (syst_info.loc["k_star_mps", "value"] 
        * syst_info.loc["m_star_msun", "value"]/m_planet_msun)
    #rvs_planet = np.sin(2*np.pi*phases)*k_planet/C_LIGHT*100

    # Calculate the fractional planet size
    r_planet = (syst_info.loc["r_planet_rearth", "value"]*R_EARTH 
        / (syst_info.loc["r_star_rsun", "value"]*R_SUN))

    # We want to compute timestep information for the start, mid, and end of
    # the exposure. This is mostly to allow us to check delta rv_planet across
    # the exposure to see how broadened the planet spectrum will be.
    epochs = ["start", "mid", "end"]

    # -------------------------------------------------------------------------
    # Loop over epochs (i.e. start, mid, and end)
    # -------------------------------------------------------------------------
    for epoch in epochs:
        # Grab the JD values for this epoch
        jds = transit_info["jd_{}".format(epoch)].values

        # Calculate the phase of the planet as phase = (JD - JD_0) / period
        phases = ((jds - syst_info.loc["jd0_days", "value"])
            /syst_info.loc["period_planet_days", "value"])
        
        # Ensure -0.5 < phase < 0.5
        phases -= phases.astype(int)

        phase_gt_half_i = np.squeeze(np.argwhere(phases > 0.5))

        if len(phase_gt_half_i) > 0:
            phases[phase_gt_half_i] = phases[phase_gt_half_i] - 1.0

        # Initialise cartesian coordinate arrays
        r_x = np.zeros(n_phase)
        r_y = np.zeros(n_phase)
        r_z = np.zeros(n_phase)

        v_x = np.zeros(n_phase)
        v_y = np.zeros(n_phase)
        v_z = np.zeros(n_phase)

        s_projected = np.zeros(n_phase)

        # Initialise scale factor proportional to blocking area of the planet
        #   1 when the planet shaddow is fully within the stellar disk
        #   0 when the planet is out of transit
        # Otherwise = fraction of the planet shadow area with the stellar disk.
        planet_area_frac = np.zeros(n_phase)

        # Intialise scaling factor to use when computing an effective mu to
        # adopt during during ingress/egress. During this time we compute an
        # effective s value equal to halfway between the limb of the star and 
        # innermost edge of the planet, and compute scl = s_new / s_old which 
        # we use to scale x and z when computing mu, scl = 1 at all other times
        scl = np.ones(n_phase)

        # ---------------------------------------------------------------------
        # Loop over all phases
        # ---------------------------------------------------------------------
        desc = "Calculating for {} phases for {} exp".format(n_phase, epoch)

        for phase_i in tqdm(range(0, n_phase), desc=desc, leave=False):
            # Compute the mean anomaly of the orbit at this phase. This
            # is the sole parameter that changes from time step to time step.
            ma_planet_rad = 2*np.pi*(phases[phase_i]-0.25) - w_ap_planet_rad

            # Convert orbital elements to cartesian coordinates and velocities
            # This function expects units of Msun, radians, & AU for its params
            r_xyz, v_xyz = compute_orbit_cartesian_pos_and_vel(
                M_star=syst_info.loc["m_star_msun", "value"],
                a=syst_info.loc["a_planet_au", "value"],
                e=syst_info.loc["e_planet", "value"],
                I=i_planet_rad,
                O=omega_lan_planet_rad,
                w=w_ap_planet_rad,
                M=ma_planet_rad,
                M_planet=m_planet_msun,
                error=1E-8)

            # Convert position vector (initially in AU) to R_star units
            r_xyz = r_xyz * AU/(syst_info.loc["r_star_rsun", "value"]*R_SUN)

            # Convert velocity vector (initially in AU/(2*pi*year)) to km/s
            v_xyz = v_xyz *2.*np.pi*AU/(365.25*60*60*24)*1E-5
    
            # Save positional and velocity vectors
            r_x[phase_i] = r_xyz[0]
            r_y[phase_i] = r_xyz[1]
            r_z[phase_i] = r_xyz[2]

            v_x[phase_i] = v_xyz[0]
            v_y[phase_i] = v_xyz[1]
            v_z[phase_i] = v_xyz[2]

            # Calculate planet projected position s = sqrt(x^2 + z^2).
            # We'll use this to work out if the planet is in transit or not.
            s = np.sqrt(r_xyz[0]**2 + r_xyz[2]**2)

            s_projected[phase_i] = s

            # -----------------------------------------------------------------
            # Compute fraction of the planet currently in transit
            # -----------------------------------------------------------------
            # Planet not in transit, no light blocked
            if s > 1.0 + r_planet:
                area_frac = 0

            # Planet in *full* transit, complete blocking
            elif s < 1.0 - r_planet:
                area_frac = 1

            # Planet in ingress or egress, partial blocking
            # Find the area blocked by treating as the intersection of two 
            # circles:
            #   https://mathworld.wolfram.com/Circle-CircleIntersection.html.
            #
            # Circle A is star, radius = R = 1
            #   x^2 + y^2 = R^2 = 1
            #
            # Circle B is planet, radius = r_planet = r
            #   (x-d)^2 + y^2 = r^2
            #
            # Combine equations and solve for x, knowing that R=1 and d=s.
            #   x   = (d^2 - r^2 + R^2) / (2d)
            #       = (s^2 - r^2 + 1) / (2s)
            #
            # Area of one circular segment is given by:
            #   A(R', d')       = R'^2 cos^-1(d'/R') - d' sqrt(R'^2 - d'^2)
            #
            # The area in transit can be found by calculating the area of the 
            # asymmetric circular "lens" composed of two circular segments.
            # Where d_1 = x, d_2 = (d-x) = (s-x), R = 1. Thus:
            #   A_1(R, d_1)   = R^2 cos^-1(d_1/R) - d_1 sqrt(R^2 - d_1^2)
            #                 = cos^1 (x) - x sqrt(1 - x^2)
            #   A_2(r, d_2)   = r^2 cos^-1((s-x)/r) - (s-x) sqrt(r^2 - (s-x)^2)
            #
            #   A             = A_1 + A_2
            #
            # And finally, the *fractional* area
            #   A_frac = A / (2* pi * r^2)
            else:
                # Calculate x
                x = (s**2 - r_planet**2 + 1) / (2*s)
                
                # Calculate the area of both circle sections
                a_1 = np.arccos(x) - x * np.sqrt(1 - x**2)
                a_2 = (r_planet**2 *np.arccos((s-x)/r_planet) 
                    - (s-x) * np.sqrt(r_planet**2 - (s-x)**2))
                
                # Calculate total area
                a_blocked = a_1 + a_2

                # Calculate the fractional area of the planet transiting
                area_frac = a_blocked / (np.pi * r_planet**2)
                
                # Since s & mu refer to the *centre* of the planet, we need to 
                # compute a new centre point for use during ingress/egress.
                # This new point is simply halfway between the innermost edge 
                # of the planet and the limb of the star.
                s_new = (s - r_planet + 1) / 2

                # We assume that x and z scale the same way, so when computing
                # mu we scale x and z based on the fractional change in s.
                scl[phase_i] = s_new / s

            # Store the fractional blocked area
            planet_area_frac[phase_i] = area_frac

            # -----------------------------------------------------------------
            # Update DataFrame
            # -----------------------------------------------------------------
            # Calculate the time steps where the planet is in transit
            is_in_transit = s_projected < (1.0 + r_planet)
            
            # Calculate mu as mu = sqrt(1 - x^2 - y^2). See Mandel & Agol 2001:
            # https://ui.adsabs.harvard.edu/abs/2002ApJ...580L.171M/abstract
            # However, during ingress and egress this becomes
            #   mu = sqrt(1 - (scl * x)^2 - (scl * y)^2)
            # with scl = 1 at all other times.
            with np.errstate(invalid="ignore"):
                mus = np.sqrt((1.0 - (scl*r_x)**2 - (scl*r_z)**2))

            # Reset the non-transiting mu values to zero    TODO is nan better?
            mus[~is_in_transit] = 0

            # Update our DataFrame
            transit_info["phase_{}".format(epoch)] = phases
            transit_info["is_in_transit_{}".format(epoch)] = is_in_transit

            transit_info["r_x_{}".format(epoch)] = r_x
            transit_info["r_y_{}".format(epoch)] = r_y
            transit_info["r_z_{}".format(epoch)] = r_z

            transit_info["v_x_{}".format(epoch)] = v_x
            transit_info["v_y_{}".format(epoch)] = v_y
            transit_info["v_z_{}".format(epoch)] = v_z

            transit_info["s_projected_{}".format(epoch)] = s_projected

            transit_info["scl_{}".format(epoch)] = scl

            transit_info["mu_{}".format(epoch)] = mus
            transit_info["planet_area_frac_{}".format(epoch)] = \
                planet_area_frac

    # -------------------------------------------------------------------------
    # Finally compute our velocities at the midpoint
    # -------------------------------------------------------------------------
    # Calculate our unitless doppler shifts for each phase, assuming that any
    # velocity is positive when in the direction of the star:
    #   γ_j = (v_bary + v_star) / c
    #   β_j = (v_bary + v_star + vsini * x_planet) / c
    #   δ_j = (v_bary + v_star + v^i_planet) / c
    #
    # These doppler shifts are applied to a wavelength vector as:
    #   λ_new = (1 +/- doppler_shift) * λ_old
    #
    # Notes on our velocity conventions:
    # - Planet velocity starts neg (towards us), and becomes pos (away from us)
    # - vsini requires an assumption of whether the planet is orbiting in plane
    #   (same sign convention as planet) or out of plane (anything from no 
    #   no effect to the opposite sign convention). For simplicity, vsini can
    #   be assumed = 0.
    # - Since the Aronson method only cares about *changes* in velocity, the
    #   stellar RV itself can be ignored, in which case the model velocity
    #   frame is the stellar frame, rather than the rest-frame.
    # - The barycentric velocity by default has the *opposite* sign to the
    #   stellar RV (per checking ADR's plumage code which was compared to Gaia 
    #   DR2). That is a positive barcentric velocity implies the Earth is
    #   moving towards the star, and we need to subtract this value to put
    #   things in the barycentric/rest frame.
    # - The doppler shift should be *positive* (1 + doppler_shift) * λ_old when
    #   *correcting* a velocity to the rest-frame.
    # - The doppler shift should be *negative* (1 - doppler_shift) * λ_old when
    #   *shifting* out of the (adopted) rest/reference-frame.
    star_rv_bcor = -1* transit_info["bcor"] + syst_info.loc["rv_star", "value"]
    vsini_planet_epoch = syst_info.loc["rv_star", "value"] * r_x

    gamma = star_rv_bcor / const.c.cgs.to(u.km/u.s)
    beta = (star_rv_bcor + vsini_planet_epoch) / const.c.cgs.to(u.km/u.s)
    delta = (star_rv_bcor + v_y) / const.c.cgs.to(u.km/u.s)

    transit_info["gamma"] = gamma
    transit_info["beta"] = beta
    transit_info["delta"] = delta

    # All done, nothing to return since we've added all computed values to
    # our transit_info DataFrame


def compute_detector_limits(fluxes):
    """Given a set of reduced fluxes, computes the minimum number of edge 
    pixels to exclude with only a single global min/max value for all detectors
    and spectral segments/orders.

    Note: original code had the following limits after this clipping was done.
     - Detector 1 -> 5:2028
     - Detector 2 -> 7:2027
     - Detector 3 -> 10:2026
    TODO: speak to Nik about this

    Parameters
    ----------
    fluxes: 3D float array
        Reduced (but otherwise unscaled) flux array of shape 
        [n_phase, n_spec, n_px].

    Returns 
    -------
    px_min, px_max: int
        Range of useful detector pixels avoiding bad edge pixels, that is 
        fluxes[:,:,px_min:px_max] are the useful pixels.
    """
    # Grab array dimensions for convenience
    n_phase = fluxes.shape[0]
    n_spec = fluxes.shape[1]
    n_wave = fluxes.shape[2]

    # Intialise detector limits: initially we use the whole detector, but these
    # values will creep towards the central value as we find bad pixels.
    px_min = 0
    px_max = n_wave
    
    # Loop over all phases and spectral segments to check for nonfinite edge px
    for phase_i in range(0, n_phase):
        for spec_i in range(0, n_spec):
            # Find and count the nonfinite pixels for this spectral segment
            is_bad_px = ~np.isfinite(fluxes[phase_i, spec_i])
            bad_px_i = np.argwhere(is_bad_px)

            num_nonfinite = np.sum(bad_px_i)

            # If we have any bad pixels, update the min/max if required
            if num_nonfinite > 0:
                # Get subarrays for the indices of the bad pixels in the first
                # and second half. If there are none, the arrays will be empty
                bad_px_start = np.squeeze(np.argwhere([bad_px_i < n_wave/2]))
                bad_px_end = np.squeeze(np.argwhere([bad_px_i > n_wave/2]))

                # Now if we have bad pixels in either half that are inwards of
                # the existing value, update the existing value.
                if len(bad_px_start) > 0 and np.max(bad_px_start) > px_min:
                    px_min = np.max(bad_px_start)

                if len(bad_px_end) > 0 and np.min(bad_px_end) < px_max:
                    px_max = np.min(bad_px_end)

    # All done
    return px_min, px_max


def interpolate_wavelength_scale(
    waves,
    fluxes,
    px_min,
    px_max,):
    """
    Interpolate the spectra from all phases onto a single global wavelength
    scale. The motivation for doing this is that initially each phase is likely
    to have slight offsets in the wavelength scale, so the objective is to put
    all spectral segments/orders across all phases onto a single wavelength 
    grid. When doing this, the minimum and maximum wavelengths adopted for each
    spectral segment are the maximum possible range shared by *alll* phases.

    TODO: interpolate uncertainties too.

    TODO: check interpolation looks sensible.

    Parameters
    ----------
    waves, fluxes: 3D float array
        Array of wavelength scale or fluxes of shape [n_phase, n_spec, n_px].

    px_min, px_max: int
        Range of useful detector pixels avoiding bad edge pixels, that is 
        fluxes[:,:,px_min:px_max] are the useful pixels.

    Returns
    -------
    wave_new: 2D float array
        Global wavelength scale of shape [n_spec, n_px].

    obs: 2D float array
        Regridded flux array of shape [n_phase, n_spec, n_px].
    """
    # Grab array dimensions for convenience
    n_phase = fluxes.shape[0]
    n_spec = fluxes.shape[1]
    n_wave = fluxes.shape[2]

    # Only consider good wavelengths
    wave_adopted = waves[:, :, px_min:px_max]

    # Intialise new wavelength vector common for all phases
    wave_new = np.zeros((n_spec,n_wave))

    # Intialise new flux array with common wavelength scale
    obs = fluxes.copy()

    # Loop over all spectral segments
    desc = "Interpolating wavelengths for {} spectral segments".format(n_spec)

    for spec_i in tqdm(range(0, n_spec), desc=desc, leave=False):
        # Find the common minimum and maximum wavelengths
        wl_min = np.max(wave_adopted[:, spec_i, 0])
        wl_max = np.min(wave_adopted[:,spec_i, -1])

        # Create a new wavelength scale for this spectral segment
        wave_new[spec_i] = np.arange(n_wave)/n_wave * (wl_max-wl_min) + wl_min

        # Now interpolate the fluxes to this new wavelength scale
        for phase_i in range(0, n_phase):
            fluxes_2 = bezier_init(
                x=wave_adopted[phase_i, spec_i],
                y=fluxes[phase_i, spec_i],)

            obs[phase_i, spec_i] = bezier_interp(
                x_a=wave_adopted[phase_i, spec_i],
                y_a=fluxes[phase_i, spec_i],
                y2_a=fluxes_2,
                x_interp=wave_new[spec_i],)

    return wave_new, obs


def sigma_clip_observations(
    obs_spec,
    sigma_lower=5,
    sigma_upper=5,
    max_iterations=5,
    time_steps=None,
    bad_px_replace_val="interpolate"):
    """Sigma clips and cleans the observations as a function of time for each
    spectral pixel. Replaces bad pixels using value specified by 
    bad_px_replace_val, currently nan, median, or interpolate.

    TODO: consider/update uncertainties.

    Parameters
    ----------
    obs_spec: 3D float array
        Observed spectra array for each timestep of shape 
        [n_phase, n_spec, n_px].

    sigma_lower, sigma_upper: float, default: 5
        Lower and upper sigma values to sigma clip with.

    max_iterations: int, default: 5
        Maximum number of sigma clipping iterations to run.

    time_steps: float array or None, default: None
        Array of JD time steps required if interpolating bad pixels.

    bad_px_replace_val: str, default: 'interpolate'
        Value to replace bad pixels with, currently either nan, median, or 
        interpolate.
    
    Returns
    -------
    obs_spec: 3D float array
        Observed spectra array for each timestep of shape 
        [n_phase, n_spec, n_px].

    bad_px_mask: 3D bool array
        Bad pixel mask corresponding to obs_spec of shape 
        [n_phase, n_spec, n_px].
    """
    # Currently supported bad pixel correction methods
    valid_bad_px_replace_vals = [
        "nan",
        "median",
        "interpolate",
    ]

    # Input checking
    if bad_px_replace_val not in valid_bad_px_replace_vals:
        raise ValueError(
            "bad_px_replace_val {} not supported, must be in{}".format(
                bad_px_replace_val, valid_bad_px_replace_vals))

    if bad_px_replace_val == "interpolate":
        if time_steps is None or len(time_steps) != obs_spec.shape[0]:
            raise ValueError(("Time steps array with len(n_phase) must be "
                              "provided when interpolating."))

    # Grab array dimensions for convenience
    n_phase = obs_spec.shape[0]
    n_spec = obs_spec.shape[1]
    n_wave = obs_spec.shape[2]

    # Initialise bad px mask to return
    bad_px_mask = np.full_like(obs_spec, False).astype(bool)

    # Duplicate data array
    obs_spec_clipped = obs_spec.copy()

    # Iterate over spectral pixels, and clip along the phase dimension
    desc = "Cleaning/clipping {} spectral segments".format(n_spec)

    for spec_i in tqdm(range(0, n_spec), desc=desc, leave=False):
        for px_i in range(0, n_wave):
            flux = obs_spec[:, spec_i, px_i]

            # Iteratively sigma clip along the phase dimension
            clipped_fluxes_ma = sigma_clip(
                data=flux,
                sigma_lower=sigma_lower,
                sigma_upper=sigma_upper,
                maxiters=max_iterations,)

            clipped_fluxes = clipped_fluxes_ma.data
            bad_px = clipped_fluxes_ma.mask

            # Now replace bad pixels according to our adopted replacement val
            if bad_px_replace_val == "nan":
                obs_spec_clipped[bad_px, spec_i, px_i] = np.nan
            
            elif bad_px_replace_val == "median":
                obs_spec_clipped[bad_px, spec_i, px_i] = np.nanmedian(flux)

            elif bad_px_replace_val == "interpolate":
                interp_flux = interp1d(x=time_steps, y=clipped_fluxes)

                obs_spec_clipped[bad_px, spec_i, px_i] = \
                    interp_flux(time_steps[bad_px])

            # Update bad px mask for this spectral pixel
            bad_px_mask[:, spec_i, px_i] = bad_px

    return obs_spec_clipped, bad_px_mask