"""Utilities functions associated with transit modelling.

Originally written in IDL by Nikolai Piskunov, ported to Python by Adam Rains.
"""
import os
import glob
import yaml
import pickle
import warnings
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
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from PyAstronomy.pyasl import instrBroadGaussFast

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

    # Restrict the maximum value of k_low to be len(x_a) - 2, such that we can
    # k_high can be the final value of the array.
    k_low = np.clip(k_low, a_min=None, a_max=(len(x_a) - 2))
    #k_low[k_low == len(x_a)] = len(x_a) - 2

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


def regrid_all_phases(
    waves,
    fluxes,
    sigmas,
    detectors,
    nod_positions,
    det_num=[1,2,3],
    reference_nod_pos="A",
    cc_range_kms=(-50,50),
    cc_step_kms=0.5,
    interpolation_method="linear",
    do_sigma_clipping=True,
    sigma_clip_level_upper=3,
    sigma_clip_level_lower=5,
    make_debug_plots=False,
    do_rigid_regrid_per_detector=False,
    n_orders_to_group=1,
    edge_px_to_ignore=40,
    n_ref_div=2,
    use_telluric_model_as_ref=False,
    telluric_wave=None,
    telluric_trans=None,
    continuum_poly_coeff=None,):
    """Function to regrid all phases (from either a single night, or multiple
    nights concatenated together in phase) onto a uniform wavelength scale and
    shift all spectra to the same wavelength frame. This is done by selecting
    either A or B as the reference, and then we use a reference phase for to
    cross-correlate against.

    Parameters
    ----------
    waves, fluxes, sigmas: 3D float array
        Input vectors of shape (n_phase, n_spec, n_px).

    detectors: int array
        Array associating spectral segments to CRIRES+ detector number of shape 
        (n_spec).

    nod_positions: str array
        Array of 'A' and 'B' corresponding to the nod position at that
        particular phase, of shape (n_phase).
    
    det_num: int array, default: [1,2,3]
        Number labels for each detector.

    reference_nod_pos: str, default: 'A'
        Which nod position to use as our reference phase, 'A' or 'B'.

    cc_range_kms: float array, default: (-50, 50)
        Minimum and maximum extent in km/s of cross-correlation.
    
    cc_step_kms: float, default: 0.5
        Step size in km/s for cross-correlation.

    interpolation_method: str, default: "linear"
        Default interpolation method to use with scipy.interp1d.
    
    do_sigma_clipping: boolean, default: True
        Whether to sigma clip arrays before cross-correlation.
    
    sigma_clip_level_upper, sigma_clip_level_lower: float, default: 3, 5
        Thresholds for sigma clipping.

    make_debug_plots: bool, default: False
        Whether to plot diagnostic plots.

    do_rigid_regrid_per_detector: bool, default: False
        If true, we cross-correlate using all orders on a given detector at 
        once.
    
    n_orders_to_group: int, default: 1
        Secondary to do_rigid_regrid_per_detector. How many adjacent orders on
        a single detector to group together when cross-correlating to increase
        the number of telluric lines present. If n_ord is not evenly divisible
        by n_orders_to_group, then at least one order will be involved in
        cross-correlation twice.
    
    edge_px_to_ignore: int, default: 40
        The number of pixels to ignore from the edge of each detector.

    n_ref_div: int, default: 2
        Used to compute the reference phase as n_phase//n_ref_div in
         conjunction with reference_nod_pos.
    
    use_telluric_model_as_ref: boolean, default: False

    telluric_wave, telluric_trans: 1D float array, default: None
        Molecfit telluric wavelength and transmission vectors, of length 
        [n_spec x n_px].

    continuum_poly_coeff: 2D float array, default: None
        Pre-fitted continuum polynomial coefficients of shape [n_spec, n_coeff]

    Returns
    -------
    wave_out: 2D float array
        Adopted wavelength scale of shape (n_spec, n_px)
    
    fluxes_corr, sigmas_corr: 3D float array
        Regridded flux and sigma arrays of shape (n_phase, n_spec, n_px).
    
    rv_shifts: 2D float array
        Array of best-fit RVs of shape (n_phase, n_spec).
    
    cc_rvs: 1D float array
        Array of RVs used for cross-correlation. Constructed as 
        np.arange(cc_range_kms[0], cc_range_kms[1], cc_step_kms).

    cc_values_all: 3D float array
        Array of cross correlation values corresponding to cc_rvs, of shape
        (n_phase, n_spec, n_cc)
    """
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    # Grab dimensions for convenience
    (n_phase, n_spec, n_px) = fluxes.shape

    # Initialise output vectors
    wave_out = np.full((n_spec, n_px), np.nan)
    fluxes_corr = np.full_like(fluxes, np.nan)
    sigmas_corr = np.full_like(sigmas, np.nan)

    # Initialise output grid of rv and doppler shifts
    rv_shifts = np.full((n_phase, n_spec), np.nan)
    doppler_shifts = np.full((n_phase, n_spec), np.nan)

    # Initialise CC rv steps and grid of CC values
    cc_rvs = np.arange(cc_range_kms[0], cc_range_kms[1], cc_step_kms)

    cc_values_all = np.zeros((n_phase, n_spec, len(cc_rvs)))

    # -------------------------------------------------------------------------
    # Loop over all detectors
    # -------------------------------------------------------------------------
    for det_i in det_num:
        # Grab subarrays for all phases for this specific detector. While
        # detectors has shape (n_phase, n_spec), we can safely assume that all
        # phases have the same detector layout so we can just index the first
        # phase.
        det_mask = detectors[0] == det_i

        waves_d = waves[:, det_mask]
        fluxes_d = fluxes[:, det_mask]
        sigmas_d = sigmas[:, det_mask]

        # Count the number of orders on this detector (this may not be the same
        # for all detectors)
        n_ord = fluxes_d.shape[1]

        # ---------------------------------------------------------------------
        # Option A: use observed phase as reference
        # ---------------------------------------------------------------------
        # Our reference phase will be the middle (in time) phase for either A
        # or B frames (depending on the value of reference_nod_pos)
        if not use_telluric_model_as_ref:
            phases = np.arange(n_phase)
            nod_positions_ref = nod_positions == reference_nod_pos
            phases_ref_pos = phases[nod_positions_ref]
            phase_ref_i = phases_ref_pos[len(phases_ref_pos)//n_ref_div]

            txt = "\nRegridding det {}: using phase #{} @ nod pos {} as ref."
            print(txt.format(det_i, phase_ref_i, reference_nod_pos))

            # Store reference wavelength vector
            wave_out[det_mask] = waves_d[phase_ref_i]

            wave_ref = waves_d[phase_ref_i]
            spec_ref = fluxes_d[phase_ref_i]

            # [Optional] Sigma clip our reference spectrum to avoid the impact of
            # bad px on the cross-correlation.
            ref_bad_px_mask = ~np.isfinite(spec_ref)

            if do_sigma_clipping:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sc_mask = sigma_clip(
                        data=spec_ref,
                        sigma_upper=sigma_clip_level_upper,
                        sigma_lower=sigma_clip_level_lower).mask
                    
                    ref_bad_px_mask = np.logical_or(ref_bad_px_mask, sc_mask)

            if edge_px_to_ignore > 0:
                ref_bad_px_mask[:, :edge_px_to_ignore] = True
                ref_bad_px_mask[:, -edge_px_to_ignore:] = True

            # Create an interpolator for the reference spectrum (ignoring bad px)
            interp_ref_flux = interp1d(
                x=wave_ref.ravel()[~ref_bad_px_mask.ravel()],
                y=spec_ref.ravel()[~ref_bad_px_mask.ravel()],
                kind=interpolation_method,
                bounds_error=False,)
        
        # ---------------------------------------------------------------------
        # Option B: use telluric model as reference
        # ---------------------------------------------------------------------
        elif use_telluric_model_as_ref:
            txt = "\nRegridding det {}: using Molecfit model as ref."
            print(txt.format(det_i))
            # Reshape telluric vector
            tw = telluric_wave.reshape(n_spec, n_px)
            tt = telluric_trans.reshape(n_spec, n_px)

            # Ensure telluric vector is ordered
            tw_ii = np.argsort(np.median(tw, axis=1))
            tw = tw[tw_ii]
            tt = tt[tw_ii]

            # Select subset for this particular detector
            wave_ref = tw[det_mask].copy()
            tell_trans = tt[det_mask].copy()
            coeff = continuum_poly_coeff[det_mask]

            # Update output wavelength vector
            wave_out[det_mask] = wave_ref.copy()

            # Apply polynomial to each telluric segment. Note that we assume
            # the polynomial is sorted correctly since we've previously sorted
            # things in wavelength order.
            for spec_i in range(n_spec//3):
                # Calculate continuum poly
                calc_continuum_poly = Polynomial(coeff[spec_i])
                tell_trans[spec_i] *= calc_continuum_poly(wave_ref[spec_i])

            # Create interpolator for the telluric vector
            interp_ref_flux = interp1d(
                x=wave_ref.ravel(),
                y=tell_trans.ravel(),
                kind=interpolation_method,
                bounds_error=False,)

            # Make sure that we enforce a per-segment cross correlation when
            # using a telluric template.
            do_rigid_regrid_per_detector = False
            n_orders_to_group = 1

        # ---------------------------------------------------------------------
        # Prepare groupings
        # ---------------------------------------------------------------------
        # Here we organise how we are (or aren't) grouping adjacent orders on a
        # single detector together for cross-correlation to increase the number
        # of telluric lines available to cross-correlate against and prevent
        # inconsistencies in RV correction between orders with fewer tellurics.
        # Grouping the orders together is easy if n_ord is divisible by
        # n_orders_to_group in which case each order only gets considered once.
        # If not, at least one order will be considered multiple times for the
        # very last  of orders.

        # If we're doing a completely rigid cross-correlation, then we simply
        # use all orders at once which means only one cross-correlation.
        if do_rigid_regrid_per_detector:
            n_set = 1
            uneven_division = False

        # If not, check to see if the number of order groups fits neatly into
        # the number of orders.
        elif n_ord % n_orders_to_group == 0:
            n_set = n_ord // n_orders_to_group
            uneven_division = False

        # Otherwise we need to give special consideration to the final group
        else:
            n_set = n_set = n_ord // n_orders_to_group + 1
            uneven_division = True

        # ---------------------------------------------------------------------
        # Cross-correlate for every set of orders on this detector
        # ---------------------------------------------------------------------
        for set_i in range(n_set):
            # Determine the lower and upper orders to be considered for each
            # grouping. In the case where we have an unequal division of sets
            # into the total number of orders, we need to give the final set
            # special consideration.
            if do_rigid_regrid_per_detector:
                ord_low = 0
                ord_high = n_ord

            elif set_i == n_set-1 and uneven_division:
                ord_low = n_ord - n_orders_to_group
                ord_high = n_set

            else:
                ord_low = set_i * n_orders_to_group
                ord_high = set_i * n_orders_to_group + n_orders_to_group

            # Determine the spectral segment index(ices) for ease of indexing
            spec_ii = np.arange(n_spec)
            spec_i = spec_ii[det_mask][ord_low:ord_high]

            print(
                "\tDet {:0.0f}".format(det_i),
                "Set {:0.0f}".format(set_i+1),
                np.nanmedian(waves_d[0, ord_low:ord_high], axis=1).astype(int))

            # -----------------------------------------------------------------
            # Cross-correlate for all phases
            # -----------------------------------------------------------------
            desc = "CCing det {:0.0f}, order set {:0.0f}/{:0.0f}".format(
                det_i, set_i+1, n_set)

            for phase_i in tqdm(range(n_phase), desc=desc, leave=False):
                # Grab vectors for this phase, of shape (n_set, n_px)
                ww = waves_d[phase_i, ord_low:ord_high]
                ff = fluxes_d[phase_i, ord_low:ord_high]
                ss = sigmas_d[phase_i, ord_low:ord_high]

                # [Optional] Sigma clip the upper bounds to remove cosmics that
                # might interfere with the cross-correlation. Note that we will
                # then RV shift the *non-clipped* fluxes for output, rather
                # than these clipped fluxes
                bpm = ~np.isfinite(ff)
                
                if do_sigma_clipping:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        bpm_sc = sigma_clip(
                            data=ff,
                            sigma_upper=sigma_clip_level_upper,
                            sigma_lower=sigma_clip_level_lower).mask
                
                    bpm = np.logical_or(bpm, bpm_sc)
                
                if edge_px_to_ignore > 0:
                    bpm[:, :edge_px_to_ignore] = True
                    bpm[:, -edge_px_to_ignore:] = True

                # Initialise cross correlation array for this phase
                cc_values = np.zeros_like(cc_rvs)

                # Run cross correlation
                for rv_i, rv in enumerate(cc_rvs):
                    # Interpolate reference spectrum to rv
                    flux_ref = interp_ref_flux(
                        wave_ref[ord_low:ord_high]
                        * (1-rv/(const.c.si.value/1000)))

                    # Mask out edges to ensure we're not interpolating across
                    # the gaps between adjacent orders.
                    if edge_px_to_ignore > 0:
                        flux_ref[:, :edge_px_to_ignore] = np.nan
                        flux_ref[:, -edge_px_to_ignore:] = np.nan

                    # Compute normalised cross correlation
                    cc_values[rv_i] = (np.nansum(ff[~bpm] * flux_ref[~bpm]) 
                        / np.sqrt(np.nansum(ff[~bpm]**2)
                                * np.nansum(flux_ref[~bpm]**2)))

                # Determine rv that gives cross correlation maximum
                rv_opt =  cc_rvs[np.argmax(cc_values)]
                rv_shifts[phase_i, spec_i] = rv_opt

                # Compute equivalent doppler shift and store. Note that this
                # has the opposite sign since we're now shifting the science
                # flux and not the reference flux.
                ds_opt = 1 + rv_opt/(const.c.si.value/1000)
                doppler_shifts[phase_i, spec_i] = ds_opt

                # Store the cross correlation values
                cc_values_all[phase_i, spec_i] = cc_values

                # -------------------------------------------------------------
                # Regrid
                # -------------------------------------------------------------
                # Construct interpolators for flux and sigma and interpolate
                bad_px_sci = np.logical_or(
                    ~np.isfinite(ff), ~np.isfinite(ss))

                interp_sci_flux = interp1d(
                    x=ww[~bad_px_sci].ravel(),
                    y=ff[~bad_px_sci].ravel(),
                    kind=interpolation_method,
                    bounds_error=False,)
                
                interp_sci_sigma = interp1d(
                    x=ww[~bad_px_sci].ravel(),
                    y=ss[~bad_px_sci].ravel(),
                    kind=interpolation_method,
                    bounds_error=False,)

                # Interpolate flux and sigmas using this doppler shift
                fluxes_corr[phase_i, spec_i] = interp_sci_flux(ww * ds_opt)
                sigmas_corr[phase_i, spec_i] = \
                    interp_sci_sigma(ww * ds_opt)

    # -------------------------------------------------------------------------
    # [Optional] Visualisation for debugging
    # -------------------------------------------------------------------------
    if make_debug_plots:
        # Plot of cross-correlation functions
        plt.close("all")
        fig_cc, axis_cc = plt.subplots()

        for pi in range(n_phase):
            if nod_positions[pi] == "A":
                colour = "r"
            else:
                colour = "b"
            
            for spec_i in range(n_spec):
                # Plot the data
                axis_cc.plot(
                    cc_rvs,
                    cc_values_all[pi,spec_i],
                    linewidth=0.5,
                    color=colour)

    return wave_out, fluxes_corr, sigmas_corr, rv_shifts, cc_rvs, cc_values_all


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

    # Compute steps in wavelength and padding in pixels, clip to be > 0
    dx1 = x[1] - x[0]
    pad1 = int(np.max([np.ceil(-gamma*x[0]/dx1), 0]))

    dx2 = x[-1] - x[-2]
    pad2 = int(np.max([np.ceil(gamma*x[-1]/dx2), 0]))

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
        yy = np.concatenate((y, y_pad))
        
        yy2_pad = -y2[-np.arange(pad2)-2]
        yy2 = np.concatenate((y2, yy2_pad))

    # Mirror the *start* of the array
    elif pad1 > 0 and pad2 == 0:
        x_pad = x[0] - (pad1-np.arange(pad1))*dx1
        xx = np.concatenate((x_pad, x))

        y_pad = y[pad1-np.arange(pad1)]
        yy = np.concatenate((y_pad, y))

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
        index_col="parameter",)
        #dtype={"value":str, "sigma":str})

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

        # Weather info
        obs_temp = fits_file[0].header["HIERARCH ESO TEL AMBI TEMP"]
        rel_humidity = fits_file[0].header["HIERARCH ESO TEL AMBI RHUM"]
        wind_dir = fits_file[0].header["HIERARCH ESO TEL AMBI WINDDIR"]
        wind_speed = fits_file[0].header["HIERARCH ESO TEL AMBI WINDSP"]
        seeing_start = fits_file[0].header["HIERARCH ESO TEL AMBI FWHM START"]
        seeing_end = fits_file[0].header["HIERARCH ESO TEL AMBI FWHM END"]


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
        "obs_temp":obs_temp,
        "rel_humidity":rel_humidity,
        "wind_dir":wind_dir,
        "wind_speed":wind_speed,
        "seeing_start":seeing_start,
        "seeing_end":seeing_end,
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
        "nod_pos", "raw_file", "obs_temp", "rel_humidity", "wind_dir",
        "wind_speed", "seeing_start", "seeing_end",]

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


# -----------------------------------------------------------------------------
# Save/load planet transit info
# -----------------------------------------------------------------------------
def save_transit_info_to_fits(
    waves,
    obs_spec_list,
    sigmas_list,
    n_transits,
    detectors,
    orders,
    transit_info_list,
    syst_info,
    fits_save_dir,
    label,
    cc_rv_shifts_list=None,
    sim_info=None,):
    """Saves prepared wavelength, spectra, sigma, detector, order, transit, and
    planet information in a single multi-extension fits file ready for use for
    modelling with the Aronson method.

    The filename will have format transit_data_X_nY.fits where X is the star
    name/label, and Y is the number of transits saved in the file.

    HDUs common to all transits:
        HDU 0 (image): 'WAVES' [n_phase, n_spec, n_wave]
        HDU 1 (image): 'DETECTORS' [n_spec]
        HDU 2 (image): 'ORDERS' [n_spec]
        HDU 3 (table): 'SYST_INFO'

    Plus a set of the following for each transit, where X is the transit count:
        HDU 4 (image): 'OBS_SPEC_X' [n_phase, n_spec, n_wave]
        HDU 5 (image): 'SIGMAS_X' [n_phase, n_spec, n_wave]
        HDU 6 (table): 'TRANSIT_INFO_X' [n_phase]

    Parameters
    ----------
    waves: float array
        Wavelength scales for each timestep of shape [n_phase, n_spec, n_px].
    
    obs_spec_list: list of 3D float arrays
        List of observed spectra (length n_transits), with one 3D float array
        transit of shape  [n_phase, n_spec, n_px].
    
    sigmas: list of 3D float arrays
        List of observed sigmas (length n_transits), with one 3D float array
        transit of shape  [n_phase, n_spec, n_px].
    
    n_transits: int
        The number of transits to be saved.

    detectors: int array
        Array associating spectral segments to CRIRES+ detector number of shape 
        [n_spec].
    
    orders: int array
        Array associating spectral segments to CRIRES+ order number of shape 
        [n_spec].
    
    transit_info_list: list of pandas DataFrame
        List of transit info (length n_transits) DataFrames containing 
        information associated with each transit time step. Each DataFrame has
        columns:

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

    cc_rv_shifts_list: list of 2D float arrays
        List (length n_transits) of RV offsets determined by cross correlation
        for each spectral segment, with each 2D float array having shape 
        [n_phase_i, n_spec] with n_phase_i being the number of phases observed
        for transit_i.

    sim_info: pandas DataFrame, default: None
        Pandas DataFrame containing the simulation settings if we're saving the
        output of a simulation. Not saved if None.
    """
    # Intialise HDU List
    hdu = fits.HDUList()

    # Intput checking (?) TODO
    pass

    # HDU 0: wavelength scale
    wave_img =  fits.PrimaryHDU(waves)
    wave_img.header["EXTNAME"] = ("WAVES", "Wavelength scale")
    hdu.append(wave_img)

    # HDU 1: detectors
    detector_img =  fits.PrimaryHDU(detectors.astype("int32"))
    detector_img.header["EXTNAME"] = (
        "DETECTORS", "CRIRES+ detector associated with each spectal segment")
    hdu.append(detector_img)

    # HDU 2: orders
    order_img =  fits.PrimaryHDU(orders.astype("int32"))
    order_img.header["EXTNAME"] = (
        "ORDERS", "CRIRES+ grating order associated with each spectal segment")
    hdu.append(order_img)

    # HDU 3: table of planet system information
    planet_tab = fits.BinTableHDU(Table.from_pandas(syst_info.reset_index()))
    planet_tab.header["EXTNAME"] = ("SYST_INFO", "Table of planet info")
    hdu.append(planet_tab)

    # Loop over all transits
    for transit_i in range(n_transits):
        # HDU 4 + transit_i*4: observed spectra
        spec_img =  fits.PrimaryHDU(obs_spec_list[transit_i])
        spec_img.header["EXTNAME"] = (
            "OBS_SPEC_{}".format(transit_i), "Observed spectra")
        hdu.append(spec_img)

        # HDU 5 + transit_i*4: observed spectra uncertainties
        sigmas_img =  fits.PrimaryHDU(sigmas_list[transit_i])
        sigmas_img.header["EXTNAME"] = (
            "SIGMAS_{}".format(transit_i), "Observed spectra uncertainties")
        hdu.append(sigmas_img)

        # HDU 6 + transit_i*4: table of information for each phase
        transit_tab = fits.BinTableHDU(
            Table.from_pandas(transit_info_list[transit_i]))
        transit_tab.header["EXTNAME"] = (
            "TRANSIT_INFO_{}".format(transit_i), "Observation info table")
        hdu.append(transit_tab)

        # [Optional] If we've regridded the data, we can also save the RV
        # shifts between each detector.
        if cc_rv_shifts_list is not None:
            # HDU 7 + transit_i*4:
            rv_img =  fits.PrimaryHDU(cc_rv_shifts_list[transit_i])
            rv_img.header["EXTNAME"] = (
                "CC_RV_{}".format(transit_i), "RV offsets per order from CC.")
            hdu.append(rv_img)

    # [optional] HDU N: table of simulation settings
    if sim_info is not None:
        sim_tab = fits.BinTableHDU(Table.from_pandas(sim_info.reset_index()))
        sim_tab.header["EXTNAME"] = ("SIM_INFO", "Simulation settings table")
        hdu.append(sim_tab)

    # Done, save
    fits_file = os.path.join(
        fits_save_dir, "transit_data_{}_n{}.fits".format(label, n_transits))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hdu.writeto(fits_file, overwrite=True)


def load_transit_info_from_fits(fits_load_dir, label, n_transit,):
    """Loads wavelength, spectra, sigma, detector, order, transit, and planet
    information from a single multi-extension fits file ready for use for
    modelling with the Aronson method.

    The filename should have format transit_data_X_nY.fits where X is the star
    name/label, and Y is the number of transits saved in the file.
    
    HDUs common to all transits:
        HDU 0 (image): 'WAVES' [n_phase, n_spec, n_wave]
        HDU 1 (image): 'DETECTORS' [n_spec]
        HDU 2 (image): 'ORDERS' [n_spec]
        HDU 3 (table): 'SYST_INFO'

    Plus a set of the following for each transit, where X is the transit count:
        HDU 4 (image): 'OBS_SPEC_X' [n_phase, n_spec, n_wave]
        HDU 5 (image): 'SIGMAS_X' [n_phase, n_spec, n_wave]
        HDU 6 (table): 'TRANSIT_INFO_X' [n_phase]

    Parameters
    ----------
    fits_load_dir: string
        Directory to load the fits file from.

    label: string
        Label to be included in the filename.

    n_transit: int
        Number of transits saved to this fits file.

    Returns
    -------
    waves: float array
        Wavelength scales for each timestep of shape [n_phase, n_spec, n_px].
    
    obs_spec_list: list of 3D float arrays
        List of observed spectra (length n_transits), with one 3D float array
        transit of shape  [n_phase, n_spec, n_px].
    
    sigmas_list: list of 3D float arrays
        List of observed sigmas (length n_transits), with one 3D float array
        transit of shape  [n_phase, n_spec, n_px].
    
    detectors: int array
        Array associating spectral segments to CRIRES+ detector number of shape 
        [n_spec].
    
    orders: int array
        Array associating spectral segments to CRIRES+ order number of shape 
        [n_spec].
    
    transit_info_list: list of pandas DataFrames
        List of transit info (length n_transits) DataFrames containing 
        information associated with each transit time step. Each DataFrame has
        columns:

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
    """
    # Load in the fits file
    fits_file = os.path.join(
        fits_load_dir, "transit_data_{}_n{}.fits".format(label, n_transit))

    with fits.open(fits_file, mode="readonly") as fits_file:
        # Load data constant across transits
        waves = fits_file["WAVES"].data.astype(float)
        detectors = fits_file["DETECTORS"].data.astype("int32")
        orders = fits_file["ORDERS"].data.astype("int32")
        syst_info = Table(fits_file["SYST_INFO"].data).to_pandas()

        # Set header for syst_info
        syst_info.set_index("parameter", inplace=True)

        # Load per-transit HDUs
        obs_spec_list = []
        sigmas_list = []
        transit_info_list = []
        
        for transit_i in range(n_transit):
            spec_hdu = "OBS_SPEC_{}".format(transit_i)
            sigma_hdu = "SIGMAS_{}".format(transit_i)
            transit_tab_hdu = "TRANSIT_INFO_{}".format(transit_i)

            obs_spec = fits_file[spec_hdu].data.astype(float)
            sigmas = fits_file[sigma_hdu].data.astype(float)
            transit_info = Table(fits_file[transit_tab_hdu].data).to_pandas()

            obs_spec_list.append(obs_spec)
            sigmas_list.append(sigmas)
            transit_info_list.append(transit_info)

    # All done, return
    return waves, obs_spec_list, sigmas_list, detectors, orders, \
        transit_info_list, syst_info


# -----------------------------------------------------------------------------
# Save/load transit components (flux, trans, tau, scale)
# -----------------------------------------------------------------------------
def save_simulated_transit_components_to_fits(
    fits_load_dir,
    label,
    n_transit,
    flux,
    tau,
    trans,
    scale,):
    """Function to save the component flux, telluric tau, planet transmission,
    and scale vectors to the same fits file as the individual epochs.

    We expect to receive only a single flux and trans vector, but as many tau
    and scale vectors as there are transits.
    
    Parameters
    ----------
    fits_load_dir: string
        Directory to load the fits file from.

    label: string
        Label to be included in the filename.

    n_transit: int
        Number of transits saved to this fits file.

    flux: 2D float array
        Model stellar flux component of shape [n_spec, n_px].

    tau: 3D float array
        Model telluric tau component of shape [n_transit, n_spec, n_px].

    trans: 2D float array
        Model planet transmission component of shape [n_spec, n_px].

    scale: list of 1D float arrays
        List of shape n_transit of adopted scale/slit losses of shape [n_phase]
    """
    # Pair the extensions with their data
    extensions = {
        "COMPONENT_FLUX":(flux, "Flux component of simulated transit."),
        "COMPONENT_TAU":(tau, "Telluric tau component of simulated transit."),
        "COMPONENT_TRANS":(trans, "Planet component of simulated transit."),
        "COMPONENT_SCALE":(scale, "Adopted scale/slit loss vector."),
    }

    # Load in the fits file
    fits_file = os.path.join(
        fits_load_dir, "transit_data_{}_n{}.fits".format(label, n_transit))

    with fits.open(fits_file, mode="update") as fits_file:
        # Constant components across transits
        for extname in ["COMPONENT_FLUX", "COMPONENT_TRANS"]:
            # First check if the HDU already exists
            if extname in fits_file:
                fits_file[extname].data = extensions[extname][0]
            
            # Not there, make and append
            else:
                hdu = fits.PrimaryHDU(extensions[extname][0])
                hdu.header["EXTNAME"] = (extname, extensions[extname][1])
                fits_file.append(hdu)

        # Components that vary from transit to transit
        for extname in ["COMPONENT_TAU", "COMPONENT_SCALE"]:
            for trans_i in range(n_transit):
                extname_i = "{}_{:0.0f}".format(extname, trans_i)

                # First check if the HDU already exists
                if extname_i in fits_file:
                    fits_file[extname_i].data = extensions[extname][0][trans_i]
                
                # Not there, make and append
                else:
                    hdu = fits.PrimaryHDU(extensions[extname][0][trans_i])
                    hdu.header["EXTNAME"] = (extname_i, extensions[extname][1])
                    fits_file.append(hdu)

        fits_file.flush()


def load_simulated_transit_components_from_fits(
    fits_load_dir,
    label,
    n_transit,):
    """Function to load the component flux, telluric tau, and planet 
    transmission vectors from a fits file.
    
    Parameters
    ----------
    fits_load_dir: string
        Directory to load the fits file from.

    label: string
        Label to be included in the filename.

    n_transit: int
        Number of transits saved to this fits file.

    Returns
    ----------
    component_flux: 2D float array
        Model stellar flux component of shape [n_spec, n_px].

    component_tau: 3D float array
        Model telluric tau component of shape [n_transit, n_spec, n_px].

    component_trans: 2D float array
        Model planet transmission component of shape [n_spec, n_px].

    component_scale: list of 1D float arrays
        List of shape n_transit of adopted scale/slit losses of shape [n_phase]
    """
    # Load in the fits file
    fits_file = os.path.join(
        fits_load_dir, "transit_data_{}_n{}.fits".format(label, n_transit))

    with fits.open(fits_file, mode="readonly") as fits_file:
        # Load data constant across transits
        component_flux = fits_file["COMPONENT_FLUX"].data.astype(float)
        component_trans = fits_file["COMPONENT_TRANS"].data.astype(float)

        # Load components that vary across transits
        component_tau = []
        component_scale = []

        for transit_i in range(n_transit):
            tau_nm = "COMPONENT_TAU_{:0.0f}".format(transit_i)
            tau = fits_file[tau_nm].data.astype(float)
            component_tau.append(tau)

            scale_nm = "COMPONENT_SCALE_{:0.0f}".format(transit_i)
            scale = fits_file[scale_nm].data.astype(float)
            component_scale.append(scale)

    component_tau = np.array(component_tau)

    # All done, return
    return component_flux, component_tau, component_trans, component_scale


# -----------------------------------------------------------------------------
# Save/load inverse model results (flux, trans, tau, scale)
# -----------------------------------------------------------------------------
def save_transit_model_results_to_fits(
    fits_load_dir,
    label,
    n_transit,
    model,
    flux,
    trans,
    tau,
    scale,
    mask,):
    """Function to save the output of a modelling run to the same fits file
    used to initialise it.

    Parameters
    ----------
    fits_load_dir: string
        Directory to load the fits file from.

    label: string
        Label to be included in the filename.

    n_transit: int
        Number of transits saved to this fits file.

    model: 3D float array
        Fitted model observation (star + tellurics + planet) matrix of shape
        [n_phase, n_spec, n_px].

    flux: 2D float array
        Fitted model stellar flux of shape [n_spec, n_px].

    trans: 2D float array
        Fitted model planet transmission of shape [n_spec, n_px].

    tau: 2D float array
        Fitted model telluric tau of shape [n_spec, n_px].

    scale: 1D float array
        Fitted model scale parameter of shape [n_phase].

    mask: 3D float array
        Mask array of shape [n_phase, n_spec, n_px]. Contains either 0 or 1.
    """
    # Pair the extensions with their data
    extensions = {
        "MODEL_OBS":(model, "Model observation combining each component."),
        "MODEL_FLUX":(flux, "Fitted Aronson stellar fluxes."),
        "MODEL_TRANS":(trans, "Fitted Aronson planet transmission."),
        "MODEL_TAU":(tau, "Fitted Aronson telluric tau."),
        "MODEL_SCALE":(scale, "Fitted Aronson model scale."),
        "MODEL_MASK":(mask, "Fitted Aronson mask."),
    }

    # Load in the fits file
    fits_file = os.path.join(
        fits_load_dir, "transit_data_{}_n{}.fits".format(label, n_transit))

    with fits.open(fits_file, mode="update") as fits_file:
        for extname in extensions.keys():
            # First check if the HDU already exists
            if extname in fits_file:
                fits_file[extname].data = extensions[extname][0]
            
            # Not there, make and append
            else:
                hdu = fits.PrimaryHDU(extensions[extname][0])
                hdu.header["EXTNAME"] = (extname, extensions[extname][1])
                fits_file.append(hdu)

            fits_file.flush()


def load_simulated_model_results_from_fits(
    fits_load_dir,
    label,
    n_transit,):
    """Function to load the output of a modelling run (model 'observation', 
    stellar flux, telluric tau, planet transmission, scale, and mask vectors)
    from a fits file.
    
    Parameters
    ----------
    fits_load_dir: string
        Directory to load the fits file from.

    label: string
        Label to be included in the filename.

    n_transit: int
        Number of transits saved to this fits file.

    Returns
    -------
    model_obs: 2D float array
        Fitted model 'observation' vector of shape [n_phase, n_spec, n_px].

    model_flux: 2D float array
        Fitted model stellar flux vector of shape [n_spec, n_px].

    model_tau: 3D float array
        Fitted model telluric tau vector of shape [n_transit, n_spec, n_px].

    model_trans: 2D float array
        Fitted model planet transmission vector of shape [n_spec, n_px].

    model_scale: 1D float array
        Fitted scale vector of scale/slit losses of shape [n_phase].

    model_mask: 1D float array
        Adopted fit mask of shape [n_phase, n_spec, n_px]
    """
    # Load in the fits file
    fits_file = os.path.join(
        fits_load_dir, "transit_data_{}_n{}.fits".format(label, n_transit))

    with fits.open(fits_file, mode="readonly") as fits_file:
        # Load data constant across transits
        model_obs = fits_file["MODEL_OBS"].data.astype(float)
        model_flux = fits_file["MODEL_FLUX"].data.astype(float)
        model_tau = fits_file["MODEL_TAU"].data.astype(float)
        model_trans = fits_file["MODEL_TRANS"].data.astype(float)
        model_scale = fits_file["MODEL_SCALE"].data.astype(float)
        model_mask = fits_file["MODEL_MASK"].data.astype(bool)

    # All done, return
    return \
        model_obs, model_flux, model_tau, model_trans, model_scale, model_mask

# -----------------------------------------------------------------------------
# Save/load normalised spectra
# -----------------------------------------------------------------------------
def save_normalised_spectra_to_fits(
    fits_load_dir,
    label,
    n_transit,
    fluxes_norm,
    sigmas_norm,
    bad_px_mask_norm,
    poly_coeff,
    transit_i,):
    """Function to save normalised flux, sigma, bad px, and fitted polynomial
    coefficient arrays as a set of fits HDUs.

    Parameters
    ----------
    fits_load_dir: string
        Directory to load the fits file from.

    label: string
        Label to be included in the filename.

    n_transit: int
        Number of transits saved to this fits file.

    fluxes_norm, sigmas_norm: 3D float array
        Normalised flux and sigma vectors of shape [n_phase, n_spec, n_px].

    bad_px_mask_norm: 3D bool array
        Bad pixel mask corresponding to flux and sigma vectors, of shape
        [n_phase, n_spec, n_px].

    poly_coeff: 3D float array
        Fitted polynomial continuum coefficients of shape 
        [n_phase, n_spec, n_coeff]

    transit_i: int
        The transit night number.
    """
    # Input checking
    ti = int(transit_i)

    # Pair the extensions with their data
    extensions = {
        "FLUXES_NORM_{}".format(ti):(
            fluxes_norm, "Continuum normalised fluxes."),
        "SIGMAS_NORM_{}".format(ti):(
            sigmas_norm, "Continuum normalised sigmas."),
        "BAD_PX_NORM_{}".format(ti):(
            bad_px_mask_norm.astype(int), "Bad px mask for norm. fluxes."),
        "CONTINUUM_POLY_COEFF_{}".format(ti):(
            poly_coeff, "Fitted continuum normalisation polynomial coeffs."),
    }

    # Load in the fits file
    fits_file = os.path.join(
        fits_load_dir, "transit_data_{}_n{}.fits".format(label, n_transit))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(fits_file, mode="update") as fits_file:
            for extname in extensions.keys():
                # First check if the HDU already exists
                if extname in fits_file:
                    fits_file[extname].data = extensions[extname][0]
                
                # Not there, make and append
                else:
                    hdu = fits.PrimaryHDU(extensions[extname][0])
                    hdu.header["EXTNAME"] = \
                        (extname, extensions[extname][1])
                    fits_file.append(hdu)

                fits_file.flush()


def load_normalised_spectra_from_fits(
    fits_load_dir,
    label,
    n_transit,
    transit_i,):
    """Function to load normalised wave, flux, sigma, and bad px arrays from a
    set of fits HDUs.

    Parameters
    ----------
    fits_load_dir: string
        Directory to load the fits file from.

    label: string
        Label to be included in the filename.

    n_transit: int
        Number of transits saved to this fits file.

    transit_i: int
        The transit night number.
        
    Returns
    -------
    waves_norm: 2D float array
        Wavelength vector of shape [n_spec, n_px].

    fluxes_norm, sigmas_norm: 3D float array
        Normalised flux and sigma vectors of shape [n_phase, n_spec, n_px].

    bad_px_mask_norm: 3D bool array
        Bad pixel mask corresponding to flux and sigma vectors, of shape
        [n_phase, n_spec, n_px].
    
    poly_coeff: 3D float array
        Fitted polynomial continuum coefficients of shape 
        [n_phase, n_spec, n_coeff]
    """
    # Input checking
    ti = int(transit_i)

    # Load in the fits file
    fits_file = os.path.join(
        fits_load_dir, "transit_data_{}_n{}.fits".format(label, n_transit))

    with fits.open(fits_file, mode="readonly") as fits_file:
        # Load data constant across transits
        fluxes_norm = fits_file["FLUXES_NORM_{}".format(ti)].data.astype(float)
        sigmas_norm = fits_file["SIGMAS_NORM_{}".format(ti)].data.astype(float)
        bad_px_mask_norm = \
            fits_file["BAD_PX_NORM_{}".format(ti)].data.astype(bool)
        poly_coeff = \
            fits_file["CONTINUUM_POLY_COEFF_{}".format(ti)].data.astype(float)

    # All done, return
    return fluxes_norm, sigmas_norm, bad_px_mask_norm, poly_coeff


# -----------------------------------------------------------------------------
# Save/load SYSREM results
# -----------------------------------------------------------------------------
def save_sysrem_residuals_to_fits(
    fits_load_dir,
    label,
    n_transit,
    sysrem_resid,
    transit_i,
    sequence,
    rv_frame,):
    """Function to save a datacube of sysrem residuals to a fits HDU. The
    residuals will have shape (n_sysrem_iter, n_phase, n_spec, n_px), with the
    transit/night number in the extension name.

    Parameters
    ----------
    fits_load_dir: string
        Directory to load the fits file from.

    label: string
        Label to be included in the filename.

    n_transit: int
        Number of transits saved to this fits file.

    sysrem_resid: 4D or 5D float array
       Datacube of SYSREM residuals of shape 
       (n_sysrem_iter, n_phase, n_spec, n_px) or 
       (n_transit, n_sysrem_iter, n_phase, n_spec, n_px).

    transit_i: int
        The transit night number.

    sequence: str
        Used to note whether the saved residuals are from separate A ('A') or B
        ('B') nodding sequences, or are interleaved ('AB').

    rv_frame: str
        RV frame that SYSREM was run in, either 'telluric' or 'stellar'.
    """
    # HDU info
    ext_name = "SYSREM_RESID_{:0.0f}_{}_{}".format(
        transit_i, sequence, rv_frame)
    ext_desc = "Residuals after running SYSREM."

    # Load in the fits file
    fits_file = os.path.join(
        fits_load_dir, "transit_data_{}_n{}.fits".format(label, n_transit))

    with fits.open(fits_file, mode="update") as fits_file:
        # First check if the HDU already exists
        if ext_name in fits_file:
            fits_file[ext_name].data = sysrem_resid
        
        # Not there, make and append
        else:
            hdu = fits.PrimaryHDU(sysrem_resid)
            hdu.header["EXTNAME"] = (ext_name, ext_desc)
            fits_file.append(hdu)

        fits_file.flush()


def load_sysrem_residuals_from_fits(
    fits_load_dir,
    label,
    n_transit,
    transit_i,
    sequence,
    rv_frame,):
    """Function to load a datacube of sysrem residuals from a fits HDU. The
    residuals will have shape (n_sysrem_iter, n_phase, n_spec, n_px), with the
    transit/night number in the extension name.

    Parameters
    ----------
    fits_load_dir: string
        Directory to load the fits file from.

    label: string
        Label to be included in the filename.

    n_transit: int
        Number of transits saved to this fits file.

    transit_i: int
        The transit night number.

    sequence: str
        Used to note whether the saved residuals are from separate A ('A') or B
        ('B') nodding sequences, or are interleaved ('AB').

    rv_frame: str
        RV frame that SYSREM was run in, either 'telluric' or 'stellar'.
        
    Returns
    -------
    sysrem_resid: 4D or 5D float array
       Datacube of SYSREM residuals of shape 
       (n_sysrem_iter, n_phase, n_spec, n_px) or 
       (n_transit, n_sysrem_iter, n_phase, n_spec, n_px).
    """
    # Construct fits path
    fits_file = os.path.join(
        fits_load_dir, "transit_data_{}_n{}.fits".format(label, n_transit))

    ext_name = "SYSREM_RESID_{:0.0f}_{}_{}".format(
        transit_i, sequence, rv_frame)

    # Open the fits file and grab the data
    with fits.open(fits_file, mode="readonly") as fits_file:
        sysrem_resid = fits_file[ext_name].data.astype(float)

    return sysrem_resid


# -----------------------------------------------------------------------------
# Dump/load cross-correlation and Kp-Vsys maps to disk
# -----------------------------------------------------------------------------
def dump_cc_results(
    filename,
    cc_rvs,
    ccv_ps,
    ccv_comb,
    vsys_steps,
    Kp_steps,
    Kp_vsys_map_ps,
    Kp_vsys_map_comb,
    Kp_vsys_map_ps_all_nights,
    Kp_vsys_map_comb_all_nights,
    nightly_snr_maps,
    max_snr,
    vsys_at_max_snr,
    kp_at_max_snr,
    sysrem_settings,):
    """Used to dump the results of scripts_transit/run_cc.py to disk as a 
    pickle file.

    Parameters
    ----------
    filename: str
        Filepath for the saved pickle.

    cc_rvs: 1D float array
        RV values that were used when cross correlating in km/s.
    
    ccv_ps: list of 4D float array
        Array of cross correlation values of shape:
        [n_sysrem_iter, n_phase, n_spec, n_rv_steps].

    ccv_comb: list of 3D float array, default: None
        3D float array of the *combined* cross correlations for each SYSREM
        iteration of shape [n_sysrem_iter, n_phase, n_rv_step].

    Kp_steps: 1D float array
        Array of Kp steps from Kp_lims[0] to Kp_lims[1] in steps of Kp_step.
    
    Kp_vsys_map_ps: list of 4D float array
        Grid of Kp_vsys maps of shape: 
        [n_sysrem_iter, n_spec, n_Kp_steps, n_rv_step]

    Kp_vsys_map_comb: list of 3D float array
        3D float array of the *combined* Kp-Vsys map of shape: 
        [n_sysrem_iter, n_Kp_steps, n_rv_step].

    Kp_vsys_map_ps_all_nights: 4D float array
        Grid of the *joint* Kp_vsys map for all nights, of shape: 
        [n_sysrem_iter, n_spec, n_Kp_steps, n_rv_step]

    Kp_vsys_map_comb_all_nights: 3D float array
        Grid of the combined *joint* Kp_vsys map for all nights, of shape: 
        [n_sysrem_iter, n_Kp_steps, n_rv_step]

    nightly_snr_maps: 3D or 4D float array
        Kp-Vsys map normalised to be in SNR units, of shape 
        [n_map, n_sysrem_iter, n_kp_steps, n_rv_steps].
    
    max_snr, vsys_at_max_snr, kp_at_max_snr: 2D float array
        Maximum SNR, and corresponding coordinates in velocity space, of shape
        [n_map, n_sysrem_iter]

    sysrem_settings: YAMLSettings object
        Settings object with attributes equivalent to YAML keys.
    """
    # Pack everything into a dictionary
    cc_dict = {
        "cc_rvs":cc_rvs,
        "ccv_ps":ccv_ps,
        "ccv_comb":ccv_comb,
        "vsys_steps":vsys_steps,
        "Kp_steps":Kp_steps,
        "Kp_vsys_map_ps":Kp_vsys_map_ps,
        "Kp_vsys_map_comb":Kp_vsys_map_comb,
        "Kp_vsys_map_ps_all_nights":Kp_vsys_map_ps_all_nights,
        "Kp_vsys_map_comb_all_nights":Kp_vsys_map_comb_all_nights,
        "nightly_snr_maps":nightly_snr_maps,
        "max_snr":max_snr,
        "vsys_at_max_snr":vsys_at_max_snr,
        "kp_at_max_snr":kp_at_max_snr,
        "sysrem_settings":sysrem_settings,}

    # Dump to disk as a pickle
    with open(filename, 'wb') as output_file:
        pickle.dump(cc_dict, output_file, pickle.HIGHEST_PROTOCOL)


def load_cc_results(filename,):
    """Counterpart function to dump_cc_results to load in the saved pickle of
    CC results.

    Parameters
    ----------
    filename: str
        Filepath for the saved pickle.

    Returns
    -------
    cc_dict: dict
        Dictionary containing the CC results and Kp_maps. Keywords are:
         - cc_rvs,
         - ccv_ps,
         - ccv_comb,
         - Kp_steps,
         - Kp_vsys_map_ps,
         - Kp_vsys_map_comb,
         - Kp_vsys_map_ps_all_nights,
         - Kp_vsys_map_comb_all_nights,
         - sysrem_settings
    """
    # Read in the dictionary
    with open(filename, 'rb') as input_file:
        cc_dict = pickle.load(input_file)

    return cc_dict


# -----------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
def calculate_transit_timestep_info(
    transit_info,
    syst_info,
    do_consider_vsini=False,):
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

    do_consider_vsini: bool, default: False
        Whether we consider vsini when computing doppler shifts. At the moment
        we consider its impact negligible so we assume gamma = beta. If this
        changes, we will need to also consider the impact parameter of the
        planet (and potentially also the fact it is a retrograde orbit?).

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
    
    gamma = star_rv_bcor / const.c.cgs.to(u.km/u.s)

    if do_consider_vsini:
        raise NotImplementedError("Currently beta = gamma, vsini assumed = 0")
        vsini_planet_epoch = syst_info.loc["vsini", "value"] * np.nan # TODO
        beta = (star_rv_bcor + vsini_planet_epoch) / const.c.cgs.to(u.km/u.s)
    else:
        beta = gamma.copy()

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
    sigmas,
    px_min,
    px_max,):
    """
    Interpolate the spectra from all phases onto a single global wavelength
    scale. The motivation for doing this is that initially each phase is likely
    to have slight offsets in the wavelength scale, so the objective is to put
    all spectral segments/orders across all phases onto a single wavelength 
    grid. When doing this, the minimum and maximum wavelengths adopted for each
    spectral segment are the maximum possible range shared by *alll* phases.

    TODO: check interpolation looks sensible.

    Parameters
    ----------
    waves, fluxes, sigmas: 3D float array
        Wavelength, flux, and sigmas vectors of shape [n_phase, n_spec, n_px].

    px_min, px_max: int
        Range of useful detector pixels avoiding bad edge pixels, that is 
        fluxes[:,:,px_min:px_max] are the useful pixels.

    Returns
    -------
    wave_new: 2D float array
        Global wavelength scale of shape [n_spec, n_px].

    obs, obs_sigmas: 2D float array
        Regridded flux and sigma arrays of shape [n_phase, n_spec, n_px].
    """
    # Grab array dimensions for convenience
    n_phase = fluxes.shape[0]
    n_spec = fluxes.shape[1]
    n_wave = fluxes.shape[2]

    # Only consider good wavelengths
    wave_adopted = waves[:, :, px_min:px_max]

    # Intialise new wavelength vector common for all phases
    wave_new = np.zeros((n_spec,n_wave))

    # Intialise new flux and sigma arrays with common wavelength scale
    obs = fluxes.copy()
    obs_sigma = sigmas.copy()

    # Loop over all spectral segments
    desc = "Interpolating wavelengths for {} spectral segments".format(n_spec)

    for spec_i in tqdm(range(0, n_spec), desc=desc, leave=False):
        # Find the common minimum and maximum wavelengths
        wl_min = np.max(wave_adopted[:, spec_i, 0])
        wl_max = np.min(wave_adopted[:,spec_i, -1])

        # Create a new wavelength scale for this spectral segment
        wave_new[spec_i] = np.arange(n_wave)/n_wave * (wl_max-wl_min) + wl_min

        # Interpolate to new wavelength scale
        for phase_i in range(0, n_phase):
            # Interpolate fluxes 
            fluxes_2 = bezier_init(
                x=wave_adopted[phase_i, spec_i],
                y=fluxes[phase_i, spec_i],)

            obs[phase_i, spec_i] = bezier_interp(
                x_a=wave_adopted[phase_i, spec_i],
                y_a=fluxes[phase_i, spec_i],
                y2_a=fluxes_2,
                x_interp=wave_new[spec_i],)
            
            # Interpolate sigmas 
            obs_sigma_2 = bezier_init(
                x=wave_adopted[phase_i, spec_i],
                y=sigmas[phase_i, spec_i],)

            obs_sigma[phase_i, spec_i] = bezier_interp(
                x_a=wave_adopted[phase_i, spec_i],
                y_a=sigmas[phase_i, spec_i],
                y2_a=obs_sigma_2,
                x_interp=wave_new[spec_i],)

    return wave_new, obs, obs_sigma


def sigma_clip_observations(
    waves,
    obs_spec,
    sigma_lower=5,
    sigma_upper=5,
    max_iterations=5,
    do_plot_sigma_clip_diagnostic=False,):
    """Sigma clips observations along the phase dimension using the median
    normalised spectra to compute a bad pixel mask. Replaces bad pixels with 
    nans.

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
    
    do_plot_sigma_clip_diagnostic: boolean, default: False
        Whether to plot a diagnostic of the sigma clipped spectra.
    
    Returns
    -------
    obs_spec: 3D float array
        Observed spectra array for each timestep of shape 
        [n_phase, n_spec, n_px].

    bad_px_mask: 3D bool array
        Bad pixel mask corresponding to obs_spec of shape 
        [n_phase, n_spec, n_px].
    """
    # Grab array dimensions for convenience
    n_phase = obs_spec.shape[0]
    n_spec = obs_spec.shape[1]
    n_wave = obs_spec.shape[2]

    # Initialise bad px mask to return
    bad_px_mask = np.full_like(obs_spec, False).astype(bool)

    # Perform our sigma clipping on *normalised* data
    med_3D = np.broadcast_to(
        np.nanmedian(obs_spec, axis=2)[:,:,None], (obs_spec.shape))
    fluxes_norm = obs_spec / med_3D

    # Create the array of fluxes we'll return
    obs_spec_clipped = obs_spec.copy()

    # Iterate over spectral pixels, and clip along the phase dimension
    desc = "Cleaning/clipping {} spectral segments".format(n_spec)

    for spec_i in tqdm(range(0, n_spec), desc=desc, leave=False):
        for px_i in range(0, n_wave):
            flux = fluxes_norm[:, spec_i, px_i].copy()

            # Iteratively sigma clip along the phase dimension for px_i
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clipped_fluxes_ma = sigma_clip(
                    data=flux,
                    sigma_lower=sigma_lower,
                    sigma_upper=sigma_upper,
                    maxiters=max_iterations,)

            bad_px = clipped_fluxes_ma.mask

            # Now replace bad pixels
            fluxes_norm[bad_px, spec_i, px_i] = np.nan
            obs_spec_clipped[bad_px, spec_i, px_i] = np.nan

            # Update bad px mask for this spectral pixel
            bad_px_mask[:, spec_i, px_i] = bad_px

    if do_plot_sigma_clip_diagnostic:
        fig, axis = plt.subplots(figsize=(20, 3))
        for phase_i in range(n_phase):
            for spec_i in range(n_spec):
                axis.plot(
                    waves[spec_i],
                    fluxes_norm[phase_i, spec_i],
                    color="k",
                    linewidth=0.5,)
        
        axis.set_title("Sigma clipping diagnostic")
        axis.set_xlabel("Wavelength (Å)")
        axis.set_ylabel("Normalised Flux")
        plt.tight_layout()

    return obs_spec_clipped, bad_px_mask


def clean_and_interpolate_spectra(
    spectra,
    e_spectra,
    mjds,
    is_A,
    n_bad_px_per_phase_threshold=5,
    sigma_threshold_phase=4.0,
    sigma_threshold_column=4.0,
    interpolation_method="cubic",
    do_extrapolation=False,):
    """Function to clean a spectral data cube prior to running SYSREM. We sigma
    clip along the phase dimension (i.e. the time series of each pixel) as well
    as along the spectral dimension. 

    Note that this function expects only spectra from a single spectral
    segment (or a flattened set of many segments.
    
    Parameters
    ----------
    spectra, e_spectra: 3D float array
        Unnormalised spectra and spectroscopic uncertainties of shape 
        [n_phase, n_spec, n_px].

    mjds: 1D float array
        MJDs associated with each phase, of shape [n_phase]. We use this for
        interpolating along the phase axis.
    
    is_A: 1D boolean array
        Mask of length [n_phase] indicating which phase belongs to which
        nodding sequence.

    n_bad_px_per_phase_threshold: int, default: 5
        Threshold for the number of bad/clipped phases per spectral pixel, 
        above which we mask out the entire column (i.e. all phases).


    sigma_threshold_phase: float, default: 4.0
        The sigma clipping threshold for when sigma clipping along the *phase*
        dimension. The sigma here refers to the characteristic sigma (over all
        phases) for a given spectral pixel, and how aberrant a particular phase
        is of one sequence (e.g. A) is compared to the mean of the other
        (e.g. B).

    sigma_threshold_column: float, default: 4.0
        The sigma clipping threshold for when determining whether to mask out
        an entire column (i.e. all phases for a given spectral pixel). The
        sigma refers to the characteristic sigma for that spectral segment, and
        we are comparing whether the mean A and B frames are sufficiently
        different from each other, which we take to mean reduction artefacts
        necessitating the masking of an entire column.

    interpolation_method: str, default: "cubic"
        Default interpolation method to use with scipy.interp1d. Can be one of: 
        ['linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic',
        'cubic', 'previous', or 'next'].

    do_extrapolation: boolean, default: False
        Whether to extrapolate edge pixels in phase (e.g. the first or last
        exposure), or whether to just mask out the entire column. Warning:
        this can cause extra artefacts.

    Returns
    -------
    spectra_clean, e_spectra_clean: 3D float array
        Cleaned and interpolated spectra + uncertainties of shape 
        [n_phase, n_spec, n_px]. Each pixel time series should either have:
        a) zero nans in the case of successful interpolation, or b) all nans,
        in the case we've masked the entire time-series.

    px_to_interp_all: 3D boolean array
        Mask indicating interpolated pixels.
    """
    # Grab shape for convenience
    (n_phase, n_spec, n_px) = spectra.shape

    if do_extrapolation:
        fill_value = "extrapolate"
    else:
        fill_value = np.nan

    # Duplicate arrays to use for output
    spectra_clean = spectra.copy()
    e_spectra_clean = e_spectra.copy()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Create mean A/B spectra to use as references
        spectra_mean_A = np.nanmean(spectra[is_A], axis=0)  # [n_spec, n_px]
        spectra_mean_B = np.nanmean(spectra[~is_A], axis=0) # [n_spec, n_px]

        # We'll be comparing A to B, and B to A, so the reference spectrum
        # corresponds to the 'opposite' sequence here. These two lists will be
        # looped over below (but are defined here for clarity).
        refs_AB = [spectra_mean_B, spectra_mean_A]
        masks_AB = [is_A, ~is_A]

        # Initialise individual pixel mask
        px_to_interp_all = np.full_like(spectra, False)

        #----------------------------------------------------------------------
        # Compute which *individual* pixels require interpolation
        #----------------------------------------------------------------------
        # Loop over both A/B sequences
        for ref_spec, seq_mask in zip(refs_AB, masks_AB):
            # Grab the number of phases present in this sequence.
            n_phase_seq = np.sum(seq_mask)
            
            # Tile our reference spectrum to 3D [n_phase_seq, n_spec, n_px].
            ref_spec_3D = np.broadcast_to(
                array=ref_spec[None,:,:],
                shape=(n_phase_seq, n_spec, n_px),)

            # Compute standard deviation per spectral px. We'll then use this
            # to assess whether a given pixel at a given phase is considered
            # an outlier to be clipped+interpolated.
            px_std_2D = np.nanstd(spectra[seq_mask], axis=0)
            px_std_3D = np.broadcast_to(
                array=px_std_2D[None,:,:],
                shape=(n_phase_seq, n_spec, n_px),)
        
            # Compute how many std deviations each pixel is from its reference.
            spectrum_n_sigma = (spectra[seq_mask] - ref_spec_3D) / px_std_3D

            # Assign all pixels beyond this threshold to be 'bad'.
            px_to_interp_all[seq_mask] = \
                np.abs(spectrum_n_sigma) > sigma_threshold_phase

        # Combine with any existing nan pixels
        px_to_interp_all = np.logical_or(px_to_interp_all, np.isnan(spectra))

        #----------------------------------------------------------------------
        # Compute which *entire* columns of pixels get masked
        #----------------------------------------------------------------------
        # In this section we are masking out entire columns (i.e. all phases)
        # of pixels. Our condition for doing this is when the mean A and B
        # spectra are sufficiently different to each other, which we take as an

        diff_AB = spectra_mean_A - spectra_mean_B
        diff_AB_std = np.nanstd(diff_AB, axis=1)
        diff_AB_std_2D = np.broadcast_to(
            array=diff_AB_std[:,None],
            shape=(n_spec, n_px),)
        
        column_n_sigma = diff_AB / diff_AB_std_2D

        do_mask_column = np.abs(column_n_sigma) > sigma_threshold_column

    #--------------------------------------------------------------------------
    # Interpolatate individual pixels flagged as aberrant
    #--------------------------------------------------------------------------
    # Loop over all spectral segments and pixels, and decide whether to
    # interpolate a given pixel, or mask out the entire column.
    for spec_i in range(n_spec):
        desc = "Cleaning spectrum {}/{}".format(spec_i+1, n_spec)
        for px_i in tqdm(range(n_px), desc=desc, leave=False):
            # For convenience, grab pixel mask for this segment/px
            px_to_interp = px_to_interp_all[:, spec_i, px_i]

            # Conditions for masking entire column:
            # a) Mean A and B px values are sufficiently different.
            # b) The column has more than the threshold level of bad px.
            # c) [If not extrapolating] The first or last px values are nan.
            nan_edges = px_to_interp[0] or px_to_interp[-1]

            if (do_mask_column[spec_i, px_i] 
                or np.sum(px_to_interp) > n_bad_px_per_phase_threshold
                or (fill_value != "extrapolate" and nan_edges)):
                # Mask out the entire column and skip the rest of the loop
                spectra_clean[:, spec_i, px_i] = np.nan
                e_spectra_clean[:, spec_i, px_i] = np.nan
                continue

            # Continue if there are no pixels needing interpolation
            if np.sum(px_to_interp) == 0:
                continue
            
            # Flux interpolator (interpolating over only non-nan px)
            interp_spectra = interp1d(
                x=mjds[~px_to_interp],
                y=spectra[:,spec_i,px_i][~px_to_interp],
                kind=interpolation_method,
                bounds_error=False,
                fill_value=fill_value,)

            # Sigma interpolator (interpolating over only non-nan px)
            interp_sigma = interp1d(
                x=mjds[~px_to_interp],
                y=e_spectra[:,spec_i,px_i][~px_to_interp],
                kind=interpolation_method,
                bounds_error=False,
                fill_value=fill_value,)

            # Interpolate fluxes and sigmas, and store
            spectra_clean[:,spec_i,px_i][px_to_interp] = \
                interp_spectra(mjds[px_to_interp])
            
            e_spectra_clean[:,spec_i,px_i][px_to_interp] = \
                interp_sigma(mjds[px_to_interp])
    
    return spectra_clean, e_spectra_clean, px_to_interp_all


def limb_darken_spectrum(spec, a_limb, mu):
    """Limb darken a given spectrum using the non-linear limb darkening law
    from Claret 2000.

    I(μ)/I(1) = 1 - Σ [a_k * (1 - μ^(k/2)) ] for k 1-4 

    Parameters
    ----------
    spec: 1D float array
        1D spectrum to limb darken.

    a_limb: 1D float array
        Array of non-linear limb darkening coefficients.

    mu: float
        μ point at which do compute limb darkening.

    Returns
    -------
    spec_ld: 1D float array
        1D limb darkened spectrum.
    """
    k_i = np.arange(len(a_limb)) + 1
    I_mu_on_I_centre = 1 - np.sum(a_limb * (1 - mu**(k_i/2)))

    spec_ld = I_mu_on_I_centre * spec

    return spec_ld


def generate_mock_flux_mu_grid(
    flux,
    a_limb,
    n_mu_samples,):
    """Generates a mock 2D flux grid of shape [n_wave, n_mu] from a 1D spectrum
    and a set of non-linear limb darkening coefficients.

    Parameters
    ----------
    flux: 1D float array
        1D spectrum flux array.

    a_limb: 1D float array
        Array of non-linear limb darkening coefficients.

    n_mu_samples: float
        Number of μ points at which to sample limb-darkening for our grid.

    Returns
    -------
    flux_grid, mu_grid: 2D float array
        Limb darkened flux and corresponding mu sampling arrays of shape
        [n_wave, n_mu].
    """
    # Uniformly sample mus from the edge to centre of the disc
    mus = np.linspace(1, 0.001, n_mu_samples,)

    # Initialise empty flux grid, and tile mus
    flux_grid = np.full((len(flux), n_mu_samples), np.nan)
    mu_grid = np.tile(mus, len(flux)).reshape((len(flux), n_mu_samples))

    # For each mu value, determine the limb-darkened flux
    for mu_i, mu in enumerate(mus):
        flux_grid[:, mu_i] = limb_darken_spectrum(flux, a_limb, mu)

    return flux_grid, mu_grid

# -----------------------------------------------------------------------------
# Handling of settings files
# -----------------------------------------------------------------------------
def load_yaml_settings(yaml_path):
    """Import our settings YAML file as a dictionary and return the object 
    equivalent.

    Parameters
    ----------
    yaml_path: string
        Path to the saved YAML file.

    Returns
    -------
    yaml_settings: YAMLSettings object
        Settings object with attributes equivalent to YAML keys.
    """
    # Load in YAML file as dictionary
    with open(yaml_path) as yaml_file:
        yaml_dict = yaml.safe_load(yaml_file)

    # Correctly set None variables
    for key in yaml_dict.keys():
        if type(yaml_dict[key]) == list:
            yaml_dict[key] = \
                [val if val != "None" else None for val in yaml_dict[key]]
        elif yaml_dict[key] == "None":
            yaml_dict[key] = None

    # Finally convert to our wrapper object form and return
    yaml_settings = YAMLSettings(yaml_dict)

    return yaml_settings


class YAMLSettings:
    """Wrapper object for settings stored in YAML file and opened with
    load_yaml_settings. Has attributes equivalent to keys in dict/YAML file.
    """
    def __init__(self, param_dict):
        for key, value in param_dict.items():
            setattr(self, key, value)

        self.param_dict = param_dict

    def __repr__(self):
        return self.param_dict.__repr__()


# -----------------------------------------------------------------------------
# Template spectra
# -----------------------------------------------------------------------------
def load_transmission_templates_from_fits(
    fits_file,
    convert_rp_rj_to_re=True,
    convert_um_to_nm=True,
    min_wl_nm=19000,
    max_wl_nm=25000,):
    """Loads wavelength, spectra, and template info for a datacube of template
    spectra generated by make_planet_transmission_grid.fits.

    By default we expect out template file to have wavelength units of microns,
    and 'flux' units of jupiter radii.

    Parameters
    ----------
    fits_file: string
        Directory to load the fits file from.

    convert_rp_rj_to_re: bool, default: True
        Whether to convert radii in units of R_Jupiter to R_Earth.

    convert_um_to_nm: bool, default: True
        Whether to convert the wavelength scale from um to nm.

    min_wl_nm, max_wl_nm: float, default: 19000, 25000
        Minimum and maximum in nm for the wavelength scale.

    Returns
    -------
    waves: float array
        Common wavelength scale for template spectra of shape [n_px].
    
    template_trans: 2D float array
        2D grid of template transmission spectra of shape  [n_spec, n_px].
    
    template_info: pandas DataFrame
        DataFrame of length [n_spec] containing booleans indicating which
        molecules are in each template spectrum.
    """
    # Load in the fits file
    with fits.open(fits_file, mode="readonly") as fits_file:
        # Load data constant across transits
        wave = fits_file["WAVE"].data.astype(float)
        spec = fits_file["SPECTRA"].data.astype(float)
        template_info = Table(fits_file["TEMPLATE_INFO"].data).to_pandas()

    # [Optional] Convert from Jupiter radii to Earth radii
    if convert_rp_rj_to_re:
        r_e = const.R_earth.value       # Earth radii in metres
        r_j = const.R_jup.value         # Jupiter radii in metres
    
        spec *= r_j / r_e

    # [Optional] Convert to nm
    if convert_um_to_nm:
        wave *= 10000

    # Enforce wavelength bounds
    wl_mask = np.logical_and(wave > min_wl_nm, wave < max_wl_nm)
    wave = wave[wl_mask]
    spec = spec[:,wl_mask]

    # All done, return
    return wave, spec, template_info


def load_telluric_spectrum(
    molecfit_fits,
    tau_fill_value=0,
    convert_to_angstrom=True,
    convert_to_nm=False,
    output_transmission=False,):
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

    TODO: merge convert_to_angstrom and convert_to_nm into a single variable,
    neaten output of transmission 

    Parameters
    ----------
    molecfit_file: string
        Filepath to molecfit fitted telluric spectrum.

    tau_fill_value: float, default: 0
        Default value for missing values in the molecfit fitted spectrum.

    convert_um_to_angstrom: boolean, default: True
        Whether to convert the wavelength scale in um to Angstrom.

    TODO

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
    elif convert_to_nm:
        telluric_wave *= 1E3

    if output_transmission:
        telluric_trans = 10**-telluric_tau
        return telluric_wave, telluric_tau, calc_telluric_tau, telluric_trans
    else:
        return telluric_wave, telluric_tau, calc_telluric_tau


def create_viper_telluric_spectrum(
    viper_fits="data/viper_stdAtmos_K.fits",
    include_H2O=True,
    tau_scale_H2O=1.0,
    tau_scale_non_H2O=1.0,
    do_broaden=True,
    resolving_power=100000,
    convert_to_um=False,):
    """Function to create a telluric spectrum from component telluric species
    transmission spectra and broaden to instrumental resolution. To account for
    the variability of tellurics, we include two optical depth scaling terms,
    one for H2O and one for non-H2O species, that are applied as multiplicative
    terms to the optical depth.

    Telluric transmission spectra are sourced from:
     - https://github.com/mzechmeister/viper/tree/master/lib/atmos

    Parameters
    ----------
    viper_fits: str, default: 'data/viper_stdAtmos_K.fits'
        Path to fits file containing model per-species transmission spectra.
    
    include_H2O: bool, default: True
        Whether to include H2O in the combined transmission spectrum, or
        exclude to apply later.

    tau_scale_H2O, tau_scale_non_H2O: float, default: 1.0
        Optical depth scaling terms for H2O and the non-H2O species
        respectfully.

    do_broaden: bool, default: True
        Whether to broaden our spectra to some instrumental resolution.

    resolving_power: int, default: 100000
        Resolving power to broaden the final transmission spectrum to.

    convert_to_um: bool, default: False
        Whether to convert from Ångström (default) to um.
        
    Returns
    -------
    wave, trans: float array
        Telluric wavelength scale and transmission spectra arrays.
    """
    # Molecular species to consider. Note this will fail if the given viper
    # fits file does not have these species (e.g. in the optical).
    K_BAND_SPECIES = ["H2O", "CH4", "CO", "CO2", "N2O"]

    # Intialise a dictionary to store the optical depths.
    telluric_tau = {}

    # Load transmission spectra (as optical depths) for all species
    with fits.open(viper_fits) as vfits:
        # Grab wavelength array and convert to um
        wave = vfits[1].data["lambda"]

        # [Optional] Convert to um
        if convert_to_um:
            wave /= 10000

        for species in K_BAND_SPECIES:
            # Import, clean
            trans = vfits[1].data[species]
            trans[np.isnan(trans)] = 1.0        # Set NaN values to 1.0
            trans[trans == 0] = 1E-15           # Avoid infinity

            telluric_tau[species] = -np.log(trans)
    
    # Create our combined transmission vector, (possibly) applying a different
    # optical depth scaling term to the H2O and non-H2O terms.
    trans = np.ones_like(wave)

    for species in K_BAND_SPECIES:
        if species == "H2O" and include_H2O:
            trans *= 10**-(telluric_tau["H2O"]*tau_scale_H2O)
        else:
            trans *= 10**-(telluric_tau[species]*tau_scale_non_H2O)

    # [Optional] Broaden to instrumental resolution
    if do_broaden:
        trans = instrBroadGaussFast(
            wvl=wave,
            flux=trans,
            resolution=resolving_power,
            edgeHandling="firstlast",
            maxsig=5,
            equid=True,)
    
    return wave, trans


def prepare_cc_template(
    planet_fits_fn,
    species_to_cc,
    syst_info,
    templ_wl_nm_bounds=(16000,30000),
    continuum_resolving_power=300,):
    """Function to import and setup an exoplanet cross-correlation template to
    use with scripts_transit/run_cc.py. The fits file format is as output by
    scripts_transit/make_planet_transmission_grid_fits.py.

    Parameters
    ----------
    planet_fits_fn: str
        Filename/path to the template fits file
    
    species_to_cc: str list
        List of molecular species model in the transit, e.g. ['H2O', 'CO'].

    syst_info: pandas DataFrame
        Pandas dataframe containing the planet properties.

    templ_wl_nm_bounds: float tuple, default: (16000,30000)
        Wavelength limits in nm when importing planet template.

    continuum_resolving_power: float, default: 300
        Resolving power for determining the 'continuum' of the planet spectrum.

    Returns
    -------
    wave_template, spectrum_template: 1D float array
        Template wavelength and spectrum vectors of length [n_wave].
    """
    #----------------------------------------------------------------------
    # Setup and template selection
    #----------------------------------------------------------------------
    # Load in petitRADRTRANS datacube of templates. These templates will be
    # in units of R_earth as a function of wavelength.
    wave_p, spec_p_all, templ_info = load_transmission_templates_from_fits(
        fits_file=planet_fits_fn,
        min_wl_nm=templ_wl_nm_bounds[0],
        max_wl_nm=templ_wl_nm_bounds[1],)

    # Clip edges to avoid edge effects introduced by interpolation
    spec_p_all = spec_p_all[:,10:-10]
    wave_p = wave_p[10:-10] / 10

    # Now select the appropriate template from trans_planet_all simulating
    # the appropriate set of molecules. Raise an exception if we don't have
    # a template with that combination of molecules.
    molecule_cols = templ_info.columns.values
    has_molecule = np.full_like(molecule_cols, False)

    for mol_i, molecule in enumerate(molecule_cols):
        if molecule in species_to_cc:
            has_molecule[mol_i] = True

    match_i = np.argwhere(np.all(has_molecule==templ_info.values, axis=1))

    if len(match_i) == 0:
        raise ValueError("Invalid molecule combination!")
    else:
        Rp_Re_vs_lambda_planet = spec_p_all[int(match_i)]

    # Convert to a transmission spectrum
    r_e = const.R_earth.si.value
    r_odot = const.R_sun.si.value

    rp = syst_info.loc["r_planet_rearth", "value"] 
    rs = syst_info.loc["r_star_rsun", "value"] * r_odot / r_e

    trans_planet = 1 - ((Rp_Re_vs_lambda_planet + rp)**2  / rs**2)

    #----------------------------------------------------------------------
    # Continuum normalisation
    #----------------------------------------------------------------------
    # Compute planet 'continuum' to normalise by
    planet_cont = instrBroadGaussFast(
        wvl=wave_p,
        flux=trans_planet,
        resolution=continuum_resolving_power,
        equid=True,
        edgeHandling="firstlast",)

    # Final continuum normalised planet spectrum with edges clipped again
    wave_template = wave_p[10:-10]
    spectrum_template = trans_planet[10:-10] / planet_cont[10:-10]

    # All done, return
    return wave_template, spectrum_template