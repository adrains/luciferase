"""Script to fit the systemic RV to a CRIRES datacube.
"""
import numpy as np
import transit.utils as tu
from astropy import constants as const
import luciferase.utils as lu
from scipy.interpolate import interp1d
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------
n_transits = 2
telluric_mask_thresold = 0.9

rv_span = (5,25)
rv_step = 0.1
rv_steps = np.arange(rv_span[0], rv_span[1]+rv_step, rv_step)

n_rv = len(rv_steps)

# Import observed spectra 'as is'
save_path = "simulations"
n_transit = 2
label = "wasp107_np_corr"

waves, fluxes_list, sigmas_list, det, orders, transit_info_list, syst_info = \
    tu.load_transit_info_from_fits(save_path, label, n_transit)

# Import telluric template for masking
molecfit_fits = "data_reduction/WASP107/220310_WASP107/molecfit_results/MODEL/BEST_FIT_MODEL.fits"

telluric_wave, _, _, telluric_trans = tu.load_telluric_spectrum(
    molecfit_fits=molecfit_fits,
    tau_fill_value=0,
    convert_to_angstrom=False,
    convert_to_nm=True,
    output_transmission=True,)

# reshape
telluric_trans = telluric_trans.reshape(waves.shape)

masked_px = telluric_trans < telluric_mask_thresold

# Import *continuum normalised* stellar template
stellar_template_fits = "data_reduction/WASP107/template_wasp_107.fits"

wave_stellar, spec_stellar = lu.load_plumage_template_spectrum(
    template_fits=stellar_template_fits,
    do_convert_air_to_vacuum_wl=True,)

# Create interpolator for stellar spectrum
interp_ref_flux = interp1d(
    x=wave_stellar,
    y=spec_stellar,
    kind="cubic",
    bounds_error=False,)

colours = ["r", "b"]

plt.close("all")

for transit_i in range(n_transits):
    # Load in continuum normalised spectra for this transit
    fluxes_norm, sigmas_norm, _, _ = tu.load_normalised_spectra_from_fits(
        fits_load_dir=save_path,
        label=label,
        n_transit=n_transit,
        transit_i=transit_i,)
    
    fluxes_norm[:,masked_px] = np.nan
    
    (n_phase, n_spec, n_px) = fluxes_norm.shape

    fluxes_norm_c = fluxes_norm.reshape((n_phase, n_spec*n_px,))

    # Grab barycentric velocities for this night
    bcors = -1*transit_info_list[transit_i]["bcor"].values

    # Create storage array
    cc_values = np.full((n_rv, n_phase), np.nan)

    desc = "CCing RVs for transit {}/{}".format(transit_i+1, n_transit)

    for rv_i, rv in enumerate(tqdm(rv_steps, leave=False, desc=desc)):
        # Generate template spectrum for each spectral segment, tile to all
        # phases, cross correlate, move on
        flux_ref_3D = np.full((n_phase, n_spec, n_px), np.nan)

        for phase_i in range(n_phase):
            bcor = bcors[phase_i]

            for spec_i in range(n_spec):
                # Interpolate to this RV
                flux_ref = interp_ref_flux(
                    waves[spec_i] * (1-(rv+bcor)/(const.c.si.value/1000)))
                
                flux_ref_3D[phase_i, spec_i] = flux_ref

        # Concatenate spectral segment and px dimensions
        flux_ref_2D = flux_ref_3D.reshape((n_phase, n_spec*n_px,))

        # Compute CC, collapse in spectral dimension to give [n_phase]
        cc_num = np.nansum(flux_ref_2D * fluxes_norm_c, axis=1)
        cc_den = np.sqrt(
            np.nansum(flux_ref_2D**2, axis=1)
            * np.nansum(fluxes_norm_c**2, axis=1))
        cc_values[rv_i,:] = cc_num / cc_den
    
    # Normalise CC (divide by sum in CC dimension)
    cc_sum = np.nansum(cc_values, axis=0)
    cc_sum_2D = np.broadcast_to(cc_sum[None,:], (n_rv, n_phase))

    cc_values /= cc_sum_2D

    # Compute median CC
    cc_1D = np.nanmedian(cc_values, axis=1)

    amp = np.max(cc_1D)
    rv_max = rv_steps[np.argmax(cc_1D)]

    # Fit Gaussian
    #gaussian_init = models.Gaussian1D(amplitude=amp, mean=0, stddev=1.)
    moffat_init = models.Moffat1D(amplitude=amp, x_0=rv_max, gamma=1, alpha=1)
    fit_curve = fitting.LevMarLSQFitter()
    #gaussian = fit_curve(gaussian_init, rv_steps, cc_1D)
    moffat = fit_curve(moffat_init, rv_steps, cc_1D)

    plt.plot(
        rv_steps,
        cc_values[:,phase_i],
        linewidth=0.5,
        label="CC {}, RV = {:0.2f} km/s".format(transit_i+1, rv_max),
        c=colours[transit_i])
    
    #plt.plot(
    #    rv_steps,
    #    gaussian(rv_steps),
    #    ".",
    #    c=colours[transit_i],
    #    label="Gaussian #{}, RV = {:0.2f} km/s".format(
    #        transit_i+1, gaussian.mean.value))
    
    plt.plot(
        rv_steps,
        moffat(rv_steps),
        ".",
        c=colours[transit_i],
        label="Moffat #{}, RV = {:0.2f} km/s".format(
            transit_i+1, moffat.x_0.value))

plt.legend()





