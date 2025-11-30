"""Script to take output of NP IDL regridded CRIRES+ transits and put them back
in the luciferase standard fits file format.
"""
import numpy as np
import transit.utils as tu
from scipy.io import readsav

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------
# Settings for the fits file we want to swap out the wavlength scale, fluxes,
# and uncertainties for (though we save under a new name). Ideally this should
# be the same file as was originally given to the IDL routines.
save_path = "simulations"
n_transit = 2
label = "wasp107_np_corr"

# New label for the file to avoid overwriting our existing file.
new_label = "wasp107_np_corr_251129"

# If false, we always aim to have n_px = 2048 per what comes out of the CRIRES+
# pipeline, regardless of any clipping that has happened in IDL. We accomplish
# this by extrapolating any NaNs in the wavelength scale (but leave fluxes and
# sigmas as is, which should leave those pixels to remain unused. If true, we
# don't care and propagate forward whatever the new n_px value is. 
# TODO: currently setting this to true causes problems later with molecfit
# templates which have n_px = 2048, so best to leave as False.
accept_npx_difference = False

#------------------------------------------------------------------------------
# Main Operation
#------------------------------------------------------------------------------
# Load fits file:
# waves: float array
#        Wavelength scales for each timestep of shape [n_phase, n_spec, n_px].
#    
#    obs_spec_list: list of 3D float arrays
#        List of observed spectra (length n_transits), with one 3D float array
#        transit of shape  [n_phase, n_spec, n_px].
#    
#    sigmas_list: list of 3D float arrays
#        List of observed sigmas (length n_transits), with one 3D float array
#        transit of shape  [n_phase, n_spec, n_px].
waves, spec_list, sigmas_list, det, orders, transit_info_list, syst_info = \
    tu.load_transit_info_from_fits(
        save_path,
        label,
        n_transit=n_transit,)

# Initialise output arrays
waves_new = np.full_like(waves, np.nan)
spec_list_new = []
sigmas_list_new = []

nightly_idl_sav_files = [
    "simulations/wasp107_wave_corr_n1_251129.sav",
    "simulations/wasp107_wave_corr_n2_251129.sav",]

# Loop over nights
for night_i in range(n_transit):
    # Import sav file
    sav_file = readsav(nightly_idl_sav_files[night_i])
    
    # Stack all arrays for this night into shape [n_spec, n_px]
    waves_night = np.vstack([
        sav_file["wave1"], sav_file["wave2"], sav_file["wave3"]])
    
    obs_night = np.concatenate([
        sav_file["obs1"], sav_file["obs2"], sav_file["obs3"]], axis=1)
    
    sigma_night = np.concatenate([
        sav_file["un1"], sav_file["un2"], sav_file["un3"]], axis=1)
    
    # Sort in ascending order of wavelength
    waves_mean = np.mean(waves_night, axis=1)
    waves_ii = np.argsort(waves_mean)

    waves_night = waves_night[waves_ii]
    obs_night = obs_night[:,waves_ii,:]
    sigma_night = sigma_night[:,waves_ii,:]

    # If this night is >= 1, enforce that the wavelengths scale is the same
    if night_i >= 1:
        is_nan = np.isnan(waves_new)
        assert np.nansum(waves_night.ravel() - waves_new[~is_nan]) == 0

    # Grab pixel count for IDL data
    (_, _, n_px_idl) = obs_night.shape

    # Initialise empty flux and sigma arrays for this night
    (n_phase, n_spec, n_px_full) = spec_list[night_i].shape

    spec_new = np.full_like(spec_list[night_i], np.nan)
    sigma_new = np.full_like(sigmas_list[night_i], np.nan)

    # Pixel numbers are exactly the same OR we don't care about px numbers
    if n_px_idl == n_px_full or accept_npx_difference:
        waves_new = waves_night
        spec_list_new.append(obs_night)
        sigmas_list_new.append(sigma_night)

    # IDL pixels have been clipped, insert into full array to preserve shape
    elif n_px_idl <= n_px_full:
        # Determine the number of edge pixels to come in by
        n_edge = (n_px_full - n_px_idl) // 2

        waves_new[:, n_edge:(n_edge+n_px_idl)] = waves_night
        spec_new[:, :, n_edge:(n_edge+n_px_idl)] = obs_night
        sigma_new[:, :, n_edge:(n_edge+n_px_idl)] = sigma_night

        spec_list_new.append(spec_new)
        sigmas_list_new.append(sigma_new)

    else:
        raise Exception("Something has gone wrong.")

# [Optional] Fix nans in the wavelength scale
if n_px_idl < n_px_full and not accept_npx_difference:
    # To avoid having nans in our wavelength scale, we need to extrapolate
    # a dummy wavelength scale for the edge pixels. While we can probably
    # do this in a nice vectorised way, we're just going to use a loop for
    # (conceptual) simplicity.
    for spec_i in range(waves_new.shape[0]):
        wave = waves_new[spec_i]

        # Start of spectral segment
        delta_lambda = np.median(np.diff(wave[n_edge:n_edge*2+1]))
        lambda_edge = wave[n_edge] - n_edge*delta_lambda

        waves_new[spec_i, :n_edge] = \
            np.arange(lambda_edge, wave[n_edge], delta_lambda)

        # End of spectral segment. For the end section, we can't simply assume
        # that we have n_edge px missing, as there might be an odd number of px
        # clipped, and this is where it will manifest. As such, we reference
        # from the front of the array, rather than backwards.
        last_idl_px = n_edge + n_px_idl - 1
        remaining_px = len(wave[last_idl_px+1:])
        delta_lambda = np.median(np.diff(wave[-2*n_edge:last_idl_px]))
        lambda_edge = wave[last_idl_px] + (remaining_px+1)*delta_lambda

        # Interpolate wavelengths to the edge of the detector. Note that due to
        # (presumably) how we've computed lambda_edge, we can end up with one
        # more wavelength point than required if there are an odd number of px
        # missing. As such, we do a HACK and slice to remaining_px+1 when
        # assigning this new wavelength scale to waves_new.
        edge_interp = np.arange(wave[last_idl_px], lambda_edge, delta_lambda)

        waves_new[spec_i, last_idl_px+1:] = edge_interp[1:remaining_px+1]

        assert np.sum(np.isnan(waves_new[spec_i])) == 0

# Save to new fits file. Note that we assume that by ordering things in
# wavelength order as was done originally, the 'det' and 'ord' arrays loaded in
# will correspond to the IDL data.
tu.save_transit_info_to_fits(
    waves=waves_new,
    obs_spec_list=spec_list_new,
    sigmas_list=sigmas_list_new,
    n_transits=n_transit,
    detectors=det,
    orders=orders,
    transit_info_list=transit_info_list,
    syst_info=syst_info,
    fits_save_dir=save_path,
    label=new_label,)