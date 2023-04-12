"""
Script to downsample a planet spectrum to a lower spectroscopic resolution
and coarser wavelength sampling such that it isn't so unwieldy when simulating.
"""
import transit.simulator as sim
from PyAstronomy.pyasl import instrBroadGaussFast

new_resolving_power = 200000
wl_downsample_fac = 10

planet_wave_fits = "data/W107b_spec_H2O_main_iso_CO_all_iso_clear_wave.fits"
planet_spec_fits = "data/W107b_spec_H2O_main_iso_CO_all_iso_clear_flux.fits"

wave_planet, flux_planet = sim.load_planet_spectrum(
    wave_file=planet_wave_fits,
    spec_file=planet_spec_fits,
    convert_cm_to_angstrom=False,)

new_wl_scale = wave_planet[::wl_downsample_fac]

# Convolve planet flux to new resolution
print("Broaden planet flux")
planet_flux_broad = instrBroadGaussFast(
    wvl=wave_planet,
    flux=flux_planet,
    resolution=new_resolving_power,
    equid=True,)

# Interpolate planet flux to new wavelength scale
print("Interpolate planet flux")
planet_spec_interp = sim.interpolate_spectrum(
    wave_new=new_wl_scale,
    wave_old=wave_planet,
    flux_old=planet_flux_broad,)

# Update filenames and save
new_wave_file = planet_wave_fits.replace(
    ".fits", "_R{}.fits".format(new_resolving_power))
new_flux_file = planet_spec_fits.replace(
    ".fits", "_R{}.fits".format(new_resolving_power))

sim.save_planet_spectrum(
    new_wave_file, new_flux_file, new_wl_scale, planet_spec_interp)



