"""Script to take petitRADTRANS .txt spectra generated with different molecules
for the same planet and to a) convolve to the instrument resolution, and b) 
combine all templates into a single fits file.
"""
import os
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d
from PyAstronomy.pyasl import instrBroadGaussFast

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------
# Molecules to look for. These will be extracted from the file names.
MOLECULES = ["CH4", "CO", "CO2", "H2O"]

# Folder to look for templates. Templates are assumed to already be in the
# formate of transmission spectra.
folder = "templates/WASP17b"

# Any files in this folder with extension as ext are considered templates
ext = "txt"

# New resolving power to convolve to
new_resolving_power = 100000

# Since we've dropped the resolution, we drop the wavelength sampling too.
wl_downsample_fac = 10

fits_save_name = "WASP17_templates_all_R{}.fits".format(new_resolving_power)

#------------------------------------------------------------------------------
# Spectra extraction, convolution, and resampling
#------------------------------------------------------------------------------
# Import files
all_model_paths = glob.glob(os.path.join(folder, "*.{}".format(ext)))
all_model_paths.sort()

has_molecules_all = []

for model_i, model_path in enumerate(all_model_paths):
    print("Template {:0.0f}/{:0.0f}: {}".format(
        model_i+1, len(all_model_paths), model_path))
    
    # Extract spectrum
    template = np.loadtxt(model_path)
    wave = template[:,0]
    spec = template[:,1]
    
    # Convolve spectrum to new new_resolving_power
    spec_broad_fine = instrBroadGaussFast(
        wvl=wave,
        flux=spec,
        resolution=new_resolving_power,
        equid=True,)

    # Interpolate this onto a coarser wavelength scale
    wave_new = wave[::wl_downsample_fac]
    spec_interp = interp1d(
            x=wave,
            y=spec_broad_fine,
            bounds_error=False,
            fill_value=np.nan,)
        
    spec_broad = spec_interp(wave_new)

    # Initialise spectra vector with correct shape
    if model_i == 0:
        all_spec = np.ones((len(all_model_paths), len(wave)))
        all_spec_broad = np.ones((len(all_model_paths), len(wave_new)))
    
    all_spec[model_i,:] = spec
    all_spec_broad[model_i,:] = spec_broad

    # Store molecular information
    has_molecules = []
    
    # Extract molecules from path
    # Assumed format: <planet>_Trans_<mol_1>+<mol_2>+<mol_n.txt
    molecules_in_fn = model_path.split("/")[-1].split("_")[-1][:-4].split("+")

    for molecule in MOLECULES:
        if molecule in molecules_in_fn:
            has_molecules.append(True)
        else:
            has_molecules.append(False)

    has_molecules_all.append(has_molecules)

# Initialise dataframe
template_df = pd.DataFrame(columns=MOLECULES, data=has_molecules_all,)

#------------------------------------------------------------------------------
# Save spectra as fits file
#------------------------------------------------------------------------------
# Intialise HDU List
hdu = fits.HDUList()

# HDU 0: wavelength scale
wave_img =  fits.PrimaryHDU(wave_new)
wave_img.header["EXTNAME"] = ("WAVE", "Wavelength scale")
hdu.append(wave_img)

# HDU 1: spectra
spectra_img =  fits.PrimaryHDU(all_spec_broad)
spectra_img.header["EXTNAME"] = ("SPECTRA", "Planet transmission spectra.")
hdu.append(spectra_img)

# HDU 3: table of planet system information
info_tab = fits.BinTableHDU(Table.from_pandas(template_df))
info_tab.header["EXTNAME"] = ("TEMPLATE_INFO", "Table molecules per template.")
hdu.append(info_tab)

# Done, save
fits_file = os.path.join(folder, fits_save_name)
hdu.writeto(fits_file, overwrite=True)