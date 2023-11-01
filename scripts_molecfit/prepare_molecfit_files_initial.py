"""
Script to produce an initial/naive set of Molecfit input files. This should be
run as follows:
    1) run prepare_molecfit_files_initial.py (***this file!***)
        - This produces an initial set of Molecfit input files using an initial
          best guess linear continuum fit. We mask out strong stellar lines
          during this fit, but the RV fit is done without masking the (very 
          prevalent) telluric features.
    2) Run Molecfit
    3) Run prepare_molecfit_files_optimised.py
        - This produces a second (or Nth in the case of iterating multiple
          times) set of Molecfit files. This time we mask out the tellurics
          when fitting for the RV, and use the initial Molecfit telluric model
          to optimise the placement of the continuum. This also dumps the
          optimised polynomial coefficients used to normalise the continuum,
          which can later be re-imported to continuum normalise spectra when
          e.g. running SYSREM.

Steps for Running Molecfit:
 - Duplicate molecfit_master folder on neon
 - Copy MAPPING_ATMOSPHERIC.fits, PIXEL_EXCLUDE.fits, SCIENCE.fits, and 
   WAVE_INCLUDE.fits into this folder.
 - Run molecfit.sh
"""
import os
import transit.utils as tu
import luciferase.spectra as ls

# -----------------------------------------------------------------------------
# Settings + filename setup
# -----------------------------------------------------------------------------
# Import settings YAML file
molecfit_settings = "scripts_molecfit/spectra_settings.yml"
ms = tu.load_yaml_settings(molecfit_settings)

# Construct filenames
cwd = os.getcwd()

# Science spectrum
sci_spec_path = os.path.join(ms.data_dir, ms.sci_spec_fits)

night_path = os.path.join(cwd, ms.working_dir, ms.night_folder,)

# Folder to save Molecfit output files to
molecfit_init_path = os.path.join(
    cwd, ms.working_dir, ms.night_folder, ms.molecfit_init_folder)

# Path to the RV template to use for this star
rv_template_path = os.path.join(cwd, ms.working_dir, ms.rv_template_fits)

# Ensure molecfit folder exists to save to
if not os.path.isdir(molecfit_init_path):
    os.mkdir(molecfit_init_path)

# -----------------------------------------------------------------------------
# Running things
# -----------------------------------------------------------------------------
# Initialise the provided spectrum as a 'Observation' object, itself containing
# spectra for each order plus other information about the observation. This
# object has associated functions for continuum normalising and producing the
# files Molecfit needs to run.
print("Importing spectra...")
ob = ls.initialise_observation_from_crires_fits(
    fits_file_extracted=sci_spec_path,
    slit_pos_spec_num=ms.slit_pos_spec_num,)

# We don't trust Molecfit to run its own continuum fitting, so we do so here
# with a simple linear fit to selected 'continuum' regions for each spectral
# segment. On the first time through with a given star, the user is asked to
# select these regions manually via matplotlib plot input, the values of which
# are then saved and can be imported on subsequent runs through.
print("Continuum normalising spectra...")
if ms.do_identify_continuum_regions:
    ob.identify_continuum_regions()
    ob.save_continuum_wavelengths_to_file(save_path=night_path)
else:
    ob.load_continuum_wavelengths_from_file(load_path=night_path)

ob.fit_polynomials_for_spectra_continuum_wavelengths()
ob.continuum_normalise_spectra()

# Use our template spectrum to fit for the RV of the observation. The best RV
# value is saved to the object, and we return the fit dictionary summarising
# the fitting. We use the best-fit RV in conjunction with a stellar template to
# mask out stellar absorption when running Molecfit.
print("Running RV fitting...")
rv_fit_dict = ob.fit_rv(
    template_spectrum_fits=rv_template_path,
    segment_contamination_threshold=ms.segment_contamination_threshold,
    ignore_segments=ms.ignore_segments,
    px_absorption_threshold=ms.px_absorption_threshold,
    rv_init_guess=ms.rv_initial_guess,
    rv_min=ms.rv_limits[0],
    rv_max=ms.rv_limits[1],
    delta_rv=ms.rv_step,
    fit_method=ms.fit_method,
    do_diagnostic_plots=ms.do_rv_diagnostic_plots,
    verbose=ms.verbose_rv_fit,
    fig_save_path=night_path,)

# Write the following molecfit input files based on our science spectrum:
# 1) SCIENCE.fits            --> science spectra
# 2) WAVE_INCLUDE.fits       --> wavelengths to include in modelling
# 3) ATM_PARAMETERS_EXT.fits --> atmospheric parameter mapping
# 4) PIXEL_EXLUDE.fits       --> pixels to exclude from modelling
print("Saving Molecfit files...")
ob.save_molecfit_fits_files(molecfit_init_path,)
ob.save_molecfit_pixel_exclude(
    molecfit_init_path,
    do_science_line_masking=ms.do_science_line_masking,
    do_plot_exclusion_diagnostic=ms.do_plot_exclusion_diagnostic,
    fig_save_path=night_path,)