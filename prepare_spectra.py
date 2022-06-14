"""
Script to prepare reduced and blaze corrected CRIRES+ spectra for analysis with
SME. Note that by necessity this process is iterative, and should proceed as
follows:
    1) Do initial RV fit (without knowledge of which regions are stongly
       affected by telluric absorption) using a broad cross-correlation
       approach.
    2) Produce input files for Molecfit with pixel masking based on regions
       where the template spectrum (interpolated to science rest frame using
       initial RV) has excessive absorption.
    3) Run Molecfit
    4) Do second RV fit, this time excluding regions affected by telluric 
       absorption using the Molecfit model and using a more focused least 
       squares approach.
    5) Optimise continuum placement using telluric model.
    6) Produce new input files for molecfit with better pixel masking
    7) Run Molecfit a second time.
"""
import os
import luciferase.spectra as lspec
import luciferase.utils as lutils

# -----------------------------------------------------------------------------
# Setup, parameters, and filenames
# -----------------------------------------------------------------------------
DO_IDENTIFY_CONTINUUM_REGIONS = False
DO_OPTIMISE_CONTINUUM_PLACEMENT = True
DO_SAVE_MOLECFIT_MODEL = True
PLOT_SPECTRA = True

# Setup filenames
base_dir = "data_reduction"
obj_dir = "wasp_107"
molecfit_dir = "molecfit"

sci_spec_fits = "wasp107_cr2res_obs_nodding_extracted_combined_blaze_corr.fits"

vald_linelist_txt = "vald_wasp107_0.05.txt"

#molecfit_model_fits = "wasp_107_molecfit_model.fits"
molecfit_model_fits = "wasp_107_molecfit_corrected_rv.fits"

rv_template_fits = "template_wasp_107.fits"

# Construct filenames
cwd = os.getcwd()

sci_spec_path = os.path.join(cwd, base_dir, obj_dir, sci_spec_fits)
vald_line_list_path = os.path.join(cwd, base_dir, obj_dir, vald_linelist_txt)
molecfit_out_path = os.path.join(cwd, base_dir, obj_dir, molecfit_dir)
molecfit_model_path = os.path.join(cwd, base_dir, obj_dir, molecfit_model_fits)
rv_template_path = os.path.join(cwd, base_dir, obj_dir, rv_template_fits)

# Ensure molecfit folder exists to save to
if not os.path.isdir(molecfit_out_path):
    os.mkdir(molecfit_out_path)

# -----------------------------------------------------------------------------
# Preparing spectra
# -----------------------------------------------------------------------------
# Initialise object
ob = lspec.initialise_observation_from_crires_fits(
    fits_file_extracted=sci_spec_path)

# Do continuum fitting (either from scrath or via pre-computed params)
if DO_IDENTIFY_CONTINUUM_REGIONS:
    ob.identify_continuum_regions()
    ob.save_continuum_wavelengths_to_file()
else:
    ob.load_continuum_wavelengths_from_file()

ob.fit_polynomials_for_spectra_continuum_wavelengths()
ob.continuum_normalise_spectra()

# Fit RV. There are two methods for this, CC for cross correlation, or LS for
# LS. CC is recommended for doing a broad RV search initially.
rv_fit_dict = ob.fit_rv(
    template_spectrum_fits=rv_template_path,
    segment_contamination_threshold=0.95,
    ignore_segments=[],
    px_absorption_threshold=0.9,
    ls_diff_step=(0.1),
    rv_init_guess=0,
    rv_min=-200,
    rv_max=200,
    delta_rv=1,
    fit_method="CC",
    do_diagnostic_plots=False,
    verbose=False,
    figsize=(16,4),)

# If we already have a Molecfit telluric model, we can load this in and use
# it to optimise the placement of the stellar continuum before writing the
# files necessary to run Molecfit again.
if DO_OPTIMISE_CONTINUUM_PLACEMENT and os.path.exists(molecfit_model_path):
    ob.initialise_molecfit_best_fit_model(
        molecfit_model_fits_file=molecfit_model_path,
        convert_um_to_nm=True,)

    # Optimise continuum placement
    ob.optimise_continuum_fit_using_telluric_model(
        do_mask_uninformative_model_px=True,
        do_mask_strong_stellar_lines=True,)

# Save the Molecfit model
if DO_SAVE_MOLECFIT_MODEL:
    ob.save_molecfit_fits_files(molecfit_out_path,)
    ob.save_molecfit_pixel_exclude(
        molecfit_out_path,
        do_science_line_masking=True,
        do_plot_exclusion_diagnostic=True)

# Plot molecfit model against spectra + linelist 
vald_linelist = lspec.read_vald_linelist(vald_line_list_path)

if PLOT_SPECTRA:
    ob.plot_spectra(
        line_list=vald_linelist,
        plot_molecfit_model=True,
        line_depth_threshold=0.0,
        do_save=True,
        figsize=(25,4),
        line_annotation_fontsize=7,
        linewidth=0.2,)
