# Reducing CRIRES+ Planet Atmospheres Data
### Setup
1. Copy python scripts into directory (for a record of script version/settings)

### Reduce Calibration Files
2. `python make_calibration_sof.py [wl_setting]`
3. `./reduce_cals.sh`

If there’s a duplicate set of flats with varying exposure times, check to see if either set are saturated, and if not, take those with the longer exposures.

### Reduce Master Spectrum
4. `python make_nodding_sof_master.py [wl_setting]`
5. `[wl_setting]/reduce_master_[wl_setting].sh`

### Reduce Time Series Spectra
6. `python make_nodding_sof_split_nAB.py [nAB] [wl_setting]/[wl_setting].sof`
7. `./reduce_1xAB.sh`

### Blaze Correction
8. `python blaze_correct_nodded_spectra.py [data_path] [cal_flat_blaze_path]`

### Reduction Quality Inspection
9. `python make_diagnostic_reduction_plots.py [data_path] [run_on_blaze_corr_spec]`

Inspect resulting pdfs
