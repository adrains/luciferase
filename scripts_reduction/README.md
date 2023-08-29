# Downloading CRIRES+ Planet Atmosphere Data
1. Use CRIRES specific ESO download portal (make sure to log in first!):
    - https://archive.eso.org/wdb/wdb/eso/crires/form
    - For **_science_** data: fill in “Target name”, “DATE OBS”, “INS WLEN ID” then Mark All, Request marked datasets. Select raw data, associated processed calibrations, and associated raw calibrations. Make sure to check the night report has the appropriate program on it.
    - For **_calibration_** data: fill in “DATE OBS” (check both the night of and the morning after), “DPR CATG” to CALIB, “INS WLEN ID” to grating setting, and “Program ID” to 60.A-9051(A). Check dates on wavelength cal files to make sure we’re not using calibrations widely separated in time with some coming from the automatic association, and some coming from the manual calibrator query.
3. Download shell script/s, scp onto desired folder, `chmod +x <filename>`
4. Run shell script as `./<filename> -u <username> -p`
    - This ensures that you’re able to download the proprietary data
    - If you get a “bad substitution error”, check that the shell scripts have `“#!/bin/bash”` at the top and not `#!/bin/sh”`
5. Uncompress fits files with `gunzip -d *.fits.Z`
    - The d is ‘decompress’


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
