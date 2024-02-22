# Exoplanet Transmission Spectroscopy Simulation & Analysis Code


## Scripts
### Data Cube Preparation
`prepare_transit_model_fits.py` prepares a cleaned .fits data cube of shape [n_transit, n_phase, n_spectrum, n_px] on a uniform wavelength grid given a set of reduced CRIRES+ transits. This requires a CSV file containing information about the planet, but otherwise runs entirely from the reduced CRIRES+ data.

### Planet Template Preparation
`make_planet_transmission_grid_fits.py` combines separate planet atmosphere templates modelled (i.e. with petitRADTRANS) with different molecular species into a single fits datacube. The datacube has three HDUs ('WAVE', 'SPEC', and 'TEMPLATE_INFO') and this datacube is the expected format of planet spectra expected by the simulator.

### Simulated Transmission Spectroscopy 'Observations'
`simulate_transit.py` generates synthetic transit observations using the 'header' information (e.g. RA, DEC, observatory, exposure, barcyentric velocity) from an observed transit and known planet properties (e.g. orbital elements, radius, mass) to generate synthetic observations given a set of stellar spectrum, planet spectrum, and telluric transmission files. The details of this simulation are controlled from `simulation_settings.yml`.

### An Inverse Method to Recover Planet Transmission Spectra
`run_transit_model.py` runs the inverse modelling approach originally published in [Aronson+2015](https://ui.adsabs.harvard.edu/abs/2015A%26A...578A.133A/abstract) on a pre-prepared .fits datacube--either real or simulated--to produce model stellar flux, telluric tau, exoplanet transmission, and slit-loss vectors. The details of this modelling are controlled from `transit_model_settings.yml`.

### SYSREM
`run_sysrem.py` runs [SYSREM](https://ui.adsabs.harvard.edu/abs/2005MNRAS.356.1466T/abstract), the iterative detrending lightcurve detrending algorithm the de facto standard method for recovery of exoplanet transmission or emission spectra in the IR. This script takes as input a data cube in the same format as expected by the inverse method (and output by `prepare_transit_model_fits.py` and `run_transit_model.py`). The parameters of the SYSREM detrending are controlled from `sysrem_settings.yml`.

- - - -
## Settings
The following are the YAML settings files associated with the scripts above:
- `simulation_settings.yml`
- `sysrem_settings.yml`
- `transit_model_settings.yml`

And they should be modified rather than the scripts themselves.

- - - -

### Analysis
1. Reduce data using CRIRES+ DRS & blaze correct the results using scripts in `scripts_reduction/`.
2. Iterate for the best-fit Molecfit telluric transmission **_and_** corrected stellar spectrum--these will both be used as the initial guess for the inverse method.
   - Use `scripts_molecfuit/prepare_molecfit_files_initial.py` to produce the initial set of Molecfit input files.
   - Run Molecfit to obtain an initial best-fit telluric model.
   - Use `scripts_molecfuit/prepare_molecfit_files_optimised.py` to optimise the continuum placement prior to running Molecfit a second time.
   - Run Molecfit to obtain an optimised best-fit telluric model.
6. Prepare a cleaned and re-gridded data cube via `prepare_transit_model_fits.py`.
7. Run the inverse model via `run_transit_model.py`.
8. Run SYSREM for comparison `run_sysrem.py`.
9. Cross correlate and compare results.
