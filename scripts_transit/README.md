## Exoplanet Transmission Spectroscopy Simulation & Analysis Code

### Data Cube Preparation
`prepare_transit_model_fits.py` prepares a cleaned .fits data cube of shape [n_transit, n_phase, n_spectrum, n_px] on a uniform wavelength grid given a set of reduced CRIRES+ transits. This requires a CSV file containing information about the planet, but otherwise runs entirely from the reduced CRIRES+ data.

### Simulated Transmission Spectroscopy 'Observations'
`simulate_transit.py` generates synthetic transit observations using the 'header' information (e.g. RA, DEC, observatory, exposure, barcyentric velocity) from an observed transit and known planet properties (e.g. orbital elements, radius, mass) to generate synthetic observations given a set of stellar spectrum, planet spectrum, and telluric transmission files. The details of this simulation are controlled from `simulation_settings.yml`.

### An Inverse Method to Recover Planet Transmission Spectra
`run_transit_model.py` runs the inverse modelling approach originally published in [Aronson+2015](https://ui.adsabs.harvard.edu/abs/2015A%26A...578A.133A/abstract) on a pre-prepared .fits datacube--either real or simulated--to produce model stellar flux, telluric tau, exoplanet transmission, and slit-loss vectors. The details of this modelling are controlled from `transit_model_settings.yml`.

- - - -

### Steps for Running the Inverse Method
1. Reduce data using CRIRES+ DRS & blaze correct the results using scripts in `scripts_reduction/`.
2. Use `prepare_spectra.py` on the master A or B spectrum (**_not_** combined due to wavelength offsets between the two) to:
   
   i. Iterate for best-fit Molecfit telluric transmission **_and_** corrected stellar spectrum--these will both be used as the initial guess for the inverse method.
   
   ii. Produce and save optimised continuum polynomial coefficients for each order/detector--these will be used for continuum normalisation prior to SYSREM being run.
   
4. Prepare a cleaned and re-gridded data cube via `prepare_transit_model_fits.py`.
5. Run the inverse model via `run_transit_model.py`.
6. Run SYSREM for comparison `run_sysrem.py`.
7. Cross correlate and compare results.
