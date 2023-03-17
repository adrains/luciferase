"""Module containing functions involved with preparing simulated exoplanet
transits from model stellar, telluric, and planetary spectra.
"""
import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Load spectra
# -----------------------------------------------------------------------------
def read_marcs_spectrum(filepath, n_max_n_mu=55, header_length=36):
    """Reads in a MARCS .itf file to extract wavelengths, weights, mu values,
    and intensities. 
    
    Note that the amount of mu points sampled varies for each wavelength point
    and ranges from 16 through 55. For speed we pre-allocate our mu and
    intensity arrays with shape [n_wave, n_max_n_mu], meaning that for any
    wavelength with < 55 samples the array will be 'padded' with nans.

    Parameters
    ----------
    filepath: str
        Filepath to MARCS .itf file.

    n_max_n_mu: int, default: 55
        The maximum number of mu samples across the disc we expect.

    header_length: int, default:36
        The number of characters in a header row.

    Returns
    -------
    wavelengths: 1D float array
        Array of wavelengths of shape [n_wave].
    
    weights: 1D float array
        Array of weights of shape [n_wave].
    
    n_samples: 1D float array
        Array of n_mu_samples of shape [n_wave].
    
    mus: 2D float array
        Array of mus of shape [n_wave, n_max_n_mu].
    
    intensities: 2D float array
        Array of intensities of shape [n_wave, n_max_n_mu].
    """
    # Initialise booleans
    have_read_header = False
    have_read_mus = False
    have_read_intensities = False

    with open(filepath) as marcs_itf:
        # Read in all lines
        all_lines = marcs_itf.readlines()

        # Count the number of wavelengths so we can preallocate an array
        line_lengths = np.array([len(row) for row in all_lines])
        n_wave = np.sum(line_lengths == header_length)

        # Preallocate arrays
        wavelengths = np.zeros(n_wave)
        weights = np.zeros(n_wave)
        n_samples = np.zeros(n_wave)

        # Finally, initialise per-wavelength vectors
        mus = np.full((n_wave, n_max_n_mu), np.nan)
        intensities = np.full((n_wave, n_max_n_mu), np.nan)

        wave_i = 0
        mu_min = 0
        intens_min = 0

        for line_str in tqdm(all_lines, desc="Reading File", leave=False):
            # Split on whitespace
            line_values = line_str.split()

            # -----------------------------------------------------------------
            # Parse line info
            # -----------------------------------------------------------------
            # Read header
            if not have_read_header:
                assert len(line_values) == 3

                wavelengths[wave_i] = float(line_values[0])
                weights[wave_i] = float(line_values[1])

                n_samp = int(line_values[2])
                n_samples[wave_i] = n_samp

            # Read mu values
            elif not have_read_mus:
                # Update mu values
                mu_max = mu_min + len(line_values)

                mus[wave_i, mu_min:mu_max] = \
                    np.array(line_values).astype(float)

                # Update min
                mu_min = mu_max

            # Read intensity values
            elif not have_read_intensities:
                # Update intensity values
                intens_max = intens_min + len(line_values)

                intensities[wave_i, intens_min:intens_max] = \
                    np.array(line_values).astype(float)

                # Update min
                intens_min = intens_max

            # -----------------------------------------------------------------
            # Checks to change state (i.e. update booleans and counters)
            # -----------------------------------------------------------------
            # Have read header
            if not have_read_header:
                have_read_header = True

            # Have read header and finished reading mus
            elif have_read_header and not have_read_mus and mu_min == n_samp:
                have_read_mus = True
            
            # Have read header, mus, and intensities
            elif have_read_header and have_read_mus and intens_min == n_samp:
                # Reset counters
                mu_min = 0
                intens_min = 0

                # Reset booleans
                have_read_header = False
                have_read_mus = False
                have_read_intensities = False

                # Index wavelength
                wave_i += 1

    return wavelengths, weights, n_samples, mus, intensities


def load_stellar_spectrum():
    """
    """
    pass

def load_planet_spectrum():
    """
    """
    pass

def load_telluric_spectrum():
    """
    """
    pass

# -----------------------------------------------------------------------------
# Interpolate spectra
# -----------------------------------------------------------------------------
def interpolate_stellar_spectrum():
    """
    """
    pass

def interpolate_telluric_spectrum():
    """
    """
    pass

def interpolate_planet_spectrum():
    """
    """
    pass

# -----------------------------------------------------------------------------
# Other
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Simulate
# -----------------------------------------------------------------------------
def simulate_transit_single_epoch():
    """
    """
    pass

    # Interpolate stellar spectrum

    # Interpolate telluric spectrum

    # Interpolate planet spectrum (if in phase)

    # Combine components

    # Apply instrumental transfer function

    # Save result



def simulate_transit_multiple_epochs():
    """
    """
    pass

def apply_instrumental_transfer_function():
    """
    """
    pass