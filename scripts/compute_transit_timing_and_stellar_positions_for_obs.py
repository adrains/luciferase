"""Script to assign transit times and positions on the stellar disc for time
series observations taken during transit. Note that this assumes a circular
orbit.
"""
import os
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

# Definte star, setup filepath
cwd = os.getcwd()

star = "LTT 1445 A b"

csv_fn = os.path.join(
    cwd, "results", "mu_in_transit_{}.tsv".format(star.replace(" ", "_")))

# Define transit and stellar properties from the literature
if star == "LTT 1445 A b":
    source = "winters+21"
    period = 5.3587657
    e_period = 0.0000043 / 24
    t_dur = 1.367 / 24
    tc = 2458412.70851
    b_impact = 0.17 #+0.15 -0.12

# Get our list of observations and sort
dir = "/home/alexis/GTO/211028_LTTS1445Ab/CRIRE.2021-10-29T*.fits"
fits_filenames = glob.glob(dir)
fits_filenames.sort()

# Initialise our output dataframe
cols = ["fn", "exp_time_days", "obs_start_jd", "obs_mid_jd", "obs_end_jd",
        "during_transit", "exp_during_transit_frac", "trans_complete_frac", 
        "x", "y", "mu",]

obs_df = pd.DataFrame(
    data=np.full((len(fits_filenames), len(cols)), np.nan), 
    index=np.arange(len(fits_filenames)), 
    columns=cols)

# For each exposure, compute the start, mid, and end times
for obs_i, filename in enumerate(fits_filenames):
    with fits.open(filename) as fits_file:
        # Import exposure time and convert to days
        exp_time_days = fits_file[0].header["EXPTIME"] / 3600 / 24

        # Import observation time and convert to JD.
        # Note: probably easier to just use MJD header keyword, but upon
        # checking these are consistent to < 1 msec.
        obs_start_jd = Time.strptime(
            fits_file[0].header["DATE-OBS"], "%Y-%m-%dT%H:%M:%S.%f").jd

        #obs_start_jd = fits_file[0].header["MJD-OBS"] + 2400000.5

        # Calculate centre and end point
        obs_mid_jd = obs_start_jd + exp_time_days/2
        obs_end_jd = obs_start_jd + exp_time_days

        # Store
        obs_df.loc[obs_i, "fn"] = filename
        obs_df.loc[obs_i, "exp_time_days"] = exp_time_days
        obs_df.loc[obs_i, "obs_start_jd"] = obs_start_jd
        obs_df.loc[obs_i, "obs_mid_jd"] = obs_mid_jd
        obs_df.loc[obs_i, "obs_end_jd"] = obs_end_jd


# Calculate the times the transit begins and ends by predicting transits into
# future. We want the second last prediction.
tc_future = np.arange(tc, obs_end_jd+period, period)

t_start = tc_future[-2] - t_dur / 2
t_end = tc_future[-2] + t_dur / 2

# Y is always equal to the impact parameter under the assumption the path
# the planet takes across the stellar disk is linear. This should be true
# where a >> R_star
y = b_impact

# Calculate the total projected horizontal distance travelled by the planet
# This is simply Pythagoras for a triangle with side R=1, height=b, and x=d/2,
# where d is the total projected distance travelled by the planet across the
# star.
d_total = 2 * np.sqrt(1**2 - y**2)

# Loop over all timesteps and calculate x, the fraction of the exposure during
# transit, fractional completion of the transit at the centre-point, and mu.
# Note that we're working in normalised units here, so we're working in units 
# of R_star, thus R = 1.
for obs_i in range(len(fits_filenames)):
    # Only meaningful to compute if we're at least partially during the transit
    if (obs_df.loc[obs_i, "obs_end_jd"] < t_start 
        or obs_df.loc[obs_i, "obs_start_jd"] > t_end):
        obs_df.loc[obs_i, "during_transit"] = False
        obs_df.loc[obs_i, "exp_during_transit_frac"] = 0
        obs_df.loc[obs_i, "trans_complete_frac"] = 0
        continue

    # We're not completely on the star yet
    elif obs_df.loc[obs_i, "obs_start_jd"] < t_start:
        exp_frac = ((obs_df.loc[obs_i, "obs_end_jd"] - t_start) 
                   / obs_df.loc[obs_i, "exp_time_days"])

    # We're moving off the star
    elif obs_df.loc[obs_i, "obs_end_jd"] > t_end:
        exp_frac = ((t_end - obs_df.loc[obs_i, "obs_start_jd"]) 
                   / obs_df.loc[obs_i, "exp_time_days"])

    # Otherwise completely on top of star
    else:
        exp_frac = 1
    
    obs_df.loc[obs_i, "exp_during_transit_frac"] = exp_frac
    obs_df.loc[obs_i, "during_transit"] = True

    # Calculate x by first calculating the fraction of the transit completed so
    # far, then multiplying by d_tot. Make sure to watch the sign.
    trans_frac = (obs_df.loc[obs_i, "obs_mid_jd"] - t_start) / t_dur
    d_current = trans_frac * d_total

    # If over halfway
    if d_current > (d_total / 2):
        x = d_current - d_total / 2

    # Otherwise
    else:
        x = (d_total / 2) - d_current

    # Calculate mu. See Mandel & Agol 2001:
    # https://ui.adsabs.harvard.edu/abs/2002ApJ...580L.171M/abstract
    mu = np.sqrt(1 - x**2 - y**2)
    
    print_str = ("i={}\ttrans_frac={:0.2f}%\texp_dur_frac={:0.2f}\t"
                 "x={:0.2f}\tmu={:0.2f}")
    print(print_str.format(obs_i, trans_frac*100, exp_frac, x, mu))

    # Save
    obs_df.loc[obs_i, "trans_frac"] = trans_frac
    obs_df.loc[obs_i, "x"] = x
    obs_df.loc[obs_i, "y"] = y
    obs_df.loc[obs_i, "mu"] = mu

# Save DataFrame
obs_df.to_csv(csv_fn, sep="\t")