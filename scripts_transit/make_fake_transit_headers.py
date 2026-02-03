"""Script to produce fake header/timestep info for unobserved transits, with
the goal of using these to simulate said transits with
scripts_transits/simulate_transit.py.
"""
import os
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import matplotlib.pyplot as plt
import transit.utils as tu

# Setup
observatory = "Paranal"
save_path = "simulations/simulated_transit_df/"

# Transit midpoints (sourced from NASA transit prediction service)
#-----------------------------------------
# WASP-107b
#-----------------------------------------
#target_id = "WASP-107"
#n_phases = 52
#planet_properties_file = "scripts_transit/planet_data_wasp107.csv"

#exp_sec = 301.4352425

# ----
# 2022
# ----
#transit_midpoint_jd = 2459649.78761 # UT11/03/2022 06:54, bcor = 11 km/s (obs)
#transit_midpoint_jd = 2459672.67357 # UT03/04/2022 04:10, bcor = 0 km/s
#transit_midpoint_jd = 2459758.49591 # UT27/06/2022 23:5, bcor = -29 km/s

# ----
# 2023
# ----
#transit_midpoint_jd = 2459998.79846 # UT23/02/2023 07:10, bcor = 18 km/s (obs)
#transit_midpoint_jd = 2460021.68441 # UT18/03/2023 04:26, bcor = 8 km/s
#transit_midpoint_jd = 2460084.62080 # UT20/05/2023 02:54, bcor = -21 km/s
#transit_midpoint_jd = 2460107.50675 # UT12/06/2023 00:10, bcor = -27 km/s

# ----
# 2024
# ----
#transit_midpoint_jd = 2460347.80930 # UT07/02/2024 07:25, bcor = 25 km/s
#transit_midpoint_jd = 2460433.63164 # UT03/05/2024 03:10, bcor = -15 km/s
transit_midpoint_jd = 2460456.51760 # UT26/25/2024 00:25, bcor = -23 km/s

#-----------------------------------------
# GJ 1214
#-----------------------------------------
target_id = "GJ1214"
n_phases = 80
planet_properties_file = "scripts_transit/planet_data_GJ1214.csv"

exp_sec = 120

# 220311 -- Mahajan et al. 2024
#transit_midpoint_jd = 2459650.84409362

# 220330 -- Mahajan et al. 2024
#transit_midpoint_jd = 2459669.80894799

# 220703 -- Mahajan et al. 2024
#transit_midpoint_jd = 2459764.63321985

# 220810 -- Mahajan et al. 2024
#transit_midpoint_jd = 2459802.56292859

# 230622 -- Mahajan et al. 2024
#transit_midpoint_jd = 2460118.64382557

# 230810 -- Schlawin et al. 2024
#transit_midpoint_jd = 2460167.63636015

# 240806 -- Mahajan et al. 2024
#transit_midpoint_jd = 2460529.54901285

# 240817 -- Mahajan et al. 2024
transit_midpoint_jd = 2460540.61184457

#------------------------------------------------------------------------------
# Construct time mid-points
#------------------------------------------------------------------------------
exp_day = exp_sec / 24 / 3600

# These are the overheads between exposures when repeating or changing between
# nodding positions, assuming a pattern of ABBA. The current values were
# eyeballed from the existing WASP-107b observations.
dt_repeat = 69  # sec
dt_change = 84  # sec

times_mid = np.zeros(n_phases)
nod_positions = np.full(n_phases, "")

# Loop over all phases in a ABBA nodding pattern and assign exposure mid-points
for phase_i in range(n_phases):
    # If phase_i = 0, default start
    if  phase_i == 0:
        times_mid[0] = 0
        nod_positions[0] = "A"
        continue
    
    # If the last index was an 'A'
    if nod_positions[phase_i-1] == "A":
        # If we're on the second index, go B
        if phase_i == 1:
            times_mid[phase_i] = times_mid[phase_i-1] + exp_sec + dt_change
            nod_positions[phase_i] = "B"

        # If we've had two A's in a row, change to B
        elif nod_positions[phase_i-2] == "A":
            times_mid[phase_i] = times_mid[phase_i-1] + exp_sec + dt_change
            nod_positions[phase_i] = "B"

        # Otherwise repeat 'A'
        else:
            times_mid[phase_i] = times_mid[phase_i-1] + exp_sec + dt_repeat
            nod_positions[phase_i] = "A"

    # If the last index was an 'B'
    if nod_positions[phase_i-1] == "B":
        # If we've had two B's in a row, change to A
        if nod_positions[phase_i-2] == "B":
            times_mid[phase_i] = times_mid[phase_i-1] + exp_sec + dt_change
            nod_positions[phase_i] = "A"

        # Otherwise repeat B
        else:
            times_mid[phase_i] = times_mid[phase_i-1] + exp_sec + dt_repeat
            nod_positions[phase_i] = "B"

# Convert to JD from seconds
times_day = times_mid / 24 / 3600
times_day -= np.median(times_day)
times_jd = times_day + transit_midpoint_jd

# Finally, add an offset of ~1 exposure from the transit midpoint
times_jd += np.random.normal(loc=0, scale=exp_day)

#------------------------------------------------------------------------------
# Airmasses
#------------------------------------------------------------------------------
# Compute airmasses for each exposure
target = SkyCoord.from_name(target_id)
jds = Time(times_jd, format="jd",)
loc = EarthLocation.of_site(observatory)

frame = AltAz(obstime=jds, location=loc)
airmasses = target.transform_to(frame).secz

# Get date representation of mid-point
date = Time(transit_midpoint_jd, format="jd",).to_value("iso").split(" ")[0]

#------------------------------------------------------------------------------
# Barycentric velocities
#------------------------------------------------------------------------------
# Compute barycentric velocities for each exposure
barycorr = target.radial_velocity_correction(obstime=jds, location=loc)
bcor = barycorr.to(u.km/u.s).value

hel_corr = target.radial_velocity_correction(
    kind="heliocentric", obstime=jds, location=loc,)
hel_corr = hel_corr.to(u.km/u.s)

#------------------------------------------------------------------------------
# Setup DataFrame
#------------------------------------------------------------------------------
# Create the DataFrame as we do for real observations. All the initial values
# (mostly) come directly from the headers, and once we've constructed the
# DataFrame we can pass it to tu.calculate_transit_timestep_info to calculate
# timestep information about the planet during transit.
df_cols = ["mjd_start", "mjd_mid", "mjd_end", "jd_start", "jd_mid", 
    "jd_end", "airmass", "bcor", "hcor", "ra", "dec", "exptime_sec", 
    "nod_pos", "raw_file", "obs_temp", "rel_humidity", "wind_dir",
    "wind_speed", "seeing_start", "seeing_end",]

transit_df = pd.DataFrame(
    data=np.full((n_phases, len(df_cols)), np.nan),
    columns=df_cols,)

transit_df["mjd_start"] = times_jd - 2400000.5 - exp_day/2
transit_df["mjd_mid"] = times_jd - 2400000.5
transit_df["mjd_end"] = times_jd - 2400000.5 + exp_day/2

transit_df["jd_start"] = times_jd - exp_day/2
transit_df["jd_mid"] = times_jd
transit_df["jd_end"] = times_jd + exp_day/2

transit_df["airmass"] = airmasses
transit_df["bcor"] = bcor
transit_df["hcor"] = hel_corr
transit_df["ra"] = target.ra.value
transit_df["dec"] = target.dec.value

transit_df["exptime_sec"] = exp_sec
transit_df["nod_pos"] = nod_positions

# All this is filler
transit_df["raw_file"] = ""
transit_df["obs_temp"] = np.nan
transit_df["rel_humidity"] = np.nan
transit_df["wind_dir"] = np.nan
transit_df["wind_speed"] = np.nan
transit_df["seeing_start"] = np.nan
transit_df["seeing_end"] = np.nan

syst_info = tu.load_planet_properties(planet_properties_file)

tu.calculate_transit_timestep_info(
    transit_info=transit_df,
    syst_info=syst_info,)

#------------------------------------------------------------------------------
# Diagnostics
#------------------------------------------------------------------------------
# Diagnostic plots of airmass, bcor, and planet xyz positions and velocities
# mostly used for sanity checking that this would be a reasonable transit to
# actually observe.
plt.close("all")
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(6,8))

# Airmasses
axes[0,0].plot(times_jd, airmasses, ".-")
axes[0,0].vlines(
    transit_midpoint_jd,
    ymin=np.min(airmasses),
    ymax=np.max(airmasses),
    linestyles="dashed",
    linewidth=0.5,
    colors="r",)
axes[0,0].set_xlabel("JD")
axes[0,0].set_ylabel("Airmass")

# Barycentric velocity
axes[0,1].plot(times_jd, transit_df["bcor"], ".-")
axes[0,1].vlines(
    transit_midpoint_jd,
    ymin=np.min(bcor),
    ymax=np.max(bcor),
    linestyles="dashed",
    linewidth=0.5,
    colors="r",)
axes[0,1].set_xlabel("JD")
axes[0,1].set_ylabel("bcor (km/s)")

# Position (X)
axes[1,0].plot(times_jd, transit_df["r_x_mid"], ".-")
axes[1,0].vlines(
    transit_midpoint_jd,
    ymin=np.min(transit_df["r_x_mid"]),
    ymax=np.max(transit_df["r_x_mid"]),
    linestyles="dashed",
    linewidth=0.5,
    colors="r",)
axes[1,0].set_xlabel("JD")
axes[1,0].set_ylabel(r"$r_x$ (km/s)")

# Position (Y)
axes[2,0].plot(times_jd, transit_df["r_y_mid"], ".-")
axes[2,0].vlines(
    transit_midpoint_jd,
    ymin=np.min(transit_df["r_y_mid"]),
    ymax=np.max(transit_df["r_y_mid"]),
    linestyles="dashed",
    linewidth=0.5,
    colors="r",)
axes[2,0].set_xlabel("JD")
axes[2,0].set_ylabel(r"$r_y$ (km/s)")

# Position (Z)
axes[3,0].plot(times_jd, transit_df["r_z_mid"], ".-")
axes[3,0].vlines(
    transit_midpoint_jd,
    ymin=np.min(transit_df["r_z_mid"]),
    ymax=np.max(transit_df["r_z_mid"]),
    linestyles="dashed",
    linewidth=0.5,
    colors="r",)
axes[3,0].set_xlabel("JD")
axes[3,0].set_ylabel(r"$r_z$ (km/s)")

# Velocity (X)
axes[1,1].plot(times_jd, transit_df["v_x_mid"], ".-")
axes[1,1].vlines(
    transit_midpoint_jd,
    ymin=np.min(transit_df["v_x_mid"]),
    ymax=np.max(transit_df["v_x_mid"]),
    linestyles="dashed",
    linewidth=0.5,
    colors="r",)
axes[1,1].set_xlabel("JD")
axes[1,1].set_ylabel(r"$v_x$ (km/s)")

# Velocity (Y)
axes[2,1].plot(times_jd, transit_df["v_y_mid"], ".-")
axes[2,1].vlines(
    transit_midpoint_jd,
    ymin=np.min(transit_df["v_y_mid"]),
    ymax=np.max(transit_df["v_y_mid"]),
    linestyles="dashed",
    linewidth=0.5,
    colors="r",)
axes[2,1].set_xlabel("JD")
axes[2,1].set_ylabel(r"$v_y$ (km/s)")

# Velocity (Z)
axes[3,1].plot(times_jd, transit_df["v_z_mid"], ".-")
axes[3,1].vlines(
    transit_midpoint_jd,
    ymin=np.min(transit_df["v_z_mid"]),
    ymax=np.max(transit_df["v_z_mid"]),
    linestyles="dashed",
    linewidth=0.5,
    colors="r",)
axes[3,1].set_xlabel("JD")
axes[3,1].set_ylabel(r"$v_z$ (km/s)")

plt.tight_layout()

plt.savefig(
    os.path.join(save_path, "{}_{}.png".format(target_id, date)), dpi=200)

#------------------------------------------------------------------------------
# Dump to file
#------------------------------------------------------------------------------
fn = "transit_info_df_{}_{}.csv".format(target_id, date)
path = os.path.join(save_path, fn)
transit_df.to_csv(path, index=False)