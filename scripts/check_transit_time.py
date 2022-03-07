"""
Script to project transit times into the future for planning.
"""
import numpy as np
import datetime
from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RECT_HEIGHT = 2
N_TRANS = 3

# Planet details
star = "LTT 1445 A b"

if star == "LTT 1445 A b":
    params = ["winters+21", "winters+19", "rains+21"]
    period = np.array([5.3587657, 5.35882, 5.35880])
    e_period = np.array([0.0000043, 0.00030, 0.0000075]) / 24
    t_dur = np.array([1.367, 1.38, 1.3698]) / 24
    tc = np.array([2458412.70851, 2458423.42629, 2458412.70874])

# Observation details
obs_start = "2021-10-29T01:26:02.456"
obs_end = "2021-10-29T04:33:47.154"

# Convert
obs_start_jd = Time.strptime(obs_start, "%Y-%m-%dT%H:%M:%S.%f").jd
obs_end_jd = Time.strptime(obs_end, "%Y-%m-%dT%H:%M:%S.%f").jd

# Now plot
plt.close("all")
fig, axis = plt.subplots()
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
colour_i = 0


# Plot what we observed
rect_obs = Rectangle(
    xy=(obs_start_jd, 0),
    width=obs_end_jd-obs_start_jd,
    height=RECT_HEIGHT*len(params),
    linewidth=0.1,
    color=colours[colour_i],
    alpha=0.6,
    label="CRIRES+ Observation")

axis.add_patch(rect_obs)
colour_i += 1

for param_i in range(len(params)):
    # Calculate the transits far into the future
    tc_future = np.arange(
        tc[param_i], obs_end_jd+period[param_i], period[param_i])

    trans_start = tc_future[-2] - t_dur[param_i] / 2

    # First plot the actual transit
    rect_trans = Rectangle(
        xy=(trans_start, param_i*2),
        width=t_dur[param_i],
        height=RECT_HEIGHT,
        linewidth=0.1,
        color=colours[colour_i],
        alpha=0.6,
        label="Transit ({})".format(params[param_i]))
    
    axis.add_patch(rect_trans)

    # Then plot the transit if we have the period 1 sigma wrong
    period_pm_1sigma = [
        period[param_i] - e_period[param_i],
        period[param_i] + e_period[param_i],]

    for period_i, period_1sigma in enumerate(period_pm_1sigma):
        # Calculate the transits far into the future
        tc_future = np.arange(
            tc[param_i], obs_end_jd+period_1sigma, period_1sigma)

        trans_start = tc_future[-2] - t_dur[param_i] / 2

        if period_i == 0:
            label = r"Transit ({}) $-1\sigma$".format(params[param_i])
            hatch = "/"
        else:
            label = r"Transit ({}) $+1\sigma$".format(params[param_i])
            hatch = "\\"

        rect_trans = Rectangle(
            xy=(trans_start, param_i*2),
            width=t_dur[param_i],
            height=RECT_HEIGHT,
            linewidth=0.1,
            fill=False,
            color=colours[colour_i],
            alpha=0.5,
            hatch=hatch,
            label=label)
        
        axis.add_patch(rect_trans)

    colour_i += 1

axis.set_xlabel("Time (JD)")
axis.set_xlim(obs_start_jd-t_dur[0]*1, obs_end_jd+t_dur[0]*1)
axis.set_ylim(-1, len(params)*2+1)
axis.set_yticks([])
axis.legend(loc="best", fontsize="x-small")
plt.show()

