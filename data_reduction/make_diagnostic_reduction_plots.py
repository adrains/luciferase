"""Script to plot a diagnostic 
"""
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, "/home/arains/code/luciferase/")

import luciferase.spectra as lspec
import luciferase.plotting as lplt

cwd = os.getcwd()

# This is the default list of files within the provided directory to work with
EXTRACTED_FILES = [
    "cr2res_obs_nodding_extractedA.fits",
    "cr2res_obs_nodding_extractedB.fits",
    "cr2res_obs_nodding_extracted_combined.fits",
]

EXTRACTED_IMAGES = [
    "cr2res_obs_nodding_combinedA.fits",
    "cr2res_obs_nodding_combinedB.fits"
]

TRACE_WAVES = [
    "cr2res_obs_nodding_trace_wave_A.fits",
    "cr2res_obs_nodding_trace_wave_B.fits"
]

# Double check we haven't been given too many inputs
if len(sys.argv) > 2:
    exception_text = (
        "Warning, too many inputs ({})! "
        "Make sure to wrap paths with wildcard characters in quotes.")

    raise Exception(exception_text.format(len(sys.argv)))

# Take as input the folder/s to consider
data_dir = sys.argv[1]

# And figure out just how many folder's we're working with
folders = glob.glob(data_dir)
folders.sort()

print("Found {:0.0f} folders to create diagnostic plots for.".format(
    len(folders)))

# Loop over the contents of each folder
for folder in folders:
    # Check that all our files actually exist
    missing_files = []
    for file_set in [EXTRACTED_FILES, EXTRACTED_IMAGES, TRACE_WAVES]:
        for file in file_set:
            filepath = os.path.join(folder, file)
            if not os.path.isfile(filepath):
                missing_files.append(filepath)

    if len(missing_files) > 0:
        print("Skipping {} due to missing files: {}".format(
            folder, missing_files))
        continue

    # All files accounted for, load in spectra
    observations = []

    for fits_file in EXTRACTED_FILES:
        file_path = os.path.join(folder, fits_file)
        observations.append(
            lspec.initialise_observation_from_crires_nodding_fits(file_path))

    # Check how many orders there are
    n_orders = observations[0].n_orders
    n_detectors = observations[0].n_detectors

    # Format of plot will be the first two rows being the A and B images + TW
    # Then every subsequent row will be a given order.

    # Number of rows is 2 + number of orders, one column for each detector
    n_col = 3
    n_rows = 2 + n_orders

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_col,
        figsize=(10, 3*n_rows+0.5),
    )

    # Adjust bounds
    #fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1,)

    # Plot A frame tracewave
    lplt.plot_trace_vs_reduced_frame(
        fits_trace=os.path.join(folder, TRACE_WAVES[0]),
        fits_image=os.path.join(folder, EXTRACTED_IMAGES[0]),
        fig=fig,
        axes=axes[0],
    )

    # Plot B frame tracewave
    lplt.plot_trace_vs_reduced_frame(
        fits_trace=os.path.join(folder, TRACE_WAVES[1]),
        fits_image=os.path.join(folder, EXTRACTED_IMAGES[1]),
        fig=fig,
        axes=axes[1],
    )

    labels = ["A", "B", "Combined",]
    colours = ["r", "g", "b",]

    spec_i = 0

    # Plot each order
    for order_i in range(n_orders):
        for detector_i in range(3):
            for obs_i, (obs, label) in enumerate(zip(observations, labels)):
                axes[2+order_i, detector_i].plot(
                    obs.spectra_1d[spec_i].wave,
                    obs.spectra_1d[spec_i].flux / np.nanmedian(obs.spectra_1d[spec_i].flux),
                    linewidth=0.2,
                    label=label,
                    color=colours[obs_i],
                    alpha=0.8,)
            
            # Label Y axes and ticks only on left
            if detector_i == 0:
                axes[2+order_i, detector_i].set_ylabel("Flux")
            else:
                axes[2+order_i, detector_i].set_yticks([])
            
            # And X axes only on bottom
            if order_i == n_orders-1:
                axes[2+order_i, detector_i].set_xlabel(r"Wavelength ($\mu$m)")
            
            # Set tick font size
            axes[2+order_i, detector_i].tick_params(
                axis='y', which='major', labelsize="xx-small")

            axes[2+order_i, detector_i].tick_params(
                axis='x', which='major', labelsize="xx-small")

            # Add legend
            leg = axes[2+order_i, detector_i].legend(
                #by_label.values(),
                #by_label.keys(),
                fontsize="xx-small",
                loc="upper center",
                ncol=3,)

            for legobj in leg.legendHandles:
                legobj.set_linewidth(1.5)
            
            # Add text
            wave = obs.spectra_1d[spec_i].wave
            w_mid = (wave[-1] - wave[0])/2 + wave[0]

            axes[2+order_i, detector_i].text(
                x=w_mid,
                y=0,
                s="Order: {:0.0f}, Detector: {:0.0f}".format(order_i, detector_i+1),
                horizontalalignment="center",
                fontsize="xx-small",
            )

            spec_i += 1

            axes[2+order_i, detector_i].set_ylim(-0.1, 2)

    # Save plot
    plt.suptitle(os.path.join(cwd, folder))
    #plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(folder, "reduction_diagnostic.pdf"))




