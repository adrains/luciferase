"""Script to plot a diagnostic plot to easily inspect how a CRIRES reduction
went. The diagnostic plot consists of the trace waves from the A and B frames
overplotted on their respective images; as well as the overplotted A, B, and
combined spectra for each detector/order. This is saved as a single PDF in the
same directory as the reduced data. This script can be called on multiple 
directories at once using wildcard characters, and if so it will attempt to
stitch each individual PDF together into a single master PDF for ease of
inspection.

Run as
------
python make_diagnostic_reduction_plots.py [data_path]

where [data_path] is the path to the data (which can include wildcards for
globbing). Make sure to wrap any wildcard filepaths in quotes to prevent 
globbing happening on the command line (versus in python). Note that this 
script requires objects and functions from luciferase to run, so the 
appropriate path should be inserted as below.
"""
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, "/home/arains/code/luciferase/")

import luciferase.spectra as lspec
import luciferase.plotting as lplt

try:
    from PyPDF2 import PdfFileMerger
    do_pdf_merge = True
except:
    print("PyPDF2 is not installed, cannot merge PDFs\n")
    do_pdf_merge = False

cwd = os.getcwd()

# Aspect ratio for imshow
ASPECT = "auto"

# Y Bounds to plot
Y_BOUND_LOW = -0.1
Y_BOUND_HIGH = 2.0

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

# Make a list of all newly created PDFs
all_diagnostic_pdfs = []

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

    print("Creating diagnostic plot for: {}".format(folder))

    # All files accounted for, load in spectra
    observations = []

    for fits_file in EXTRACTED_FILES:
        file_path = os.path.join(folder, fits_file)
        observations.append(
            lspec.initialise_observation_from_crires_nodding_fits(file_path))

    # Get information about orders and detectors
    n_orders = observations[0].n_orders
    min_order =  observations[0].min_order
    max_order =  observations[0].max_order
    n_detectors = observations[0].n_detectors

    # Format of plot will be the first two rows being the A and B images + TW
    # Then every subsequent row will be a given order.

    # Number of rows is 2 + number of orders, one column for each detector
    n_col = 3
    n_rows = 2 + n_orders

    plt.close("all")
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_col,
        figsize=(10, 3*n_rows+0.5),)

    # Adjust bounds
    fig.subplots_adjust(
        left=0.05, bottom=0.025, right=0.95, top=0.95, wspace=0.05,)

    # Plot A frame tracewave
    lplt.plot_trace_vs_reduced_frame(
        fits_trace=os.path.join(folder, TRACE_WAVES[0]),
        fits_image=os.path.join(folder, EXTRACTED_IMAGES[0]),
        fig=fig,
        axes=axes[0],
        aspect=ASPECT,
        plot_title=True,)

    # Plot B frame tracewave
    lplt.plot_trace_vs_reduced_frame(
        fits_trace=os.path.join(folder, TRACE_WAVES[1]),
        fits_image=os.path.join(folder, EXTRACTED_IMAGES[1]),
        fig=fig,
        axes=axes[1],
        aspect=ASPECT,
        plot_title=True,)

    labels = ["A", "B", "Comb",]
    colours = ["r", "g", "b",]

    spec_i = 0

    # Plot each order
    for order_i in range(n_orders):
        for det_i in range(3):
            # Plot each of A, B, and combined data
            for obs_i, (obs, label) in enumerate(zip(observations, labels)):
                # Normalise, avoiding edges
                flux = obs.spectra_1d[spec_i].flux
                flux_norm = flux / np.nanmedian(flux[20:-20])

                # Determine SNR
                snr = int(np.nanmedian(flux) / np.sqrt(np.nanmedian(flux)))

                # Plot with label having SNR in brackets
                axes[2+order_i, det_i].plot(
                    obs.spectra_1d[spec_i].wave,
                    flux_norm,
                    linewidth=0.1,
                    label="{} (S/N~{:0.0f})".format(label, snr),
                    color=colours[obs_i],
                    alpha=0.8,)
            
            # Label Y axes and ticks only on left
            if det_i == 0:
                axes[2+order_i, det_i].set_ylabel("Flux")
            else:
                axes[2+order_i, det_i].set_yticks([])
            
            # And X axes only on bottom
            if order_i == n_orders-1:
                axes[2+order_i, det_i].set_xlabel(r"Wavelength ($\mu$m)")
            
            # Set tick font size
            axes[2+order_i, det_i].tick_params(
                axis='y', which='major', labelsize="xx-small")

            axes[2+order_i, det_i].tick_params(
                axis='x', which='major', labelsize="xx-small")

            # Add legend
            leg = axes[2+order_i, det_i].legend(
                fontsize="xx-small",
                loc="upper center",
                ncol=3,)

            for legobj in leg.legendHandles:
                legobj.set_linewidth(1.5)
            
            # Add text labelling the order/detector combo
            wave = obs.spectra_1d[spec_i].wave
            w_mid = (wave[-1] - wave[0])/2 + wave[0]

            axes[2+order_i, det_i].text(
                x=w_mid,
                y=0,
                s="Order: {:0.0f}, Detector: {:0.0f}".format(
                    min_order+order_i, det_i+1),
                horizontalalignment="center",
                fontsize="xx-small",)

            # Set appropriate y limits
            axes[2+order_i, det_i].set_ylim(Y_BOUND_LOW, Y_BOUND_HIGH)

            # Finally index our spectral count
            spec_i += 1

    # Add title to current folder
    plt.suptitle(os.path.join(cwd, folder))

    # Only show if we're plotting a single PDF
    if len(folders) == 1:
        plt.show()

    # Save plot
    pdf_name = os.path.join(folder, "reduction_diagnostic.pdf")
    all_diagnostic_pdfs.append(pdf_name)

    plt.savefig(pdf_name)

# All plots are made, if we made more than one stitch them together
# https://stackoverflow.com/questions/3444645/merge-pdf-files
if do_pdf_merge and len(all_diagnostic_pdfs) > 1:
    merger = PdfFileMerger()

    root_path = os.path.dirname(os.path.dirname(pdf_name))
    night_name = os.path.split(root_path)[-1]
    merged_pdf = os.path.join(
        root_path, "reduction_diagnostic_{}.pdf".format(night_name))

    for pdf in all_diagnostic_pdfs:
        merger.append(open(pdf, 'rb'))

    with open(merged_pdf, 'wb') as fout:
        merger.write(fout)


