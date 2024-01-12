"""Script to plot a diagnostic plot to easily inspect how a CRIRES reduction
went. This is part of a series of python scripts to assist with reducing raw
CRIRES+ data, which should be run in the following order:

    1 - make_calibration_sof.py             [reducing calibration files]
    2 - make_nodding_sof_master.py          [reducing master AB nodded spectra]
    3 - make_nodding_sof_split_nAB.py       [reducing nodding time series]
    4 - blaze_correct_nodded_spectra.py     [blaze correcting all spectra]
    5 - make_diagnostic_reduction_plots.py  [creating multipage PDF diagnostic]

The diagnostic plot consists of the trace waves from the A and B frames
overplotted on their respective images; as well as the overplotted A, B, and
combined spectra for each detector/order. This is saved as a single PDF in the
same directory as the reduced data. This script can be called on multiple 
directories at once using wildcard characters, and if so it will attempt to
stitch each individual PDF together into a single master PDF for ease of
inspection.

Run as
------
python make_diagnostic_reduction_plots.py [data_path] [run_on_blaze_corr_spec]

where [data_path] is the path to the data (which can include wildcards for
globbing). Make sure to wrap any wildcard filepaths in quotes to prevent 
globbing happening on the command line (versus in python). Note that this 
script requires objects and functions from luciferase to run, so the 
appropriate path should be inserted as below. 

[run_on_blaze_corr_spec] is a boolean indicating whether to instead generate
the diagnostic PDF plot for blaze corrected spectra, that is those with the
label "*_blaze_corr.fits" on the end. This parameter is optional, and if not
provided [run_on_blaze_corr_spec] = False.
"""
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip

sys.path.insert(1, "/home/arains/code/luciferase/")

import luciferase.spectra as lspec
import luciferase.plotting as lplt
import luciferase.reduction as lred

try:
    from PyPDF2 import PdfFileMerger
    do_pdf_merge = True
except:
    print("PyPDF2 is not installed, cannot merge PDFs\n")
    do_pdf_merge = False

cwd = os.getcwd()

# Aspect ratio for imshow
ASPECT = "auto"

# Values used to clip extremes of trace wave plot
VMIN_PC = 15
VMAX_PC = 85

# Y Bounds to plot
Y_BOUND_LOW = -0.1
Y_BOUND_HIGH_N_SIGMA = 5.0

# Edge pixels to avoid when normalising by median
EDGE_PX = 200

# Seeing FWHM fflag value. Seeing values above this will be noted as concerning
FWHM_GOOD_TRESHOLD = 0.2
ARCSEC_PER_PX = 0.059
RESAMPLING_FAC = 12

# Determines whether to annotate each order with its median SNR or median
# percentage uncertainty. This is important because SNR as determined from
# Poisson statistics is only valid if the units of flux are in raw counts, vs
# the result of a median/mean/weighted average combination of frames. Percent
# uncertainty should be used for visualising CRIRES+ master reductions.
# TODO: unimplemented currently
#USE_POISSON_UNCERTAINTY_INSTEAD_OF_PIPELINE_SNR = True

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

SLIT_FUNCS = [
    "cr2res_obs_nodding_slitfuncA.fits",
    "cr2res_obs_nodding_slitfuncB.fits"
]

# Double check we haven't been given too many inputs
if len(sys.argv) > 3:
    exception_text = (
        "Warning, too many inputs ({})! "
        "Make sure to wrap paths with wildcard characters in quotes.")

    raise Exception(exception_text.format(len(sys.argv)))

# Take as input the folder/s to consider
data_dir = sys.argv[1]

# Consider the optional parameter of whether to run on blaze corrected spectra.
# If so, modify our base EXTRACTED_FILES to have '_blaze_corr' on the end.
if len(sys.argv) > 2:
    if sys.argv[2].upper() == "TRUE":
        run_on_blaze_corr_spec = True
    else:
        run_on_blaze_corr_spec = False

    if run_on_blaze_corr_spec:
        print("Running on blaze corrected spectra.")
        pdf_base = "reduction_diagnostic_blaze_corr"

        # Update our list of extracted files
        EXTRACTED_FILES = [fn.replace(".fits", "_blaze_corr.fits")
                        for fn in EXTRACTED_FILES]
    else:
        pdf_base = "reduction_diagnostic"
else:
    run_on_blaze_corr_spec = False
    pdf_base = "reduction_diagnostic"
    
# Figure out just how many folder's we're working with
folders = glob.glob(data_dir)
folders.sort()

# Convert to absolute paths
folders = [os.path.abspath(folder) for folder in folders]

print("Found {:0.0f} folder/s to create diagnostic plots for.".format(
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

    for file_i, extr_file in enumerate(EXTRACTED_FILES):
        extr_path = os.path.join(folder, extr_file)

        # Only A and B frames have a slit function
        if "combined" in extr_file:
            sf_path = None
        else:
            sf_path = os.path.join(folder, SLIT_FUNCS[file_i])

        observations.append(
            lspec.initialise_observation_from_crires_nodding_fits(
                fits_file_nodding_extracted=extr_path,
                fits_file_slit_func=sf_path,))

    # Get information about orders and detectors
    n_orders = observations[0].n_orders
    min_order =  observations[0].min_order
    max_order =  observations[0].max_order
    n_detectors = observations[0].n_detectors

    # Format of plot will be the first two rows being the A and B images + TW
    # Then every subsequent row will be a given order.

    # Number of rows is 3 + number of orders, one column for each detector
    n_col = 3
    n_ax_base = 3
    n_rows = n_ax_base + n_orders

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
        plot_title=True,
        vmin_pc=VMIN_PC,
        vmax_pc=VMAX_PC,)

    # Plot B frame tracewave
    lplt.plot_trace_vs_reduced_frame(
        fits_trace=os.path.join(folder, TRACE_WAVES[1]),
        fits_image=os.path.join(folder, EXTRACTED_IMAGES[1]),
        fig=fig,
        axes=axes[1],
        aspect=ASPECT,
        plot_title=True,
        vmin_pc=VMIN_PC,
        vmax_pc=VMAX_PC,)

    labels = ["A", "B", "Comb",]
    colours = ["r", "g", "b",]
    fmts = ["-", "--"]

    spec_i = 0

    # Plot each order: TODO: A/B/combined indexing might be different
    for order_i in range(n_orders):
        for det_i in range(3):
            # Not all orders are available for all detectors. Abort if either:
            #  1) We're out of spectra objects.
            #  2) Our current det_i counter =/= det_i for the next spectra obj.
            if (spec_i+1 > len(observations[0].spectra_1d) 
                or observations[0].spectra_1d[spec_i].detector_i != det_i + 1):
                # Delete the axis, index counter, but don't do anything else.
                fig.delaxes(axes[n_ax_base+order_i, det_i])
                spec_i += 1
                continue
            
            # Initialise array of FWHMs
            fwhms = []
            fwhm_above_threshold = False

            # Plot each of A, B, and combined data
            for obs_i, (obs, label) in enumerate(zip(observations, labels)):
                # Normalise, avoiding edges
                flux = obs.spectra_1d[spec_i].flux
                flux_norm = flux / np.nanmedian(flux[EDGE_PX:-EDGE_PX])

                # Record seeing and flaf if concerning
                fwhms.append(obs.spectra_1d[spec_i].seeing_arcsec)

                if obs.spectra_1d[spec_i].seeing_arcsec > FWHM_GOOD_TRESHOLD:
                    fwhm_above_threshold = True

                # Report the SNR based on Poisson uncertainties
                #if USE_POISSON_UNCERTAINTY_INSTEAD_OF_PIPELINE_SNR:
                #    snr = int(np.nanmedian(flux) / np.sqrt(np.nanmedian(flux)))
                #    leg_label = "{} (S/N~{:0.0f})".format(label, snr)

                # Report the SNR based on the pipeline uncertainties
                # TODO: properly implement the scaled Poisson uncertainties
                # (i.e. scale counts by NDIT, NEXP, NABCYCLES, and GAIN)
                sigma = obs.spectra_1d[spec_i].sigma
                snr = np.nanmedian(flux) / np.nanmedian(sigma)
                leg_label = "{} (S/N~{:0.0f})".format(label, snr)

                # Plot the slit function for A and B frames
                if label in ["A", "B"]:
                    # Get x axis for slit func
                    n_px = len(obs.spectra_1d[spec_i].slit_func)
                    xx = ((np.arange(n_px) - n_px/2) * ARCSEC_PER_PX 
                          / (RESAMPLING_FAC-1))

                    label = "{}:{:0.0f}".format(
                                label, obs.spectra_1d[spec_i].order_i)
                    
                    # Mask slit function to avoid spikes
                    slit_func = obs.spectra_1d[spec_i].slit_func.copy()
                    #slit_func[slit_func > 1] = np.nan

                    sf_line, = axes[n_ax_base-1, det_i].plot(
                        xx,
                        slit_func,
                        linestyle=fmts[obs_i],
                        linewidth=0.2,
                        label=label,
                        alpha=0.8,)

                    # Set limit
                    axes[n_ax_base-1, det_i].set_ylim([0, 0.4])

                    # Label
                    y_max = np.max(obs.spectra_1d[spec_i].slit_func)
                    y_max_i = np.argmax(obs.spectra_1d[spec_i].slit_func)

                    axes[n_ax_base-1, det_i].text(
                        x=xx[y_max_i],
                        y=y_max,
                        s=label,
                        color=sf_line.get_color(),
                        horizontalalignment="center",
                        fontsize=4,
                    )

                # Plot with label having SNR in brackets
                axes[n_ax_base+order_i, det_i].plot(
                    obs.spectra_1d[spec_i].wave,
                    flux_norm,
                    linewidth=0.2,
                    label=leg_label,
                    color=colours[obs_i],
                    alpha=0.8,)
            
            # Label Y axes only on left
            if det_i == 0:
                axes[n_ax_base+order_i, det_i].set_ylabel("Flux")
            
            axes[n_ax_base+order_i, det_i].set_yticks([])
            axes[n_ax_base-1, det_i].set_yticks([])
            
            # X label for slit func
            axes[n_ax_base-1, det_i].set_xlabel(
                "Seeing (arcsec)",
                fontsize="xx-small",)

            # But X label only on bottom for spectra
            if order_i == n_orders-1:
                axes[n_ax_base+order_i, det_i].set_xlabel(
                    r"Wavelength ($\mu$m)")
            
            # Set tick font size for slit func axis
            axes[n_ax_base-1, det_i].tick_params(
                axis='y', which='major', labelsize="xx-small")
            
            axes[n_ax_base-1, det_i].tick_params(
                axis='x', which='major', labelsize="xx-small")

            # And spectra axis
            axes[n_ax_base+order_i, det_i].tick_params(
                axis='y', which='major', labelsize="xx-small")

            axes[n_ax_base+order_i, det_i].tick_params(
                axis='x', which='major', labelsize="xx-small")

            # Add title to slit func plots
            axes[n_ax_base-1, det_i].set_title(
                "slit_func, detector {}".format(det_i+1),
                fontsize="xx-small",)

            # Add legend to slit func plot
            leg_sf = axes[n_ax_base-1, det_i].legend(
                fontsize="xx-small",
                loc="right",
                ncol=2,)

            for legobj in leg_sf.legendHandles:
                legobj.set_linewidth(1.5)

            # Add legend to spectra plot
            leg = axes[n_ax_base+order_i, det_i].legend(
                fontsize="xx-small",
                loc="upper center",
                ncol=3,)

            for legobj in leg.legendHandles:
                legobj.set_linewidth(1.5)
            
            # Add text labelling the order/detector combo
            wave = obs.spectra_1d[spec_i].wave
            w_mid = (wave[-1] - wave[0])/2 + wave[0]

            # Note that order_i is better thought of as the plot index, which
            # is why we're sourcing the actual order_i from the spectum itself.
            axes[n_ax_base+order_i, det_i].text(
                x=w_mid,
                y=0,
                s="Order: {:0.0f}, Detector: {:0.0f}".format(
                    obs.spectra_1d[spec_i].order_i, det_i+1),
                horizontalalignment="center",
                fontsize="xx-small",)

            # Add text identifying the seeing
            if fwhm_above_threshold:
                fwhm_text_colour = "r"
            else:
                fwhm_text_colour = "k"

            seeing_label = "Seeing: A~{:0.3f}\", B~{:0.3f}\"".format(
                fwhms[0], fwhms[1])

            axes[n_ax_base+order_i, det_i].text(
                x=w_mid,
                y=0.1,
                s=seeing_label,
                horizontalalignment="center",
                fontsize="xx-small",
                color=fwhm_text_colour,)

            # Set appropriate y limits
            clipped_flux = sigma_clip(
                flux_norm[np.isfinite(flux_norm)], 
                sigma_upper=5)
            upper_bound = 1 + Y_BOUND_HIGH_N_SIGMA * np.nanstd(clipped_flux)
            axes[n_ax_base+order_i, det_i].set_ylim(Y_BOUND_LOW, upper_bound)

            # Finally index our spectral count
            spec_i += 1

    # Add title to current folder
    if run_on_blaze_corr_spec:
        title = "{} (blaze corrected)".format(os.path.join(cwd, folder))
    else:
        title = os.path.join(cwd, folder)

    plt.suptitle(title)

    # Save plot
    pdf_name = os.path.join(folder, "reduction_diagnostic.pdf")
    all_diagnostic_pdfs.append(pdf_name)

    plt.savefig(pdf_name)

    # Only show if we're plotting a single PDF
    if len(folders) == 1:
        plt.show()

# All plots are made, if we made more than one stitch them together
# https://stackoverflow.com/questions/3444645/merge-pdf-files
if do_pdf_merge and len(all_diagnostic_pdfs) > 1:
    merger = PdfFileMerger()

    root_path = os.path.dirname(os.path.dirname(pdf_name))
    night_name = os.path.split(root_path)[-1]
    merged_pdf = os.path.join(
        root_path, "{}_{}.pdf".format(pdf_base, night_name))

    for pdf in all_diagnostic_pdfs:
        merger.append(open(pdf, 'rb'))

    with open(merged_pdf, 'wb') as fout:
        merger.write(fout)