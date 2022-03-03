"""
"""
import os
import glob
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from astropy.io import fits

# Ensure the plotting folder exists to save to
here_path = os.path.dirname(__file__)
plot_dir = os.path.abspath(os.path.join(here_path, "..", "plots"))

if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

def plot_2d_time_series_spectra(
    spectra_path_with_wildcard,
    label="",
    path="plots"):
    """Function to plot a 2D image of a series of CRIRES+ spliced spectra.
    """
    # Grab filenames and sort
    spec_seq_files = glob.glob(spectra_path_with_wildcard)
    spec_seq_files.sort()

    # Put all spectra in arrays
    wave = []
    spectra = []
    e_spectra = []
    times = []

    for spec_file in spec_seq_files:
        spec_fits = fits.open(spec_file)
        wave.append(spec_fits[1].data["SPLICED_1D_WL"])
        spectra.append(spec_fits[1].data["SPLICED_1D_SPEC"])
        e_spectra.append(spec_fits[1].data["SPLICED_1D_ERR"])
        times.append(spec_fits[0].header["DATE-OBS"].split("T")[-1].split(".")[0])

    obj = fits.getval(spec_file, "OBJECT")

    wave = np.array(wave)
    spectra = np.array(spectra)
    e_spectra = np.array(e_spectra)

    # Mask spectra
    mask = np.logical_and(spectra < 10, spectra > 0)
    masked_spec = np.ma.masked_array(spectra, ~mask)

    # Plot spectra in chunks
    delta_wl = np.abs(wave[0, :-1] - wave[0, 1:])
    gap_px = np.argwhere(delta_wl > 10*np.median(delta_wl))[:,0] + 1
    gap_i0 = 0

    # Plot spectra
    plt.close("all")
    fig, axes = plt.subplots(1, len(gap_px))

    vmin = np.min(masked_spec)
    vmax = np.max(masked_spec)

    for gi, gap_i in enumerate(gap_px):
        axes[gi].imshow(
            masked_spec[:, gap_i0:gap_i],
            aspect="auto",
            interpolation="none",
            extent=[wave[0, gap_i0:gap_i][0], wave[0, gap_i0:gap_i][-1], 1, 13],
            vmin=vmin,
            vmax=vmax,)

        # Hide y labels for all but the first panel
        if gi > 0:
            axes[gi].set_yticks([])
            axes[gi].set_yticklabels([])

        plt.setp(axes[gi].get_xticklabels(), rotation=90, ha='right')

        # Update the lower bound
        gap_i0 = gap_i
    
    fig.supxlabel("Wavelength")
    fig.supylabel("Time")
    plt.gcf().set_size_inches(18, 3)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0)

    save_fn = "2d_spectra_{}".format(obj)
    
    if label != "":
        save_fn = save_fn + "_" + label

    save_path_pdf = os.path.join(path, "{}.pdf".format(save_fn))
    save_path_png = os.path.join(path, "{}.pdf".format(save_fn))
    plt.savefig(save_path_pdf)
    plt.savefig(save_path_png, dpi=300)


def plot_1d_spectra(
    spectra_path_with_wildcard,
    offset=0,
    label="",
    path="plots",
    leg_ncol=9,
    leg_bbox_to_anchor=(0.5,1.2),
    y_axis_pad=20,):
    """Function to plot 1D extracted CRIRES+ spectra that have been blaze
    corrected and spliced.
    """
    # Grab filenames and sort
    spec_seq_files = glob.glob(spectra_path_with_wildcard)
    spec_seq_files.sort()

    # Setup colours
    cmap = cm.get_cmap("viridis")
    colour_steps = np.linspace(0, 0.9, len(spec_seq_files))

    # Plot sequence of spectra
    plt.close("all")
    fig, axis = plt.subplots()

    for si, sf in enumerate(spec_seq_files):
        # Read wl, spectrum, and err
        wl = fits.open(sf)[1].data["SPLICED_1D_WL"]
        spec = fits.open(sf)[1].data["SPLICED_1D_SPEC"]
        e_spec = fits.open(sf)[1].data["SPLICED_1D_ERR"]

        # Grab observation times and the object name from the headers
        obsdate = fits.getval(sf, "DATE-OBS").split("T")[-1].split(".")[0]
        obj = fits.getval(sf, "OBJECT")

        # Mask out bad regions
        bad_px_mask = np.logical_or(spec > 3, spec < 0)

        # Plot spectra in chunks
        delta_wl = np.abs(wl[:-1] - wl[1:])
        gap_px = np.argwhere(delta_wl > 10*np.median(delta_wl))[:,0] + 1
        gap_i0 = 0

        # Add last pixel
        gap_px = np.concatenate((gap_px, [len(wl)]))

        for gap_i in gap_px:
            axis.plot(
                wl[gap_i0:gap_i][~bad_px_mask[gap_i0:gap_i]],
                spec[gap_i0:gap_i][~bad_px_mask[gap_i0:gap_i]] + offset*si,
                color=cmap(colour_steps[si]),
                linewidth=0.1,
                label="Spec #{}, {}".format(si, obsdate),
                alpha=0.8)

            # Update the lower bound
            gap_i0 = gap_i
    
    # Grab the object name
    obj = fits.getval(sf, "OBJECT")

    # Plot (unique) legend and adjust linewidths
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Shrink current axis by 20%
    #box = axis.get_position()
    #axis.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    leg = axis.legend(
        by_label.values(),
        by_label.keys(),
        fontsize="x-small",
        loc="upper center",
        #bbox_to_anchor=leg_bbox_to_anchor,
        ncol=leg_ncol)

    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.5)

    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Flux (normalised)")
    axis.set_xlim([wl[0]-10, wl[-1]+10])
    axis.set_ylim([-1, si*offset+y_axis_pad])
    axis.set_yticks([])
    plt.gcf().set_size_inches(18, 4)
    plt.tight_layout()
    
    # Save
    save_fn = "1d_spectra_{}".format(obj.replace(" ","_"))

    if label != "":
        save_fn = save_fn + "_" + label

    save_path_pdf = os.path.join(path, "{}.pdf".format(save_fn))
    save_path_png = os.path.join(path, "{}.png".format(save_fn))
    plt.savefig(save_path_pdf)
    plt.savefig(save_path_png, dpi=300)


def plot_molecfit_performance(
    molecfit_path,
    sci_spec_file="SCIENCE.fits",
    model_spec_file="MODEL/BEST_FIT_MODEL.fits",
    corr_spec_file="CORRECT/SCIENCE_TELLURIC_CORR_SCIENCE.fits",
    ignore_px=20,
    linewidth=0.1,):
    """Function to plot a comparison of Molecfit's performance of correcting
    for telluric features. Saved as a pdf and png in <molecfit_path>/plots/.

    Parameters
    ----------
    molecfit_path: string
        Filepath to the base molecfit directory.

    sci_spec_file, model_spec_file, corr_spec_file: string
        Relative filepaths to the science, best fitting telluric model, and
        telluric corrected science fits files stored within the base molecfit
        directory.
    
    ignore_px: int, default: 20
        How many pixels to ignore from each side of the detector when plotting.

    linewidth: float, default: 0.1
        Linewidth for plotted spectra.
    """
    # Setup file paths
    sci_spec_file = os.path.join(molecfit_path, sci_spec_file)
    model_spec_file = os.path.join(molecfit_path, model_spec_file)
    corr_spec_file = os.path.join(molecfit_path, corr_spec_file)

    # Open files
    sci_spec = fits.open(sci_spec_file)
    model_spec = fits.open(model_spec_file)
    corr_spec = fits.open(corr_spec_file)

    # Initialise plots
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)

    # Plot science spectrum before and after correction
    for spec_i in range(1,len(corr_spec)):
        # Plot original science spectrum
        ax1.plot(
            sci_spec[spec_i].data['WAVE'][ignore_px:-ignore_px],
            sci_spec[spec_i].data['SPEC'][ignore_px:-ignore_px],
            label="Science",
            color="C0",
            alpha=0.8,
            linewidth=linewidth,)

        # Plot corrected science spectrum
        ax2.plot(
            corr_spec[spec_i].data['WAVE'][ignore_px:-ignore_px],
            corr_spec[spec_i].data['SPEC'][ignore_px:-ignore_px],
            label="Science (Corrected)",
            color="C0",
            alpha=0.8,
            linewidth=linewidth,)

    # Overplot telluric model (first just points, then lines)
    ax1.plot(
        1000.*model_spec[1].data['lambda'],
        model_spec[1].data['mflux'],
        label='Telluric Model',
        color="C1",
        marker='.',
        markersize=linewidth*3,
        markeredgewidth=0,
        alpha=0.8,
        linewidth=0)
     
    for chip_i in range(max(model_spec[1].data['chip'])):
        chip_indices = np.where(model_spec[1].data['chip'] == chip_i+1)
        ax1.plot(
            1000.*model_spec[1].data['mlambda'][chip_indices],
            model_spec[1].data['mflux'][chip_indices],
            color="C1",
            linewidth=linewidth)

    # Plot (unique) legend and adjust linewidths
    for axis in (ax1, ax2):
        handles, labels = axis.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        leg = axis.legend(
            by_label.values(),
            by_label.keys(),
            fontsize="small",
            loc="upper center",
            ncol=2,)

        for legobj in leg.legendHandles:
            legobj.set_linewidth(1.5)

    ax2.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Flux (normalised)")
    ax2.set_ylabel("Flux (normalised)")

    # Set x limits
    wls = 1000.*model_spec[1].data['lambda']

    ax1.set_xlim([np.min(wls)-10, np.max(wls)+10])
    ax2.set_xlim([np.min(wls)-10, np.max(wls)+10])
    #ax1.set_xticks([])

    # Set y limits
    med_flux = np.nanmedian(model_spec[1].data['mflux'])

    ax1.set_ylim([0, 2*med_flux])
    ax2.set_ylim([0, 2*med_flux])
    
    # Set title with object and date
    obj = sci_spec[0].header["OBJECT"]
    date = sci_spec[0].header["DATE-OBS"].split("T")[0].split(".")[0]

    fig.suptitle("{} ({})".format(obj, date))

    plt.gcf().set_size_inches(18, 4)
    plt.tight_layout()
    
    # Save
    save_loc = os.path.join(
        molecfit_path, "plots", "molec_fit_results_{}".format(
            obj.replace(" ", "_")))

    save_path_pdf = os.path.join("{}.pdf".format(save_loc))
    save_path_png = os.path.join("{}.png".format(save_loc))
    plt.savefig(save_path_pdf)
    plt.savefig(save_path_png, dpi=300)