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
    """Function to plot a 2D image of a series of CRIRES+ spliced spectra
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
    leg_ncol=8,
    leg_bbox_to_anchor=(0.5,2)):
    """
    """
    # Grab filenames and sort
    spec_seq_files = glob.glob(spectra_path_with_wildcard)
    spec_seq_files.sort()

    # Setup colours
    cmap = cm.get_cmap("viridis")
    colour_steps = np.linspace(0, 0.9, len(spec_seq_files))

    # Plot sequence of spectra
    plt.close("all")
    for si, sf in enumerate(spec_seq_files):
        # Read wl, spectrum, and err
        wl = fits.open(sf)[1].data["SPLICED_1D_WL"]
        spec = fits.open(sf)[1].data["SPLICED_1D_SPEC"]
        e_spec = fits.open(sf)[1].data["SPLICED_1D_ERR"]

        # Grab observation times and the object name from the headers
        obsdate = fits.getval(sf, "DATE-OBS").split("T")[-1].split(".")[0]
        obj = fits.getval(sf, "OBJECT")

        # Mask out bad regions
        bad_px_mask = np.logical_or(spec > 10, spec < 0)

        # Plot spectra in chunks
        delta_wl = np.abs(wl[:-1] - wl[1:])
        gap_px = np.argwhere(delta_wl > 10*np.median(delta_wl))[:,0] + 1
        gap_i0 = 0

        # Add last pixel
        gap_px = np.concatenate((gap_px, [len(wl)]))

        for gap_i in gap_px:
            plt.plot(
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

    leg = plt.legend(
        by_label.values(),
        by_label.keys(),
        fontsize="small",
        loc="upper center",
        bbox_to_anchor=leg_bbox_to_anchor,
        ncol=leg_ncol)

    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.5)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux (normalised)")
    plt.xlim([wl[0]-10, wl[-1]+10])
    plt.gcf().set_size_inches(18, 3)
    plt.tight_layout()
    
    # Save
    save_fn = "1d_spectra_{}".format(obj.replace(" ","_"))

    if label != "":
        save_fn = save_fn + "_" + label

    save_path_pdf = os.path.join(path, "{}.pdf".format(save_fn))
    save_path_png = os.path.join(path, "{}.pdf".format(save_fn))
    plt.savefig(save_path_pdf)
    plt.savefig(save_path_png, dpi=300)