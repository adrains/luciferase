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
    alt_name="",
    path="plots",
    leg_ncol=10,
    y_axis_pad=20,
    n_edge_px_to_mask=5,
    fig_size=(18,4),
    leg_fontsize="x-small",):
    """Function to plot 1D extracted CRIRES+ spectra that have been blaze
    corrected and spliced. Saved as a pdf and png in <molecfit_path>/plots/.

    Parameters
    ----------
    spectra_path_with_wildcard: string
        Path to spectra fits files to glob, use * as a wildcard to select all
        time series observations.

    offset: float, default: 0
        Y offset to apply to each spectrum. A value of 0 means the spectra are
        overplotted.

    label: string, default: ""
        Additional label to apply to end of saved plots.

    alt_name: string, default: ""
        Alternate star name to use in plot and filename.
    
    path: string, default "plots"
        Directory to save plots. Defaults to luciferase/plots/.

    leg_ncol: int, default: 10
        The number of columns to display in the legend.

    y_axis_pad: float, default: 20
        Padding to apply to ymax so that legend fits inside bounds.

    n_edge_px_to_mask: int, default: 5
        Number of pixels to not plot from edge of each spectral segment.
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
        with fits.open(sf) as sci_fits:
            wl = sci_fits[1].data["SPLICED_1D_WL"]
            spec = sci_fits[1].data["SPLICED_1D_SPEC"]
            e_spec = sci_fits[1].data["SPLICED_1D_ERR"]

            # Grab observation times and the object name from the headers
            date = sci_fits[0].header["DATE-OBS"].split("T")[0]
            time = sci_fits[0].header["DATE-OBS"].split("T")[-1].split(".")[0]
            obj = sci_fits[0].header["OBJECT"]

            # Get nodding position. Note that we have to this from the file
            # name, as the fits headers are the same for the A and B extracted
            # nods.
            if "extractedA" in sf:
                nod_pos = "A"
            elif "extractedB" in sf:
                nod_pos = "B"
            else:
                raise NameError("Unexpected filename - must be either A or B.")

        # Mask out bad regions, including any edge pixels
        bad_px_mask = np.logical_or(spec > 3, spec < 0)
        bad_px_mask

        # Plot spectra in chunks
        delta_wl = np.abs(wl[:-1] - wl[1:])
        gap_px = np.argwhere(delta_wl > 10*np.median(delta_wl))[:,0] + 1
        gap_i0 = 0

        # Add last pixel
        gap_px = np.concatenate((gap_px, [len(wl)]))

        for gap_i in gap_px:
            # Isolate spectral segments
            wave_segment = wl[gap_i0:gap_i][~bad_px_mask[gap_i0:gap_i]]
            spec_segment = spec[gap_i0:gap_i][~bad_px_mask[gap_i0:gap_i]]

            axis.plot(
                wave_segment[n_edge_px_to_mask:-n_edge_px_to_mask],
                spec_segment[n_edge_px_to_mask:-n_edge_px_to_mask] + offset*si,
                color=cmap(colour_steps[si]),
                linewidth=0.1,
                label="Spec {} ({}), {}".format(si, nod_pos, time),
                alpha=0.8)

            # Update the lower bound
            gap_i0 = gap_i
    
    # Grab the object name
    obj = fits.getval(sf, "OBJECT")

    # Plot (unique) legend and adjust linewidths
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    leg = axis.legend(
        by_label.values(),
        by_label.keys(),
        fontsize=leg_fontsize,
        loc="upper center",
        ncol=leg_ncol)

    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.5)

    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Flux (normalised)")
    axis.set_xlim([wl[0]-10, wl[-1]+10])
    axis.set_ylim([-1, si*offset+y_axis_pad])
    axis.set_yticks([])

    # Set title with object and date
    if alt_name == "":
        fig.suptitle("{} ({})".format(obj, date))
        save_fn = "1d_spectra_{}".format(obj.replace(" ","_"))
    else:
        fig.suptitle("{} / {} ({})".format(alt_name, obj, date))
        save_fn = "1d_spectra_{}".format(alt_name.replace(" ","_"))

    plt.gcf().set_size_inches(fig_size[0], fig_size[1])
    plt.tight_layout()
    
    # Save
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


def plot_trace_vs_reduced_frame(
    fits_trace,
    fits_image,
    fig=None,
    axes=None,):
    """Diagnostic plotting function to check CRIRES+ extraction by plotting
    trace waves on top of the reduced 2D frame/image.

    Adapted from a script originally written by Dr Thomas Marquart. 

    Parameters
    ----------
    fits_trace: string
        Filepath to trace wave fits file.

    fits_image: string
        Filepath to corresponding fits image.

    fig: matplotlib.figure.Figure, default: None
        Figure object if incorporating this diagnostic into a larger plot.
    
    axes: array of matplotlib.axes._subplots.AxesSubplot, default: None
        Array of axes objects if incorporating this diagnostic into larger 
        plot.
    """
    with fits.open(fits_trace) as tw_hdu, fits.open(fits_image) as img_hdu:
        # Setup axes if we haven't been provided with any
        if fig is None and axes is None:
            plt.close("all")
            fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
            
            # Make a note to do final adjustment and save figure
            do_save_and_adjust = True

        # Otherwise don't save as these axes will be part of a larger plot
        else:
            do_save_and_adjust = False

        # Initialise X axis pixel array
        px_x = np.arange(2048)

        # Loop over the three detectors
        for det_i in range(3):
            # Turn off axis ticks
            axes[det_i].set_xticks([])
            axes[det_i].set_yticks([])

            # Load in trace wave data for this detector
            try: 
                tw_data = tw_hdu[det_i+1].data
            except:
                print('extension {:0.0f} is missing, skipping.'.format(
                    det_i+1))
                continue

            if tw_data is None:
                print('Data for CHIP{:0.0f} is empty, skipping.'.format(
                    det_i+1))
                continue
            
            # Load image data for this detector and plot
            img_data = img_hdu['CHIP{:0.0f}.INT1'.format(det_i+1)].data
            #img_data = np.ma.masked_where(np.isnan(img_data),img_data)
            img_data = np.nan_to_num(img_data)

            axes[det_i].imshow(
                img_data,
                origin='lower',
                cmap='plasma',
                vmin=np.percentile(img_data,5),
                vmax=np.percentile(img_data,98))

            # Loop over each individual trace and plot
            for tw in tw_data:
                # Plot extraction upper/lower bounds + trace after fitting poly
                pol = np.polyval(tw['Upper'][::-1], px_x)
                axes[det_i].plot(px_x, pol, ':w')

                pol = np.polyval(tw['Lower'][::-1], px_x)
                axes[det_i].plot(px_x, pol, ':w')

                pol = np.polyval(tw['All'][::-1], px_x)
                axes[det_i].plot(px_x, pol, '--w')

                if np.isnan(pol[1024]):
                    continue
                
                # Label the orders
                text = 'order: {:0.0f}     trace: {}'.format(
                    tw['order'], tw['TraceNb'])

                axes[det_i].text(
                    x=1024, 
                    y=pol[1024],
                    s=text,
                    color="w", 
                    horizontalalignment='center',
                    verticalalignment='center',
                    size=8)

            axes[det_i].axis((1,2048,1,2048))

        if do_save_and_adjust:
            fig.tight_layout(pad=0.02)
            figname = fits_trace.replace('.fits','.png')
            plt.savefig(figname, dpi=240)