"""Functions related to saving LaTeX tables for papers.
"""
import numpy as np
import pandas as pd
from astropy.time import Time

def make_observations_table(
    transit_info_list,
    fluxes_list,
    sigmas_list,):
    """Creates a LaTeX table of standard deviations for each theta coefficient
    for a set of Cannon models for blue, red, and all wavelengths. Table is
    saved to paper/table_theta_std_comp.tex.

    Parameters
    ----------
    fluxes_list, sigmas_list: list of 3D float arrays
        List of observed spectra and uncertainties (length n_transits), with
        n_transit 3D float arrays of shape [n_phase, n_spec, n_px].
    
    transit_info_list: list of pandas DataFrames
        List of transit info (length n_transits) DataFrames containing 
        information associated with each transit time step. Each DataFrame has
        columns:
    """
    n_transit = len(transit_info_list)

    header = []
    table_rows = []
    footer = []

    # Construct the header of the table
    header.append("\\begin{table}")
    header.append("\\centering")
    
    col_format = "c" * (n_transit + 2)

    header.append("\\begin{tabular}{%s}" % col_format)
    header.insert(2, r"\caption{Summary of observations}")

    # -------------------------------------------------------------------------
    # Prepare
    # -------------------------------------------------------------------------
    row_names = [
        ("Date (UTC)", "-"),
        ("Time (UTC)", "-"),
        (r"$N_{\rm exp}$ (in/out)", "-"),
        (r"$t_{\rm exp}$", "sec"),
        ("Airmass", "-"),
        #"CRIRES Band",
        ("SNR", "-"),
        (r"RV$_{\rm bary}$", r"\,km\,s$^{-1}$"),
        (r"$\Delta$RV$_{\rm bary}$", r"\,km\,s$^{-1}$"),
        (r"RV$_{\rm planet}$", r"\,km\,s$^{-1}$"),
        (r"$\Delta$RV$_{\rm planet}$", r"\,km\,s$^{-1}$"),
    ]

    data = np.hstack(((row_names), np.full((len(row_names), n_transit), "")))
    columns = ["Parameter", "Unit"] 
    columns += ["Night {}".format(i+1) for i in range(n_transit)]
    table_df = pd.DataFrame(data=data, columns=columns)

    # Loop over all transit nights
    for transit_i in range(n_transit):
        night = "Night {}".format(transit_i+1)
        
        transit_info = transit_info_list[transit_i]
        fluxes = fluxes_list[transit_i]
        sigmas = sigmas_list[transit_i]
        
        # Date/time
        mjd_start = transit_info["mjd_start"].values[0]
        time_start = Time(mjd_start, format="mjd")
        datetime_start_str = str(time_start.datetime)
        date_start_str = datetime_start_str.split(" ")[0]
        time_start_str = datetime_start_str.split(" ")[1][:5]
        
        mjd_end = transit_info["mjd_end"].values[-1]
        time_end = Time(mjd_end, format="mjd")
        datetime_end_str = str(time_end.datetime)
        date_end_str = datetime_end_str.split(" ")[0]
        time_end_str = datetime_end_str.split(" ")[1][:5]

        table_df.at[0, night] = "{}".format(date_start_str)
        table_df.at[1, night] = "{}--{}".format(time_start_str, time_end_str)

        # N Exp
        n_exp = len(transit_info)
        n_exp_in = np.sum(transit_info["is_in_transit_mid"])
        n_exp_out = np.sum(~transit_info["is_in_transit_mid"])
        table_df.at[2, night] = "{} ({}/{})".format(n_exp, n_exp_in, n_exp_out)

        # T Exp
        assert len(set(transit_info["exptime_sec"].values)) == 1
        t_exp = int(transit_info["exptime_sec"].values[0] / 100) * 100
        table_df.at[3, night] = "{:0.0f}".format(t_exp)

        # Airmass
        am_0 = transit_info["airmass"].values[0]
        am_1 = transit_info["airmass"].values[-1]
        table_df.at[4, night] = r"${:0.2f}-{:0.2f}$".format(am_0, am_1)

        # SNR
        snr = fluxes / sigmas
        snr_med = np.nanmedian(snr)
        table_df.at[5, night] = "{:0.0f}".format(snr_med)

        # RV (bary)
        rv_bary_min = -1* transit_info["bcor"].values[0]
        rv_bary_max = -1* transit_info["bcor"].values[-1]
        delta_bcor = -1* np.median(np.diff(transit_info["bcor"].values))

        table_df.at[6, night] = r"${:0.2f}-{:0.2f}$".format(
            rv_bary_min, rv_bary_max)
        table_df.at[7, night] = "{:0.2f}".format(delta_bcor)

        # RV (planet)
        rv_p_min = transit_info["v_y_mid"].values[0]
        rv_p_max = transit_info["v_y_mid"].values[-1]
        delta_rv_p = np.median(np.diff(transit_info["v_y_mid"].values))

        table_df.at[8, night] = "{:0.2f}$-${:0.2f}".format(rv_p_min, rv_p_max)
        table_df.at[9, night] = "{:0.2f}".format(delta_rv_p)

    # -------------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------------
    title_row = " & ".join(table_df.columns.values) + r"\\"
    table_rows.append(title_row)
    table_rows.append("\hline")

    for index, row in table_df.iterrows():
        latex_row = " & ".join(row) + r"\\"
        table_rows.append(latex_row)
        
    # Finish the table
    footer.append("\\end{tabular}")
    footer.append(r"\label{tab:obs_tab}")
    footer.append("\\end{table}")

    table = header + table_rows + footer
    np.savetxt("paper/table_obs_tab.tex", table, fmt="%s")