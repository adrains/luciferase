"""Functions related to saving LaTeX tables for papers.
"""
import os
import numpy as np
import pandas as pd
from astropy.time import Time

# Ensure the plotting folder exists to save to
here_path = os.path.dirname(__file__)
plot_dir = os.path.abspath(os.path.join(here_path, "..", "paper"))

if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

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


# -----------------------------------------------------------------------------
# System Summary
# -----------------------------------------------------------------------------
def make_table_system_summary(syst_info,):
    """Make the LaTeX table to summarise the system information.

    Parameters
    ----------
    syst_info: pandas DataFrame
        DataFrame of system information with columns:
        ['parameter', 'value', 'sigma', 'reference', 'bib_ref', 'comment']
    """
    # Load in the TESS target info
    cols = [
        "Parameter",
        "Unit",
        "Value",
        "Reference",]
    
    header = []
    table_rows = []
    footer = []
    notes = []
    
    # Construct the header of the table
    header.append("\\begin{table}")
    header.append("\\centering")
    header.append("\\caption{System info}")
    header.append("\\label{tab:syst_info}")
    
    header.append("\\begin{tabular}{%s}" % ("c"*len(cols)))
    header.append("\hline")
    header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols))
    header.append("\hline")

    # -------------------------------------------------------------------------
    # Star
    # -------------------------------------------------------------------------
    table_rows.append(r"\multicolumn{4}{c}{Star} \\")
    table_rows.append("\hline")

    # IDs
    id_gaia = syst_info.loc["gaia_dr3_id", "value"]
    ref_gaia = syst_info.loc["gaia_dr3_id", "bib_ref"]

    id_2mass = syst_info.loc["2mass_id", "value"]
    ref_2mass = syst_info.loc["2mass_id", "bib_ref"]

    #table_rows.append(r"Gaia DR3 ID & - & {} & \citet{{{}}} \\".format(
    #    id_gaia, ref_gaia))
    #table_rows.append(r"2MASS ID & - & {} & \citet{{{}}} \\".format(
    #    id_2mass, ref_2mass))
    
    # RA
    ra = float(syst_info.loc["ra_deg", "value"])
    ra_hr = np.floor(ra / 15)
    ra_min = np.floor((ra / 15 - ra_hr) * 60)
    ra_sec = ((ra / 15 - ra_hr) * 60 - ra_min) * 60
    ra_str = "{:02.0f} {:02.0f} {:05.2f}".format(ra_hr, ra_min, ra_sec)

    ra_ref = syst_info.loc["ra_deg", "bib_ref"]

    table_rows.append(r"RA & hh:mm:ss.ss & {} & \citet{{{}}} \\".format(
        ra_str, ra_ref))

    # DEC
    dec = float(syst_info.loc["dec_deg", "value"])
    dec_deg = np.floor(dec)
    dec_min = np.floor((dec - dec_deg) * 60)
    dec_sec = ((dec - dec_deg) * 60 - dec_min) * 60
    dec_str = "{:+02.0f} {:02.0f} {:05.2f}".format(dec_deg, dec_min, dec_sec)

    dec_ref = syst_info.loc["dec_deg", "bib_ref"]

    table_rows.append(r"Dec & dd:mm:ss.ss & {} & \citet{{{}}} \\".format(
        dec_str, dec_ref))
    
    # Plx
    plx = syst_info.loc["plx", "value"]
    e_plx = syst_info.loc["plx", "sigma"]

    plx_ref = syst_info.loc["plx", "bib_ref"]

    table_rows.append(r"Parallax & mas & ${}\pm{}$ & \citet{{{}}} \\".format(
        plx, e_plx, plx_ref))

    # Distance
    dist = syst_info.loc["dist_pc", "value"]
    e_dist = syst_info.loc["dist_pc", "sigma"]

    dist_ref = syst_info.loc["dist_pc", "bib_ref"]

    table_rows.append(r"Distance & pc & ${}\pm{}$ & \citet{{{}}} \\".format(
        dist, e_dist, dist_ref))

    # Systemic RV
    rv_syst = syst_info.loc["rv_star", "value"]
    e_rv_syst = syst_info.loc["rv_star", "sigma"]

    rv_syst_ref = syst_info.loc["rv_star", "bib_ref"]

    table_rows.append(
        r"RV & km\,s$^{{-1}}$ & ${}\pm{}$ & \citet{{{}}} \\".format(
        rv_syst, e_rv_syst, rv_syst_ref))

    # K mag
    k_mag = syst_info.loc["K_mag_2mass", "value"]
    e_k_mag = syst_info.loc["K_mag_2mass", "sigma"]

    k_mag_ref = syst_info.loc["K_mag_2mass", "bib_ref"]

    table_rows.append(
        r"$K_S$ mag & - & ${}\pm{}$ & \citet{{{}}} \\".format(
        k_mag, e_k_mag, k_mag_ref))
    
    # Mass
    m_star = syst_info.loc["m_star_msun", "value"]
    e_m_star = syst_info.loc["m_star_msun", "sigma"]

    m_star_ref = syst_info.loc["m_star_msun", "bib_ref"]

    table_rows.append(
        r"$M_\star$ & $M_\odot$ & ${}\pm{}$ & \citet{{{}}} \\".format(
        m_star, e_m_star, m_star_ref))

    # Radius
    r_star = syst_info.loc["r_star_rsun", "value"]
    e_r_star = syst_info.loc["r_star_rsun", "sigma"]

    r_star_ref = syst_info.loc["r_star_rsun", "bib_ref"]

    table_rows.append(
        r"$R_\star$ & $R_\odot$ & ${}\pm{}$ & \citet{{{}}} \\".format(
        r_star, e_r_star, r_star_ref))

    # Teff
    teff = syst_info.loc["teff_k", "value"]
    e_teff = syst_info.loc["teff_k", "sigma"]

    teff_ref = syst_info.loc["teff_k", "bib_ref"]

    table_rows.append(
        r"$T_{{\rm eff}}$ & K & ${}\pm{}$ & \citet{{{}}} \\".format(
        teff, e_teff, teff_ref))

    # vsini
    vsini = syst_info.loc["vsini", "value"]
    e_vsini = syst_info.loc["vsini", "sigma"]

    vsini_ref = syst_info.loc["vsini", "bib_ref"]

    table_rows.append(r"$v \sin i$ & km\,s$^{{-1}}$ & ${}\pm{}$ & \citet{{{}}} \\".format(
        vsini, e_vsini, vsini_ref))

    # -------------------------------------------------------------------------
    # Planet
    # -------------------------------------------------------------------------
    table_rows.append("\hline")
    table_rows.append(r"\multicolumn{4}{c}{Planet} \\")
    table_rows.append("\hline")

    # Mass
    m_p = syst_info.loc["m_planet_mearth", "value"]
    e_m_p = syst_info.loc["m_planet_mearth", "sigma"]

    m_p_ref = syst_info.loc["m_planet_mearth", "bib_ref"]

    table_rows.append(
        r"$M_P$ & $M_\oplus$ & ${}\pm{}$ & \citet{{{}}} \\".format(
        m_p, e_m_p, m_p_ref))

    # Radius
    r_p = syst_info.loc["r_planet_rearth", "value"]
    e_r_p = syst_info.loc["r_planet_rearth", "sigma"]

    r_p_ref = syst_info.loc["r_planet_rearth", "bib_ref"]

    table_rows.append(
        r"$R_P$ & $R_\oplus$ & ${}\pm{}$ & \citet{{{}}} \\".format(
        r_p, e_r_p, r_p_ref))

    # a, semi-major axis
    sma = syst_info.loc["a_planet_au", "value"]
    e_sma = syst_info.loc["a_planet_au", "sigma"]

    sma_ref = syst_info.loc["a_planet_au", "bib_ref"]

    table_rows.append(r"$a$ & AU & ${}\pm{}$ & \citet{{{}}} \\".format(
        sma, e_sma, sma_ref))

    # e, ccentricity
    ecc = syst_info.loc["e_planet", "value"]
    e_ecc = syst_info.loc["e_planet", "sigma"]

    ecc_ref = syst_info.loc["e_planet", "bib_ref"]

    table_rows.append(r"$e$ & - & ${}\pm{}$ & \citet{{{}}} \\".format(
        ecc, e_ecc, ecc_ref))

    # i, inclination
    inc = syst_info.loc["i_planet_deg", "value"]
    e_inc = syst_info.loc["i_planet_deg", "sigma"]

    inc_ref = syst_info.loc["i_planet_deg", "bib_ref"]

    table_rows.append(r"$i$ & deg & ${}\pm{}$ & \citet{{{}}} \\".format(
        inc, e_inc, inc_ref))

    # Omega, longitude of the ascending node in degrees

    # omega, argument of periapsis in degrees

    # K star rv
    k_star = syst_info.loc["k_star_mps", "value"]
    e_k_star = syst_info.loc["k_star_mps", "sigma"]

    k_star_ref = syst_info.loc["K_mag_2mass", "bib_ref"]

    table_rows.append(
        r"$K$ & m\,s$^{{-1}}$ & ${}\pm{}$ & \citet{{{}}} \\".format(
        k_star, e_k_star, k_star_ref))

    # Transit duration
    trans_dur = syst_info.loc["transit_dur_hours", "value"]
    e_trans_dur = syst_info.loc["transit_dur_hours", "sigma"]

    trans_dur_ref = syst_info.loc["transit_dur_hours", "bib_ref"]

    table_rows.append(
        r"Transit Duration & hr & ${}\pm{}$ & \citet{{{}}} \\".format(
        trans_dur, e_trans_dur, trans_dur_ref))

    # jd0
    jd0 = syst_info.loc["jd0_days", "value"]
    e_jd0 = syst_info.loc["jd0_days", "sigma"]

    jd0_ref = syst_info.loc["jd0_days", "bib_ref"]

    table_rows.append(r"JD (mid) & day & ${}\pm{}$ & \citet{{{}}} \\".format(
        jd0, e_jd0, jd0_ref))

    # period
    period = syst_info.loc["period_planet_days", "value"]
    e_period = syst_info.loc["period_planet_days", "sigma"]

    period_ref = syst_info.loc["period_planet_days", "bib_ref"]

    table_rows.append(r"$P$ & day & ${}$ & \citet{{{}}} \\".format(
        period, period_ref))
    
    # -------------------------------------------------------------------------
    # Wrap up
    # -------------------------------------------------------------------------
    # Finish the table
    footer.append("\\hline")
    footer.append("\\end{tabular}")
    
    # Add notes section with references
    notes.append("\\begin{minipage}{\linewidth}")
    notes.append("\\vspace{0.1cm}")
    
    notes.append("\\textbf{Notes:} $^a$ TESS Object of Interest ID, "
                 "$^b$ TESS Input Catalogue ID "
                 "\citep{stassun_tess_2018, stassun_revised_2019},"
                 "$^c$2MASS \citep{skrutskie_two_2006}, "
                 "$^c$Gaia \citep{brown_gaia_2018} - "
                 " note that Gaia parallaxes listed here have been "
                 "corrected for the zeropoint offset, "
                 "$^d$Number of candidate planets, NASA Exoplanet Follow-up "
                 "Observing Program for TESS \\\\")
    
    notes.append("\\end{minipage}")
    footer.append("\\end{table}")
    
    # Write the tables
    table = header + table_rows + footer# + notes

    # Write the table
    np.savetxt("paper/table_system_info.tex", table, fmt="%s")