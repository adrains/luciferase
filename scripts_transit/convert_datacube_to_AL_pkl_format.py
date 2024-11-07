"""Converts luciferase format datacube to Alexis Lavail's format.

Description of Alexis' format can be found here:
https://box.in2p3.fr/index.php/s/pCJKtJbzzD4NtaM

FL needs the following to run his pipeline:
[wave, flux, error, time, and v_bary]
"""
import numpy as np
import pickle
import transit.utils as tu

save_path = "simulations"
label = "wasp107_np_corr"
n_transit = 2

# Import data
waves, fluxes_list, sigmas_list, det, orders, transit_info_list, syst_info = \
    tu.load_transit_info_from_fits(save_path, label, n_transit)

# Initialise dictionary keys
dict_keys = ["script_version", "nodpos", "rawfilename", "nodpair", "wave", 
    "wave_model", "spec", "err", "snr", "airmass", "bjd_tdb", "berv", "orders",
    "tell", "rawheaders", "slitfunctionFWHM-det1", "slitfunctionFWHM-det2",
    "slitfunctionFWHM-det3"]

# Write one pkl per night
for transit_i in range(n_transit):
    print("Running on night {}/{}".format(transit_i+1, n_transit))
    # Intialise dict
    data_dict = {}

    # For convenience, grab dataframe for this night
    transit_info = transit_info_list[transit_i]
    (n_phase, n_spec, n_px) = fluxes_list[transit_i].shape
    flux = fluxes_list[transit_i]
    sigma = sigmas_list[transit_i]

    # Nodding position
    data_dict["nod_pos"] = transit_info["nod_pos"].values

    # Raw filename
    data_dict["rawfilename"] = transit_info["raw_file"].values

    # Nod pair
    data_dict["nodpair"] = ["pair{}".format(fn.split("/")[-3][5:]) 
                            for fn in transit_info["raw_file"].values]
    
    # Wave [n_spec, n_phase, n_px]
    wave_3D = np.broadcast_to(waves[:,None,20:-20], (n_spec, n_phase, n_px-40))
    data_dict["wave"] = wave_3D

    # Wave model [n_spec, n_phase, n_px]
    data_dict["wave_model"] = []

    # Spectrum [n_spec, n_phase, n_px]
    spec_3D = flux[:,:,20:-20]
    data_dict["spec"] = spec_3D

    # err [n_spec, n_phase, n_px]
    err_3D = sigma[:,:,20:-20]
    data_dict["err"] = err_3D

    # SNR [n_spec, n_phase]
    data_dict["snr"] = np.nanmedian(flux / sigma, axis=-1).T

    # airmass [n_phase]
    data_dict["airmass"] = transit_info["airmass"].values

    # bjd_tdb [n_phase]
    data_dict["bjd_tbd"] = transit_info["mjd_mid"].values + 2400000.5

    # berv [n_phase], in cm
    data_dict["berv"] = transit_info["bcor"].values * 10000 # convert to cm/s

    # orders [n_spec]
    data_dict["orders"] = orders

    # Write pickle file
    fn = "{}/{}_n{}.pkl".format(save_path, label, transit_i+1)

    with open(fn, "wb") as pkl:
        pickle.dump(data_dict, pkl,)