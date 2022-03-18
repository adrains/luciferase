"""Originally written by Alexis Lavail, modified by Adam Rains.

Script to generate SOF files for CRIRES+ reductions in nodding mode.

Run as:
    python3 nodding-make_sofs.py [RAW2MASTER.XML]

Where you replace [RAW2MASTER.XML] with the actual CRIRES..._raw2master.xml 
filename, or pass in multiple XML files using a wildcard.

Creates a nod.sof SOF file and a reduce.sh script that reduces your nodding 
observations for each XML file, and places each in a separate subfolder for
each grating/wavelength setting (e.g. K2148).
"""
import numpy as np
import xml.etree.ElementTree as ET
import glob
import sys
from astropy.io import fits
import subprocess
import sys,os

# Get list of XML file names
xml_files = glob.glob(sys.argv[1])
print(xml_files)
nf = len(xml_files)
print(nf)

# Before we start looping, archive the old reduce.sh script if it exists
reduce_script = "reduce.sh"

if os.path.isfile(reduce_script):
    bashCommand = "mv {} {}.old".format(reduce_script, reduce_script)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

# And make a new reduction script
with open(reduce_script, "w") as rs:
    rs.write("#!/bin/bash\n")

cmd = "chmod +x {}".format(reduce_script)
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

# For each XML file write a pair of <wl_setting>.sof and reduce.sh files in a
# subfolder for each grating setting (e.g. K2148).
for xml_file in xml_files:
    print(xml_file)
    sof = 'nod.sof'
    with open(sof, 'w') as wri: 
        # Parse XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Loop on XML elements
        for elem in root:
            for subelem in elem:
                # Science observations
                if subelem.attrib['category'] == 'OBS_NODDING_OTHER':
                    # Print update, write line to SOF file
                    print("{:s} \t {:s}".format(
                        subelem.attrib['category'],
                        subelem.attrib['name']))

                    file_path = os.path.join(
                        os.getcwd(),
                        "{}.fits".format(subelem.attrib['name'],))

                    wri.write("{}\t{}\n".format(
                        file_path, subelem.attrib['category']))

                    ffi = subelem.attrib['name'] + '.fits'

                # Dark or flat
                else:
                    print("{:s} \t {:s}".format(
                        subelem.attrib['category'],
                        subelem[0][0].attrib['name']))
                    
                    if (subelem.attrib['category'] == 'CAL_DARK_BPM' 
                        or subelem.attrib['category'] == 'CAL_FLAT_MASTER'):
                        # Write line to SOF file
                        file_path = os.path.join(
                            os.getcwd(),
                            "{}.fits".format(subelem[0][0].attrib['name'],))

                        wri.write("{}\t{}\n".format(
                            file_path, subelem.attrib['category']))
        
        # Grab the grating/wavelength setting from a science frame, then get 
        # an appropritate precomputed trace wave file (from the most up-to-date
        # version of the pipeline) 
        sci = fits.open(ffi)
        wl_setting = sci[0].header['HIERARCH ESO INS WLEN ID']
        wri.write(
            '/home/tom/pCOMM/cr2re-calib/{}_tw.fits\t UTIL_WAVE_TW'.format(
                wl_setting))
    
    # Make a folder for this wavelength setting if it doesn't already exist
    print(wl_setting)
    try:
        os.mkdir(wl_setting)
    except FileExistsError:
        pass
    
    # Move the new SOF file into this subdirectory and rename
    cwd = os.getcwd()
    new_sof_path = os.path.join(cwd, wl_setting, "{}.sof".format(wl_setting))
    bashCommand = "mv nod.sof {}".format(new_sof_path)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # And finally write the a file containing esorex reduction commands
    with open("reduce.sh", 'a') as ww:
        ww.write("cd {}\n".format(os.path.join(cwd, wl_setting)))
        esorex_cmd = ('esorex cr2res_obs_nodding --extract_swath_width=800'
                      + ' --extract_oversample=12 --extract_height=30 '
                      + new_sof_path + '\n')
        ww.write(esorex_cmd)
