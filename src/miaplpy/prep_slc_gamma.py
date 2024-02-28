#!/usr/bin/env python3
############################################################
# Program is part of MiaplPy                                #
# Author:  Mahmud Haghighi, Sara Mirzaee, Zhang Yunjun, Heresh Fattahi  #
############################################################
# Modified from miaplpy/prep_slc_isce.py and mintpy/prep_gamma.py

import os
import sys
import warnings
import logging
from mintpy.prep_gamma import (get_lalo_ref, extract_metadata4geometry_radar, extract_metadata4geometry_geo)
from mintpy.utils import readfile as mt_readfile

warnings.filterwarnings("ignore")

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


blockPrint()
from mintpy.utils import writefile
from miaplpy.objects.utils import read_attribute
enablePrint()


EXAMPLE = """example:
  prep_slc_gamma.py ../rslc/20*/*slc   #for slc
  prep_slc_gamma.py ../rslc/20*/*slc ../geometry/*.dem ../geometry/*.UTM_TO_RDC   #for slc and geometry files
  """


def extract_metadata4slc(fname):
    print('preparing RSC file for ', fname)
    slc_metadata = read_attribute(fname, metafile_ext='.par')
    # update date yyyy -> yyyymmgg
    # TODO: update read_attribute to correctly read date from par file
    atr = read_gamma_slc_par(fname+'.par', skiprows=2)
    slc_metadata['DATE'] = atr['date']
    slc_metadata.pop('date',None)

    get_lalo_ref(fname+'.par', slc_metadata)

    # write .rsc file
    rsc_file = fname + '.rsc'
    update_mode = True
    writefile.write_roipac_rsc(slc_metadata, rsc_file,
                                   update_mode=update_mode,
                                   print_msg=False)
    return


def main(iargs=None):
    fnames = list(iargs)

    # loop for each file
    for fname in fnames:
        print(fname)
        file_ext = fname.split('.')[-1]
        # slc
        if file_ext in ['slc', 'rslc']:
            extract_metadata4slc(fname)

            # geometry - geo
        elif file_ext in ['utm_to_rdc'] or fname.endswith('utm.dem'):
            extract_metadata4geometry_geo(fname)

            # geometry - radar
        elif fname.endswith(('rdc.dem', 'hgt_sim')):
            extract_metadata4geometry_radar(fname)

    return


def read_gamma_slc_par(fname, delimiter=':', skiprows=3):
    """from Mintpy
    Read GAMMA .par/.off file into a python dict structure.
    Parameters: fname : str.
                    File path of .par, .off file.
                delimiter : str, optional
                    String used to separate values.
                skiprows : int, optional
                    Skip the first skiprows lines.
    Returns:    parDict : dict
                    Attributes dictionary
    """
    # Read txt file
    with open(fname) as f:
        lines = f.readlines()[skiprows:]

    # convert list of str into dict
    parDict = {}
    for line in lines:
        c = [i.strip() for i in line.strip().split(delimiter, 1)]
        if len(c) < 2 or line.startswith(('%', '#')):
            next
        else:
            key = c[0]
            value = str.replace(c[1], '\n', '').split("#")[0].split()[0].strip()
            parDict[key] = value
            if key.lower() == 'date':
                value = ''.join(str.replace(c[1], '\n', '').split("#")[0].split()[0:3])
                parDict[key] = value

    parDict = mt_readfile._attribute_gamma2roipac(parDict)
    parDict = mt_readfile.standardize_metadata(parDict)

    return parDict


###################################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
