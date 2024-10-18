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
import argparse
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
  prep_slc_gamma.py -s ../rslc/20*/*slc -r ../geometry/*.dem -g ../geometry/*.utm_to_rdc   #for slc and geometry files
  """


def create_parser():
    """Command line parser."""
    parser = argparse.ArgumentParser(description='Prepare ISCE metadata files.',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)
    parser.add_argument('-s', '--slc-files', nargs='+', dest='slc_files', type=str, default=None,
                        help='List of resampled slc files that will be used in miaplpy\n'
                             'e.g.: ../rslc/2018*/20*slc')
    parser.add_argument('-r', '--radar-geom', nargs='+', dest='radar_geom_files', type=str, default=None,
                        help=' Geometry files in radar coosdinates. e.g.: ../geometry/sim_20170223.rdc.dem')
    parser.add_argument('-g', '--geo-geom', nargs='+', dest='geo_geom_files', type=str, default=None,
                        help=' Geometry files in geo coosdinates. e.g.: ../geometry/sim_20170223.utm_to_rdc ')
    # TODO: implement overwrite mode
    # parser.add_argument('--force', dest='update_mode', action='store_false',
    #                     help='Force to overwrite all .rsc metadata files. Not implemented yet!')
    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args()
    inps = vars(inps)
    if all(not i for i in [inps['slc_files'], inps['radar_geom_files'], inps['geo_geom_files']]):
        parser.print_usage()
        raise SystemExit('error: at least one of the following arguments are required: -s, -g, -m')
    return inps


def extract_metadata4slc(fname, update_mode=False):
    print('preparing RSC file for ', fname)
    slc_metadata = read_attribute(fname, metafile_ext='.par')
    # update date yyyy -> yyyymmgg
    # TODO: update read_attribute to correctly read date from par file
    atr = read_gamma_slc_par(fname + '.par', skiprows=2)
    slc_metadata['DATE'] = atr['date']
    slc_metadata.pop('date', None)

    get_lalo_ref(fname + '.par', slc_metadata)

    # write .rsc file
    rsc_file = fname + '.rsc'
    update_mode = True
    writefile.write_roipac_rsc(slc_metadata, rsc_file,
                               update_mode=update_mode,
                               print_msg=False)
    return


def main(iargs=None):
    inps = cmd_line_parse(iargs)

    if inps['geo_geom_files']:
        fnames = inps['geo_geom_files']
        for fname in fnames:
            if fname.endswith(('UTM_TO_RDC', 'inc')):  # TODO: add other extentions
                extract_metadata4geometry_geo(fname)
            else:
                raise Exception(f'File {fname} not supported. Use UTM_TO_RDC or inc.')

    if inps['radar_geom_files']:
        fnames = inps['radar_geom_files']
        for fname in fnames:
            if fname.endswith(('rdc.dem')):
                extract_metadata4geometry_radar(fname)
            else:
                raise Exception(f'File {fname} not supported. Use rdc.dem.')

    if inps['slc_files']:
        fnames = inps['slc_files']
        for fname in fnames:
            if fname.endswith(('.slc', '.rslc')):
                extract_metadata4slc(fname)
            else:
                raise Exception(f'File {fname} not supported. Use .slc os .rslc.')
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
