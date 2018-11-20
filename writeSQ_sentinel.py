#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################
import numpy as np
import os
import sys
import argparse
import glob

sys.path.insert(0, os.getenv('RSMAS_ISCE'))
from _pysqsar_utilities import comp_matr
from dataset_template import Template

##############################################################################
EXAMPLE = """example:
  writeSQ_sentinel.py LombokSenAT156VV.template 20170823/20170823.slc.full
"""

def create_parser():
    """ Creates command line argument parser object. """

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, epilog=EXAMPLE)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('custom_template_file', nargs='?',
                        help='custom template with option settings.\n')
    parser.add_argument('-s','--slcdir', dest='slc_dir', type=str, required=True, help='slc file directory (date/date.slc.full)')


    return parser


def command_line_parse(args):
    """ Parses command line agurments into inps variable. """

    parser = create_parser()
    inps = parser.parse_args(args)

    return inps


########################################
def main(iargs=None):
    """
        Overwrite filtered SLC images in Isce merged/SLC directory.
    """

    inps = command_line_parse(iargs)

    inps.template = Template(inps.custom_template_file).get_options()

    project_name = os.path.basename(inps.custom_template_file).partition('.')[0]
    project_dir = os.getenv('SCRATCHDIR') + '/' + project_name
    slave_dir = project_dir + '/merged/SLC'
    sq_dir = project_dir + '/SqueeSAR'
    slc_list = os.listdir(slave_dir)
    
    patch_list = glob.glob(sq_dir+'/PATCH*')
    patch_list = list(map(lambda x: x.split('/')[-1], patch_list))
    
    range_win = int(inps.template['squeesar.wsizerange'])
    azimuth_win = int(inps.template['squeesar.wsizeazimuth'])

    patch_rows = np.load(sq_dir + '/rowpatch.npy')
    patch_cols = np.load(sq_dir + '/colpatch.npy')

    patch_rows_overlap = np.load(sq_dir + '/rowpatch.npy')
    patch_rows_overlap[1, 0, 0] = patch_rows_overlap[1, 0, 0] - azimuth_win + 1
    patch_rows_overlap[0, 0, 1::] = patch_rows_overlap[0, 0, 1::] + azimuth_win + 1
    patch_rows_overlap[1, 0, 1::] = patch_rows_overlap[1, 0, 1::] - azimuth_win + 1
    patch_rows_overlap[1, 0, -1] = patch_rows_overlap[1, 0, -1] + azimuth_win - 1

    patch_cols_overlap = np.load(sq_dir + '/colpatch.npy')
    patch_cols_overlap[1, 0, 0] = patch_cols_overlap[1, 0, 0] - range_win + 1
    patch_cols_overlap[0, 0, 1::] = patch_cols_overlap[0, 0, 1::] + range_win + 1
    patch_cols_overlap[1, 0, 1::] = patch_cols_overlap[1, 0, 1::] - range_win + 1
    patch_cols_overlap[1, 0, -1] = patch_cols_overlap[1, 0, -1] + range_win - 1

    first_row = patch_rows_overlap[0, 0, 0]
    last_row = patch_rows_overlap[1, 0, -1]
    first_col = patch_cols_overlap[0, 0, 0]
    last_col = patch_cols_overlap[1, 0, -1]

    n_line = last_row - first_row
    width = last_col - first_col

    slc_file = slave_dir + '/' + inps.slc_dir
    out_map = np.memmap(slc_file, dtype=np.complex64, mode='r+', shape=(n_line, width))


    image_ind = [i for (i, val) in enumerate(slc_list) if val == inps.slc_dir.split('/')[0]]


    for patch in patch_list:

        row = int(patch.split('PATCH')[-1].split('_')[0])
        col = int(patch.split('PATCH')[-1].split('_')[1])
        row1 = patch_rows_overlap[0, 0, row]
        row2 = patch_rows_overlap[1, 0, row]
        col1 = patch_cols_overlap[0, 0, col]
        col2 = patch_cols_overlap[1, 0, col]

        patch_lines = patch_rows[1][0][row] - patch_rows[0][0][row]
        patch_samples = patch_cols[1][0][col] - patch_cols[0][0][col]

        rslc_patch = np.memmap(sq_dir + '/' + patch  + '/RSLC_ref',
                               dtype=np.complex64, mode='r', shape=(len(slc_list), patch_lines, patch_samples))


        f_row = (row1 - patch_rows[0, 0, row])
        l_row = patch_lines - (patch_rows[1, 0, row] - row2)
        f_col = (col1 - patch_cols[0, 0, col])
        l_col = patch_samples - (patch_cols[1, 0, col] - col2)

        out_map[row1:row2 + 1, col1:col2 + 1] = \
            rslc_patch[image_ind, f_row:l_row + 1, f_col:l_col + 1]


    del out_map


if __name__ == '__main__':
    """
        Overwrite filtered SLC images.
    """

    main()




