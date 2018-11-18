#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################
import numpy as np
import os
import isce
import isceobj
import sys
import argparse
import glob

sys.path.insert(0, os.getenv('RSMAS_ISCE'))
from _pysqsar_utilities import send_logger_squeesar, comp_matr
from rsmas_logging import loglevel
from dataset_template import Template

logger_write = send_logger_squeesar()

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

    logger_write.log(loglevel.INFO, os.path.basename(sys.argv[0]) + " " + sys.argv[1] + " " + sys.argv[2])
    inps.template = Template(inps.custom_template_file).get_options()

    project_name = os.path.basename(inps.custom_template_file).partition('.')[0]
    project_dir = os.getenv('SCRATCHDIR') + '/' + project_name
    slave_dir = project_dir + '/merged/SLC'
    sq_dir = project_dir + '/SqueeSAR'
    slc_list = os.listdir(slave_dir)
    
    patch_list = glob.glob(sq_dir+'/PATCH*')
    patch_list = list(map(x.split('/')[-1], for x in patch_list))
    
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

    g = inps.slc_dir
    image_ind = [i for (i, val) in enumerate(slc_list) if val == g.split('/')[0]]

    while g == inps.slc_dir:

        RSLCamp = np.zeros([int(n_line), int(width)])
        RSLCphase = np.zeros([int(n_line), int(width)])

        for patch in patch_list:

            row = int(patch.split('PATCH')[-1].split('_')[0])
            col = int(patch.split('PATCH')[-1].split('_')[1])
            row1 = patch_rows_overlap[0, 0, row]
            row2 = patch_rows_overlap[1, 0, row]
            col1 = patch_cols_overlap[0, 0, col]
            col2 = patch_cols_overlap[1, 0, col]
            amp = np.load(sq_dir + '/' + patch + '/Amplitude_ref.npy')
            ph = np.load(sq_dir + '/' + patch + '/Phase_ref.npy')

            f_row = (row1 - patch_rows[0, 0, row])
            l_row = np.size(amp, axis=1) - (patch_rows[1, 0, row] - row2)
            f_col = (col1 - patch_cols[0, 0, col])
            l_col = np.size(amp, axis=2) - (patch_cols[1, 0, col] - col2)

            RSLCamp[row1:row2 + 1, col1:col2 + 1] = \
                amp[image_ind, f_row:l_row + 1, f_col:l_col + 1]  # ampw.reshape(s1,s2)
            RSLCphase[row1:row2 + 1, col1:col2 + 1] = \
                ph[image_ind, f_row:l_row + 1, f_col:l_col + 1]  # phw.reshape(s1,s2)

        data = comp_matr(RSLCamp, RSLCphase)
        slc_file = slave_dir + '/' + inps.slc_dir

        #with open(slc_file + '.xml', 'r') as fid:
        #    xml_lines = fid.readlines()


        out_map = np.memmap(slc_file, dtype=np.complex64, mode='r+', shape=(n_line, width))
        out_map[:, :] = data

        out_img = isceobj.createSlcImage()
        out_img.setAccessMode('write')
        out_img.setFilename(slc_file)
        out_img.setWidth(width)
        out_img.setLength(n_line)
        out_img.renderVRT()
        out_img.renderHdr()

        del out_map

        #with open(slc_file + '.xml', 'w') as fid:
        #    for line in xml_lines:
        #        fid.write(line)

        g = 0


if __name__ == '__main__':
    """
        Overwrite filtered SLC images.
    """

    main()




