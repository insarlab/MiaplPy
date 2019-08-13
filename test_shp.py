#!/usr/bin/env python3
# Author: Sara Mirzaee

import numpy as np
import argparse
import os
import sys
import time
import glob
from minsar.objects import message_rsmas
from datetime import datetime
import minopy_utilities as mnp
from minsar.utils.process_utilities import create_or_update_template
from minsar.objects.auto_defaults import PathFind
import dask

pathObj = PathFind()

#######################


def main(iargs=None):
    """
        Divides the whole scene into patches for parallel processing
    """

    inps = command_line_parse(iargs)
    inps = create_or_update_template(inps)

    inps.minopy_dir = os.path.join(inps.work_dir, pathObj.minopydir)
    pathObj.patch_dir = inps.minopy_dir + '/PATCH'

    pathObj.slave_dir = os.path.join(inps.work_dir, pathObj.mergedslcdir)

    patch_list = glob.glob(inps.minopy_dir + '/PATCH*')
    patch_list = list(map(lambda x: x.split('/')[-1], patch_list))

    range_win = int(inps.template['minopy.range_window'])
    azimuth_win = int(inps.template['minopy.azimuth_window'])

    patch_rows = np.load(inps.minopy_dir + '/rowpatch.npy')
    patch_cols = np.load(inps.minopy_dir + '/colpatch.npy')

    patch_rows_overlap = np.load(inps.minopy_dir + '/rowpatch.npy')
    patch_rows_overlap[1, 0, 0] = patch_rows_overlap[1, 0, 0] - azimuth_win + 1
    patch_rows_overlap[0, 0, 1::] = patch_rows_overlap[0, 0, 1::] + azimuth_win + 1
    patch_rows_overlap[1, 0, 1::] = patch_rows_overlap[1, 0, 1::] - azimuth_win + 1
    patch_rows_overlap[1, 0, -1] = patch_rows_overlap[1, 0, -1] + azimuth_win - 1

    patch_cols_overlap = np.load(inps.minopy_dir + '/colpatch.npy')
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

    list_slv = os.listdir(pathObj.slave_dir)
    list_slv = [datetime.strptime(x, '%Y%m%d') for x in list_slv]
    list_slv = np.sort(list_slv)
    list_slv = [x.strftime('%Y%m%d') for x in list_slv]

    pathObj.out_dir = os.path.join(inps.work_dir, 'out_test_shp')
    if not os.path.exists(pathObj.out_dir):
        os.mkdir(pathObj.out_dir)

    Amplitude = np.zeros([n_line, width], np.float32)
    Phase = np.zeros([n_line, width], np.float32)
    Amplitude_ref = np.zeros([n_line, width], np.float32)
    Phase_ref = np.zeros([n_line, width], np.float32)
    SHP = np.zeros([n_line, width], np.int)

    RSLC = np.memmap(pathObj.out_dir + '/RSLC', dtype=np.complex64, mode='w+', shape=(len(list_slv), n_line, width))

    for t in range(len(list_slv)):

        for patch in patch_list:

            row = int(patch.split('PATCH')[-1].split('_')[0])
            col = int(patch.split('PATCH')[-1].split('_')[1])
            patch_name = pathObj.patch_dir + str(row) + '_' + str(col)

            row1 = patch_rows_overlap[0, 0, row]
            row2 = patch_rows_overlap[1, 0, row]
            col1 = patch_cols_overlap[0, 0, col]
            col2 = patch_cols_overlap[1, 0, col]

            patch_lines = patch_rows[1][0][row] - patch_rows[0][0][row]
            patch_samples = patch_cols[1][0][col] - patch_cols[0][0][col]

            f_row = (row1 - patch_rows[0, 0, row])
            l_row = patch_lines - (patch_rows[1, 0, row] - row2)
            f_col = (col1 - patch_cols[0, 0, col])
            l_col = patch_samples - (patch_cols[1, 0, col] - col2)

            count = np.load(patch_name + '/count.npy')
            n_image, line, sample = count[0], count[1], count[2]

            rslc = np.memmap(patch_name + '/RSLC', dtype=np.complex64, mode='r', shape=(n_image, line, sample))
            rslc_ref = np.memmap(patch_name + '/RSLC_ref', dtype=np.complex64, mode='r', shape=(n_image, line, sample))
            shp = np.memmap(patch_name + '/SHP', dtype='byte', mode='r', shape=(15 * 21, count[1], count[2]))

            Amplitude[row1:row2 + 1, col1:col2 + 1] = np.abs(rslc[t, f_row:l_row + 1, f_col:l_col + 1])

            Phase[row1:row2 + 1, col1:col2 + 1] = np.angle(rslc[t, f_row:l_row + 1, f_col:l_col + 1])

            Amplitude_ref[row1:row2 + 1, col1:col2 + 1] = np.abs(rslc_ref[t, f_row:l_row + 1, f_col:l_col + 1])

            Phase_ref[row1:row2 + 1, col1:col2 + 1] = np.angle(rslc_ref[t, f_row:l_row + 1, f_col:l_col + 1])
            RSLC[t:t+1, row1:row2 + 1, col1:col2 + 1] = rslc[t, f_row:l_row + 1, f_col:l_col + 1]

            if t == 0:
                SHP[row1:row2 + 1, col1:col2 + 1] = np.sum(shp, axis=0)[f_row:l_row + 1, f_col:l_col + 1]

        np.save(pathObj.out_dir + '/Amplitude' + str(t) + '.npy', Amplitude)
        np.save(pathObj.out_dir + '/Phase' + str(t) + '.npy', Phase)
        np.save(pathObj.out_dir + '/Amplitude_ref' + str(t) + '.npy', Amplitude_ref)
        np.save(pathObj.out_dir + '/Phase_ref' + str(t) + '.npy', Phase_ref)

        if t == 0:
            np.save(pathObj.out_dir + '/SHP.npy', SHP)

    np.save(pathObj.out_dir + '/AM.npy', np.abs(RSLC[:, 200:400, 700:1250]))
    np.save(pathObj.out_dir + '/PH.npy', np.angle(RSLC[:, 200:400, 700:1250]))

    return


def create_parser():
    """ Creates command line argument parser object. """
    parser = argparse.ArgumentParser(description='Divides the whole scene into patches for parallel processing')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('customTemplateFile', nargs='?', help='custom template with option settings.\n')

    return parser


def command_line_parse(args):
    """ Parses command line agurments into inps variable. """

    parser = create_parser()
    inps = parser.parse_args(args)
    return inps


if __name__ == '__main__':
    '''
    Divides the whole scene into patches for parallel processing.
    
    '''
    main()
