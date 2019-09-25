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
    message_rsmas.log('.', os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1::]))

    inps = command_line_parse(iargs)
    inps = create_or_update_template(inps)
    inps.minopy_dir = os.path.join(inps.work_dir, pathObj.minopydir)
    pathObj.patch_dir = inps.minopy_dir + '/PATCH'

    pathObj.slave_dir = os.path.join(inps.work_dir, pathObj.mergedslcdir)

    pathObj.list_slv = os.listdir(pathObj.slave_dir)
    pathObj.list_slv = [datetime.strptime(x, '%Y%m%d') for x in pathObj.list_slv]
    pathObj.list_slv = np.sort(pathObj.list_slv)
    pathObj.list_slv = [x.strftime('%Y%m%d') for x in pathObj.list_slv]

    inps.range_window = int(inps.template['minopy.range_window'])
    inps.azimuth_window = int(inps.template['minopy.azimuth_window'])

    slc = mnp.read_image(pathObj.slave_dir + '/' + pathObj.list_slv[0] + '/' + pathObj.list_slv[0] + '.slc')  #
    pathObj.n_image = len(pathObj.list_slv)
    pathObj.lin = slc.shape[0]
    pathObj.sam = slc.shape[1]

    inps.patch_list = glob.glob(inps.minopy_dir + '/PATCH*')

    pathObj.patch_rows = np.load(inps.minopy_dir + '/rowpatch.npy')
    pathObj.patch_cols = np.load(inps.minopy_dir + '/colpatch.npy')

    test_patch(inps.patch_list)
    test_phase_linking(inps)

    return


def create_parser():
    """ Creates command line argument parser object. """
    parser = argparse.ArgumentParser(description='Divides the whole scene into patches for parallel processing')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('custom_template_file', nargs='?', help='custom template with option settings.\n')

    return parser


def command_line_parse(args):
    """ Parses command line agurments into inps variable. """

    parser = create_parser()
    inps = parser.parse_args(args)
    return inps


def test_patch(patch_list):

    data_name = pathObj.slave_dir + '/' + pathObj.list_slv[1] + '/' + pathObj.list_slv[1] + '.slc'
    slc = np.memmap(data_name, dtype=np.complex64, mode='r', shape=(pathObj.lin, pathObj.sam))
    slctest = np.zeros([pathObj.lin, pathObj.sam]) + 0j

    for patch in patch_list:

        patch_row, patch_col = patch.split('/PATCH')[-1].split('_')
        patch_row, patch_col = (int(patch_row), int(patch_col))
        patch_name = pathObj.patch_dir + str(patch_row) + '_' + str(patch_col)

        count = np.load(patch_name + '/count.npy')
        pathObj.n_image, line, sample = count[0], count[1], count[2]

        rslc = np.memmap(patch_name + '/RSLC', dtype=np.complex64, mode='r', shape=(pathObj.n_image, line, sample))
        slctest[pathObj.patch_rows[0][0][patch_row]:pathObj.patch_rows[1][0][patch_row],
                pathObj.patch_cols[0][0][patch_col]:pathObj.patch_cols[1][0][patch_col]] = rslc[1, :, :]

    if slc.all() == slctest.all():
        return print('create_patch: PASSED')
    else:
        return print('create_patch: FAILED')


def test_phase_linking(inps):

    patch_dir = inps.minopy_dir + '/PATCH0_1'
    count_dim = np.load(patch_dir + '/count.npy')
    n_image = count_dim[0]
    length = count_dim[1]
    width = count_dim[2]

    pathObj.rslc = np.memmap(patch_dir + '/RSLC', dtype=np.complex64, mode='r',
                             shape=(n_image, length, width))

    pathObj.rslc_ref = np.memmap(patch_dir + '/RSLC_ref', dtype=np.complex64, mode='r',
                             shape=(n_image, length, width))

    if pathObj.rslc.all() == pathObj.rslc_ref.all():
        return print('phase_linking: FAILED')
    else:
        return print('phase_linking: PASSED')


if __name__ == '__main__':
    '''
    Divides the whole scene into patches for parallel processing.

    '''
    main()
