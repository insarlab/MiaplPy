#!/usr/bin/env python3
# Author: Sara Mirzaee

import numpy as np
import argparse
import os
import sys
import time
sys.path.insert(0, os.getenv('SQUEESAR'))
import _pysqsar_utilities as pysq
#from dask import delayed, compute
#######################

def create_parser():
    """ Creates command line argument parser object. """
    parser = argparse.ArgumentParser(description='Divides the whole scene into patches for parallel processing')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('-s', '--slc_dir', dest='slc_dir', type=str, required=True, help='Input SLC directory')
    parser.add_argument('-q', '--squeesar_dir', dest='output_dir', type=str, required=True, help='Patch directory')
    parser.add_argument('-p', '--patch_size', dest='patch_size', type=str, default='200', help='Patch size')
    parser.add_argument('-r', '--range_window', dest='range_win', type=str, default='21'
                        , help='SHP searching window size in range direction. -- Default : 21')
    parser.add_argument('-a', '--azimuth_window', dest='azimuth_win', type=str, default='15'
                        , help='SHP searching window size in azimuth direction. -- Default : 15')

    return parser


def command_line_parse(args):
    """ Parses command line agurments into inps variable. """

    parser = create_parser()
    inps = parser.parse_args(args)
    return inps


def create_patch(inps, name):
    patch_row, patch_col = name.split('_')
    patch_row, patch_col = (int(patch_row), int(patch_col))
    patch_name = inps.patch_dir + str(patch_row) + '_' + str(patch_col)

    line = inps.patch_rows[1][0][patch_row] - inps.patch_rows[0][0][patch_row]
    sample = inps.patch_cols[1][0][patch_col] - inps.patch_cols[0][0][patch_col]

    a = True
    if a==True: #not os.path.isfile(patch_name + '/count.npy'):
        if not os.path.isdir(patch_name):
            os.mkdir(patch_name)

        rslc = np.memmap(patch_name + '/RSLC', dtype=np.complex64, mode='w+', shape=(inps.n_image, line, sample))

        count = 0
    
        for dirs in inps.list_slv:
            data_name = inps.slave_dir + '/' + dirs + '/' + dirs + '.slc'
            slc = np.memmap(data_name, dtype=np.complex64, mode='r', shape=(inps.lin, inps.sam))

            rslc[count, :, :] = slc[inps.patch_rows[0][0][patch_row]:inps.patch_rows[1][0][patch_row],
                                inps.patch_cols[0][0][patch_col]:inps.patch_cols[1][0][patch_col]]
            count += 1
            del slc
        del rslc

        np.save(patch_name + '/count.npy', [inps.n_image,line,sample])
    else:
        print('Next patch...')
    return "PATCH" + str(patch_row) + '_' + str(patch_col) + " is created"


########################################################
def main(iargs=None):
    """
        Divides the whole scene into patches for parallel processing
    """

    inps = command_line_parse(iargs)
    inps.slave_dir = inps.slc_dir
    inps.sq_dir = inps.output_dir
    inps.patch_dir = inps.sq_dir + '/PATCH'
    inps.list_slv = os.listdir(inps.slave_dir)

    inps.range_win = int(inps.range_win)
    inps.azimuth_win = int(inps.azimuth_win)

    if not os.path.isdir(inps.sq_dir):
        os.mkdir(inps.sq_dir)

    slc = pysq.read_image(inps.slave_dir + '/' + inps.list_slv[0] + '/' + inps.list_slv[0] + '.slc')  #
    inps.n_image = len(inps.list_slv)
    inps.lin = slc.shape[0]
    inps.sam = slc.shape[1]
    del slc

    inps.patch_rows, inps.patch_cols, inps.patch_list = \
        pysq.patch_slice(inps.lin, inps.sam, inps.azimuth_win, inps.range_win, np.int(inps.patch_size))

    np.save(inps.sq_dir + '/rowpatch.npy', inps.patch_rows)
    np.save(inps.sq_dir + '/colpatch.npy', inps.patch_cols)

    time0 = time.time()
    if os.path.isfile(inps.sq_dir + '/flag.npy'):
        print('patchlist exist')
    else:
        for patch in inps.patch_list:
            create_patch(inps, patch)

        # values = [delayed(create_patch)(inps, x) for x in inps.patch_list]
        # compute(*values, scheduler='processes')

    np.save(inps.sq_dir + '/flag.npy', 'patchlist_created')
    timep = time.time() - time0

    print("Done Creating PATCH. time:{} min".format(timep/60))


if __name__ == '__main__':
    '''
    Divides the whole scene into patches for parallel processing. 
 
    '''
    main()