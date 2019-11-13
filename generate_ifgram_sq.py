#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################
import importlib
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import numpy as np
import os
import glob
import time
from isceobj.Image.IntImage import IntImage
from FilterAndCoherence import estCoherence, runFilter
from minopy.objects.arg_parser import MinoPyParser
from minopy.objects.slcStack import slcStack
import h5py
########################################


def main(iargs=None):
    """
        Overwrite filtered SLC images in Isce merged/SLC directory.
    """

    Parser = MinoPyParser(iargs, script='generate_ifgram')
    inps = Parser.parse()

    slc_file = os.path.join(inps.work_dir, 'inputs/slcStack.h5')
    slcObj = slcStack(slc_file)
    slcObj.open(print_msg=False)
    date_list = slcObj.get_date_list()

    patch_list = glob.glob(inps.work_dir+'/patches/patch*')
    patch_list = list(map(lambda x: x.split('/')[-1], patch_list))

    range_win = int(inps.range_win)
    azimuth_win = int(inps.azimuth_win)

    success = False
    while success is False:
        try:
            patch_rows = np.load(inps.work_dir + '/patches/rowpatch.npy')
            patch_cols = np.load(inps.work_dir + '/patches/colpatch.npy')
            success = True
        except:
            success = False

    patch_rows_overlap = np.zeros(np.shape(patch_rows), dtype=int)
    patch_rows_overlap[:, :, :] = patch_rows[:, :, :]
    patch_rows_overlap[1, 0, 0] = patch_rows_overlap[1, 0, 0] - azimuth_win + 1
    patch_rows_overlap[0, 0, 1::] = patch_rows_overlap[0, 0, 1::] + azimuth_win + 1
    patch_rows_overlap[1, 0, 1::] = patch_rows_overlap[1, 0, 1::] - azimuth_win + 1
    patch_rows_overlap[1, 0, -1] = patch_rows_overlap[1, 0, -1] + azimuth_win - 1

    patch_cols_overlap = np.zeros(np.shape(patch_cols), dtype=int)
    patch_cols_overlap[:, :, :] = patch_cols[:, :, :]
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

    if 'inputs' in inps.ifg_dir:

        if not os.path.isdir(inps.ifg_dir):
            os.mkdir(inps.ifg_dir)

        output_geo = inps.ifg_dir + '/geometryRadar.h5'
        Quality = np.zeros([n_line, width])
        SHP = np.zeros([n_line, width])

        doq = True
    else:

        ifgram = os.path.basename(inps.ifg_dir).split('_')
        master_ind = date_list.index(ifgram[0])
        slave_ind = date_list.index(ifgram[1])

        if not os.path.isdir(inps.ifg_dir):
            os.mkdir(inps.ifg_dir)

        output_int = inps.ifg_dir + '/fine.int'
        ifg = np.memmap(output_int, dtype=np.complex64, mode='w+', shape=(n_line, width))
        doq = False

    for patch in patch_list:

        row = int(patch.split('patch')[-1].split('_')[0])
        col = int(patch.split('patch')[-1].split('_')[1])
        row1 = patch_rows_overlap[0, 0, row]
        row2 = patch_rows_overlap[1, 0, row]
        col1 = patch_cols_overlap[0, 0, col]
        col2 = patch_cols_overlap[1, 0, col]

        patch_lines = patch_rows[1, 0, row] - patch_rows[0, 0, row]
        patch_samples = patch_cols[1, 0, col] - patch_cols[0, 0, col]

        f_row = row1 - patch_rows[0, 0, row]
        l_row = row2 - patch_rows[0, 0, row]
        f_col = col1 - patch_cols[0, 0, col]
        l_col = col2 - patch_cols[0, 0, col]

        if doq:
            qlty = np.memmap(inps.work_dir + '/patches/' + patch + '/quality',
                             dtype=np.float32, mode='r', shape=(patch_lines, patch_samples))
            Quality[row1:row2 + 1, col1:col2 + 1] = qlty[f_row:l_row + 1, f_col:l_col + 1]

            shp_p = np.memmap(inps.work_dir + '/patches/' + patch + '/shp',
                             dtype='byte', mode='r', shape=(range_win*azimuth_win, patch_lines, patch_samples))

            SHP[row1:row2 + 1, col1:col2 + 1] = np.sum(shp_p[:, f_row:l_row + 1, f_col:l_col + 1], axis=0)

        else:

            rslc_patch = np.memmap(inps.work_dir + '/patches/' + patch + '/rslc_ref',
                               dtype=np.complex64, mode='r', shape=(np.int(inps.n_image), patch_lines, patch_samples))
            ifg_patch = np.zeros([patch_lines, patch_samples])+0j

            master = rslc_patch[master_ind, :, :]
            slave = rslc_patch[slave_ind, :, :]

            for kk in range(0, patch_lines):
                ifg_patch[kk, f_col:l_col + 1] = master[kk, f_col:l_col + 1] * np.conj(slave[kk, f_col:l_col + 1])

            ifg[row1:row2 + 1, col1:col2 + 1] = ifg_patch[f_row:l_row + 1, f_col:l_col + 1]

    if doq:

        f = h5py.File(output_geo, 'a')
        if 'quality' in f.keys():
            del f['quality']

        ds = f.create_dataset('quality',
                              data=Quality,
                              dtype=np.float32,
                              chunks=True,
                              compression='lzf')
        ds.attrs['MODIFICATION_TIME'] = str(time.time())

        if 'shp' in f.keys():
            del f['shp']
        ds = f.create_dataset('shp',
                              data=SHP,
                              dtype=np.float32,
                              chunks=True,
                              compression='lzf')
        ds.attrs['MODIFICATION_TIME'] = str(time.time())

        f.close()

    else:

        ifg = None

        obj_int = IntImage()
        obj_int.setFilename(output_int)
        obj_int.setWidth(width)
        obj_int.setLength(n_line)
        obj_int.setAccessMode('READ')
        obj_int.renderHdr()
        obj_int.renderVRT()

        filt_file = inps.ifg_dir + '/filt_fine.int'
        filter_strength = 0.1

        runFilter(output_int, filt_file, filter_strength)

        cor_file = os.path.join(inps.ifg_dir, 'filt_fine.cor')

        estCoherence(filt_file, cor_file)


if __name__ == '__main__':
    """
        Overwrite filtered SLC images.
    """

    main()
