#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################
import numpy as np
import os
import sys
import argparse
import glob
import gdal
import isce
import isceobj
from isceobj.Util.ImageUtil import ImageLib as IML
from FilterAndCoherence import estCoherence, runFilter
from minopy.objects.arg_parser import MinoPyParser

########################################


def main(iargs=None):
    """
        Overwrite filtered SLC images in Isce merged/SLC directory.
    """

    Parser = MinoPyParser(iargs, script='generate_ifgram')
    inps = Parser.parse()

    patch_list = glob.glob(inps.work_dir+'/PATCH*')
    patch_list = list(map(lambda x: x.split('/')[-1], patch_list))

    range_win = int(inps.range_win)
    azimuth_win = int(inps.azimuth_win)

    patch_rows = np.load(inps.work_dir + '/rowpatch.npy')
    patch_cols = np.load(inps.work_dir + '/colpatch.npy')

    patch_rows_overlap = np.load(inps.work_dir + '/rowpatch.npy')
    patch_rows_overlap[1, 0, 0] = patch_rows_overlap[1, 0, 0] - azimuth_win + 1
    patch_rows_overlap[0, 0, 1::] = patch_rows_overlap[0, 0, 1::] + azimuth_win + 1
    patch_rows_overlap[1, 0, 1::] = patch_rows_overlap[1, 0, 1::] - azimuth_win + 1
    patch_rows_overlap[1, 0, -1] = patch_rows_overlap[1, 0, -1] + azimuth_win - 1

    patch_cols_overlap = np.load(inps.work_dir + '/colpatch.npy')
    patch_cols_overlap[1, 0, 0] = patch_cols_overlap[1, 0, 0] - range_win + 1
    patch_cols_overlap[0, 0, 1::] = patch_cols_overlap[0, 0, 1::] + range_win + 1
    patch_cols_overlap[1, 0, 1::] = patch_cols_overlap[1, 0, 1::] - range_win + 1
    patch_cols_overlap[1, 0, -1] = patch_cols_overlap[1, 0, -1] + range_win - 1

    print(patch_rows_overlap, patch_cols_overlap)

    first_row = patch_rows_overlap[0, 0, 0]
    last_row = patch_rows_overlap[1, 0, -1]
    first_col = patch_cols_overlap[0, 0, 0]
    last_col = patch_cols_overlap[1, 0, -1]

    n_line = last_row - first_row
    width = last_col - first_col

    if 'geom_master' in inps.ifg_dir:

        if not os.path.isdir(inps.ifg_dir):
            os.mkdir(inps.ifg_dir)

        output_quality = inps.ifg_dir + '/Quality.rdr'
        output_shp = inps.ifg_dir + '/SHP.rdr'
        Quality = IML.memmap(output_quality, mode='write', nchannels=1,
                             nxx=width, nyy=n_line, scheme='BIL', dataType='f')
        SHP = IML.memmap(output_shp, mode='write', nchannels=1,
                             nxx=width, nyy=n_line, scheme='BIL', dataType='int')
        doq = True
    else:

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
            qlty = np.memmap(inps.work_dir + '/' + patch + '/quality',
                             dtype=np.float32, mode='r', shape=(patch_lines, patch_samples))
            Quality.bands[0][row1:row2 + 1, col1:col2 + 1] = qlty[f_row:l_row + 1, f_col:l_col + 1]

            shp_p = np.memmap(inps.work_dir + '/' + patch + '/shp',
                             dtype='byte', mode='r', shape=(range_win*azimuth_win, patch_lines, patch_samples))

            SHP.bands[0][row1:row2 + 1, col1:col2 + 1] = np.sum(shp_p[:, f_row:l_row + 1, f_col:l_col + 1], axis=0)

        else:
            rslc_patch = np.memmap(inps.work_dir + '/' + patch + '/rslc_ref',
                               dtype=np.complex64, mode='r', shape=(np.int(inps.n_image), patch_lines, patch_samples))
            ifg_patch = np.zeros([patch_lines, patch_samples])+0j
            master = rslc_patch[0, :, :]
            slave = rslc_patch[np.int(inps.ifg_index) + 1, :, :]

            for kk in range(0, patch_lines):
                ifg_patch[kk, f_col:l_col + 1] = master[kk, f_col:l_col + 1] * np.conj(slave[kk, f_col:l_col + 1])

            ifg[row1:row2 + 1, col1:col2 + 1] = ifg_patch[f_row:l_row + 1, f_col:l_col + 1]

    if doq:

        Quality = None

        IML.renderISCEXML(output_quality, 1, n_line, width, 'f', 'BIL')

        out_img = isceobj.createImage()
        out_img.load(output_quality + '.xml')
        out_img.imageType = 'f'
        out_img.renderHdr()
        try:
            out_img.bands[0].base.base.flush()
        except:
            pass

        cmd = 'gdal_translate -of ENVI -co INTERLEAVE=BIL ' + output_quality + '.vrt ' + output_quality
        os.system(cmd)

        ## SHP

        SHP = None

        IML.renderISCEXML(output_shp, 1, n_line, width, 'int', 'BIL')
        out_img = isceobj.createImage()
        out_img.load(output_shp + '.xml')
        out_img.imageType = 'int'
        out_img.renderHdr()
        try:
            out_img.bands[0].base.base.flush()
        except:
            pass

        cmd = 'gdal_translate -of ENVI -co INTERLEAVE=BIL ' + output_shp + '.vrt ' + output_shp
        os.system(cmd)

    else:

        ifg = None

        obj_int = isceobj.createIntImage()
        obj_int.setFilename(output_int)
        obj_int.setWidth(width)
        obj_int.setLength(n_line)
        obj_int.setAccessMode('READ')
        obj_int.renderHdr()
        obj_int.renderVRT()

        filt_file = inps.ifg_dir + '/filt_fine.int'
        filter_strength = 0.5

        runFilter(output_int, filt_file, filter_strength)

        cor_file = os.path.join(inps.ifg_dir, 'filt_fine.cor')

        estCoherence(filt_file, cor_file)


if __name__ == '__main__':
    """
        Overwrite filtered SLC images.
    """

    main()
