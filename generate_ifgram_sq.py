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


##############################################################################
def create_parser():
    """ Creates command line argument parser object. """

    parser = argparse.ArgumentParser(description='Generate interferogram, coherence and quality map from '
                                                 'phase linking inversion outputs')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('-mp', '--minopy_dir', dest='minopy_dir', type=str, required=True,
                        help='minopy directory (inversion results)')
    parser.add_argument('-i', '--ifg_dir', dest='output_dir', type=str, required=True, help='interferogram directory')
    parser.add_argument('-x', '--ifg_index', dest='ifg_index', type=str, required=True, help='interferogram index in 3D array (inversion results)')
    parser.add_argument('-r', '--range_window', dest='range_win', type=str, default='21'
                        , help='SHP searching window size in range direction. -- Default : 21')
    parser.add_argument('-a', '--azimuth_window', dest='azimuth_win', type=str, default='15'
                        , help='SHP searching window size in azimuth direction. -- Default : 15')
    parser.add_argument('-q', '--acquisition_number', dest='n_image', type=str, default='20', help='number of images acquired')
    parser.add_argument('-A', '--azimuth_looks', type=str, dest='azimuth_looks', default=3, help='azimuth looks')
    parser.add_argument('-R', '--range_looks', type=str, dest='range_looks', default=9, help='range looks')
    parser.add_argument('-m', '--plmethod', dest='plmethod', type=str, default='sequential_EMI',
                        help='Phase linking method ["EVD","EMI","PTA","sequential_EVD","sequential_EMI",'
                             '"sequential_PTA"] default: sequential EMI.')

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

    patch_list = glob.glob(inps.minopy_dir+'/PATCH*')
    patch_list = list(map(lambda x: x.split('/')[-1], patch_list))

    range_win = int(inps.range_win)
    azimuth_win = int(inps.azimuth_win)

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

    print(patch_rows_overlap, patch_cols_overlap)

    first_row = patch_rows_overlap[0, 0, 0]
    last_row = patch_rows_overlap[1, 0, -1]
    first_col = patch_cols_overlap[0, 0, 0]
    last_col = patch_cols_overlap[1, 0, -1]

    n_line = last_row - first_row
    width = last_col - first_col

    if 'geom_master' in inps.output_dir:

        if not os.path.isdir(inps.output_dir):
            os.mkdir(inps.output_dir)

        output_quality = inps.output_dir + '/Quality.rdr'
        output_shp = inps.output_dir + '/SHP.rdr'
        Quality = IML.memmap(output_quality, mode='write', nchannels=1,
                             nxx=width, nyy=n_line, scheme='BIL', dataType='f')
        SHP = IML.memmap(output_shp, mode='write', nchannels=1,
                             nxx=width, nyy=n_line, scheme='BIL', dataType='int')
        doq = True
    else:

        if not os.path.isdir(inps.output_dir):
            os.mkdir(inps.output_dir)

        output_int = inps.output_dir + '/fine.int'
        ifg = np.memmap(output_int, dtype=np.complex64, mode='w+', shape=(n_line, width))
        doq = False

    for patch in patch_list:

        row = int(patch.split('PATCH')[-1].split('_')[0])
        col = int(patch.split('PATCH')[-1].split('_')[1])
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
            qlty = np.memmap(inps.minopy_dir + '/' + patch + '/quality',
                             dtype=np.float32, mode='r', shape=(patch_lines, patch_samples))
            Quality.bands[0][row1:row2 + 1, col1:col2 + 1] = qlty[f_row:l_row + 1, f_col:l_col + 1]

            shp_p = np.memmap(inps.minopy_dir + '/' + patch + '/SHP',
                             dtype='byte', mode='r', shape=(range_win*azimuth_win, patch_lines, patch_samples))

            SHP.bands[0][row1:row2 + 1, col1:col2 + 1] = np.sum(shp_p[:, f_row:l_row + 1, f_col:l_col + 1], axis=0)

        else:
            rslc_patch = np.memmap(inps.minopy_dir + '/' + patch + '/RSLC_ref',
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

        filt_file = inps.output_dir + '/filt_fine.int'
        filter_strength = 0.5

        runFilter(output_int, filt_file, filter_strength)

        cor_file = os.path.join(inps.output_dir, 'filt_fine.cor')

        estCoherence(filt_file, cor_file)


if __name__ == '__main__':
    """
        Overwrite filtered SLC images.
    """

    main()
