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
from FilterAndCoherence import estCoherence


##############################################################################
def create_parser():
    """ Creates command line argument parser object. """

    parser = argparse.ArgumentParser(description='Generate interferogram, coherence and quality map from '
                                                 'phase linking inversion outputs')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('-sq','--squeesar_dir', dest='squeesar_dir', type=str, required=True,
                        help='squeesar directory (inversion results)')
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
                        help='Phase linking method ["EVD","EMI","PTA","sequential_EVD","sequential_EMI","sequential_PTA"] '
                             'default: sequential EMI.')

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

    
    patch_list = glob.glob(inps.squeesar_dir+'/PATCH*')
    patch_list = list(map(lambda x: x.split('/')[-1], patch_list))
    
    
    range_win = int(inps.range_win)
    azimuth_win = int(inps.azimuth_win)
    

    patch_rows = np.load(inps.squeesar_dir + '/rowpatch.npy')
    patch_cols = np.load(inps.squeesar_dir + '/colpatch.npy')

    patch_rows_overlap = np.load(inps.squeesar_dir + '/rowpatch.npy')
    patch_rows_overlap[1, 0, 0] = patch_rows_overlap[1, 0, 0] - azimuth_win + 1
    patch_rows_overlap[0, 0, 1::] = patch_rows_overlap[0, 0, 1::] + azimuth_win + 1
    patch_rows_overlap[1, 0, 1::] = patch_rows_overlap[1, 0, 1::] - azimuth_win + 1
    patch_rows_overlap[1, 0, -1] = patch_rows_overlap[1, 0, -1] + azimuth_win - 1

    patch_cols_overlap = np.load(inps.squeesar_dir + '/colpatch.npy')
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



    if 'geom_master' in inps.output_dir:

        if not os.path.isdir(inps.output_dir):
            os.mkdir(inps.output_dir)

        outputq = inps.output_dir + '/Quality.rdr'
        Quality = IML.memmap(outputq, mode='write', nchannels=1,
                             nxx=width, nyy=n_line, scheme='BIL', dataType='f')
        doq = True
    else:

        if not os.path.isdir(inps.output_dir):
            os.mkdir(inps.output_dir)

        outputint = inps.output_dir + '/filt_fine.int'
        ifg = np.memmap(outputint , dtype=np.complex64, mode='w+', shape=(n_line, width))
        doq = False


    for patch in patch_list:

        row = int(patch.split('PATCH')[-1].split('_')[0])
        col = int(patch.split('PATCH')[-1].split('_')[1])
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

        if doq:
            qlty = np.memmap(inps.squeesar_dir + '/' + patch  + '/quality',
                             dtype=np.float32, mode='r', shape=(patch_lines, patch_samples))
            Quality.bands[0][row1:row2 + 1, col1:col2 + 1] = qlty[f_row:l_row + 1, f_col:l_col + 1]
        else:
            rslc_patch = np.memmap(inps.squeesar_dir + '/' + patch  + '/RSLC_ref',
                               dtype=np.complex64, mode='r', shape=(np.int(inps.n_image), patch_lines, patch_samples))
            ifg_patch = np.zeros([patch_lines, patch_samples])+0j
            master = rslc_patch[0,:,:]
            slave = rslc_patch[np.int(inps.ifg_index),:,:]


            for kk in range(0, patch_lines):
                ifg_patch[kk, 0:patch_samples + 1] = master[kk, 0:patch_samples + 1] * np.conj(slave[kk, 0:patch_samples + 1])

            ifg[row1:row2 + 1, col1:col2 + 1] = ifg_patch[f_row:l_row + 1, f_col:l_col + 1]


    if doq:

        Quality = None

        IML.renderISCEXML(outputq, 1, n_line, width, 'f', 'BIL')

        out_img = isceobj.createImage()
        out_img.load(outputq + '.xml')
        out_img.imageType = 'f'
        out_img.renderHdr()
        try:
            out_map.bands[0].base.base.flush()
        except:
            pass

        cmd = 'gdal_translate -of ENVI -co INTERLEAVE=BIL ' + outputq + '.vrt ' + outputq
        os.system(cmd)


        ds = gdal.Open(outputq, gdal.GA_ReadOnly)

        ds.SetMetadata({'plmethod': inps.plmethod})

        ds = None


    else:

        ifg = None

        obj_int = isceobj.createIntImage()
        obj_int.setFilename(outputint)
        obj_int.setWidth(width)
        obj_int.setLength(n_line)
        obj_int.setAccessMode('READ')
        obj_int.renderHdr()
        obj_int.renderVRT()

        corfile = os.path.join(inps.output_dir,'filt_fine.cor')


        estCoherence(outputint, corfile)



if __name__ == '__main__':
    """
        Overwrite filtered SLC images.
    """

    main()




