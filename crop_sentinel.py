#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################
import numpy as np
import os
import sys
import gdal
import argparse
import shutil
import isce
import isceobj
from isceobj.Util.ImageUtil import ImageLib as IML
from mergeBursts import multilook


##############################################################################

def create_parser():
    """ Creates command line argument parser object. """

    parser = argparse.ArgumentParser(description='Crops the scene given bounding box in lat/lon')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,help='Input SLC')
    parser.add_argument('-o', '--output', dest='output', type=str, required=True,help='Output cropped SLC')
    parser.add_argument('-b', '--bbox', dest='bbox', type=str, default=None,
                        help="row/col Bounding frow lastrow firstcol lastcol. "
                             "-- Example : '1000 2000 14000 15000' " "-- crop area")

    parser.add_argument('-m', '--multilook', action='store_true', dest='multilook', default=False,
                    help = 'Multilook the merged products. True or False')
    parser.add_argument('-z', '--azimuth_looks', dest='azimuthLooks', type=str, default='3'
                        , help='Number of looks in azimuth. -- Default : 3')

    parser.add_argument('-r', '--range_looks', dest='rangeLooks', type=str, default='9'
                        , help='Number of looks in range. -- Default : 9')

    parser.add_argument('-t', '--multilook_tool', dest='multilook_tool', type=str, default='gdal'
                        , help='Multilooking tool: gdal or isce')


    return parser


def command_line_parse(iargs=None):
    """ Parses command line agurments into inps variable. """

    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    return inps



def main(iargs=None):
    """
    Crops SLC images from Isce merged/SLC directory.
    """

    inps = command_line_parse(iargs)

    crop_area = inps.bbox.split('/')

    first_row = np.int(crop_area[0][1:-1])
    last_row = np.int(crop_area[1][1:-1])
    first_col = np.int(crop_area[2][1:-1])
    last_col = np.int(crop_area[3][1:-1])

    n_lines = last_row - first_row
    width = last_col - first_col


    if inps.output.split('.')[1]=='slc':

        ds = gdal.Open(inps.input + '.vrt', gdal.GA_ReadOnly)
        inp_file = ds.GetRasterBand(1).ReadAsArray()
        del ds

        out_map = np.memmap(inps.output, dtype=np.complex64, mode='write', shape=(n_lines, width))
        out_map[:, :] = inp_file[first_row:last_row, first_col:last_col]

        out_img = isceobj.createSlcImage()
        out_img.setAccessMode('read')
        out_img.setFilename(inps.output)
        out_img.setWidth(width)
        out_img.setLength(n_lines)
        out_img.renderVRT()
        out_img.renderHdr()

        del out_map
        cmd = 'gdal_translate -of ENVI ' + inps.output + '.vrt ' + inps.output
        os.system(cmd)

    else:


        img = isceobj.createImage()
        img.load(inps.input + '.xml')
        bands = img.bands
        data_type = IML.NUMPY_type(img.dataType)
        scheme = img.scheme

        ds = gdal.Open(inps.input + '.vrt', gdal.GA_ReadOnly)
        inp_file = ds.GetRasterBand(1).ReadAsArray()

        if bands == 2:
            inp_file2 = ds.GetRasterBand(2).ReadAsArray()
        del ds, img

        if not (inp_file.shape[0] == n_lines and inp_file.shape[1] == width):

            out_map = IML.memmap(inps.output, mode='r+', nchannels=bands,
                            nxx=width, nyy=n_lines, scheme=scheme, dataType=data_type)

            if bands == 2:
                out_map.bands[0][:, :] = inp_file[first_row:last_row, first_col:last_col]
                out_map.bands[1][:, :] = inp_file2[first_row:last_row, first_col:last_col]
            else:
                out_map.bands[0][:, :] = inp_file[first_row:last_row, first_col:last_col]

            IML.renderISCEXML(inps.output, bands,
                            n_lines, width,
                            data_type, scheme)

            out_img = isceobj.createImage()
            out_img.load(inps.output + '.xml')
            out_img.imageType = data_type
            out_img.renderHdr()
            try:
                out_map.bands[0].base.base.flush()
            except:
                pass

            cmd = 'gdal_translate -of ENVI ' + filename + '.vrt ' + filename
            os.system(cmd)


    if inps.multilook:
        print('multilooking')
        multilook(inps.input, outname=inps.output+'.ml',
                  alks=inps.azimuthLooks, rlks=inps.rangeLooks,
                  multilook_tool=inps.multilook_tool, no_data=None)


if __name__ == '__main__':
    '''
    Crop SLCs.
    '''
    main()
