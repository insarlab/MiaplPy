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


def cropSLC(inps):

    ds = gdal.Open(inps.input + '.vrt', gdal.GA_ReadOnly)
    inp_file = ds.GetRasterBand(1).ReadAsArray()
    inp_file = inp_file[inps.first_row:inps.last_row, inps.first_col:inps.last_col]
    del ds

    out_map = np.memmap(inps.output, dtype=np.complex64, mode='write', shape=(inps.n_lines, inps.width))
    out_map[:, :] = inp_file

    out_img = isceobj.createSlcImage()
    out_img.setAccessMode('read')
    out_img.setFilename(inps.output)
    out_img.setWidth(inps.width)
    out_img.setLength(inps.n_lines)
    out_img.renderVRT()
    out_img.renderHdr()

    del out_map
    cmd = 'gdal_translate -of ENVI ' + inps.output + '.vrt ' + inps.output
    os.system(cmd)

    return inps

def cropQualitymap(inps):

    img = isceobj.createImage()
    img.load(inps.input + '.xml')
    bands = img.bands
    data_type = IML.NUMPY_type(img.dataType)
    scheme = img.scheme

    ds = gdal.Open(inps.input + '.vrt', gdal.GA_ReadOnly)
    inp_file = ds.GetRasterBand(1).ReadAsArray()
    inp_file = inp_file[inps.first_row:inps.last_row, inps.first_col:inps.last_col]

    if bands == 2:
        inp_file2 = ds.GetRasterBand(2).ReadAsArray()
        inp_file2 = inp_file2[inps.first_row:inps.last_row, inps.first_col:inps.last_col]
    del ds, img

    if not (inp_file.shape[0] == inps.n_lines and inp_file.shape[1] == inps.width):

        out_map = IML.memmap(inps.output, mode='write', nchannels=bands,
                             nxx=inps.width, nyy=inps.n_lines, scheme=scheme, dataType=data_type)

        if bands == 2:
            out_map.bands[0][0::, 0::] = inp_file
            out_map.bands[1][0::, 0::] = inp_file2
        else:
            out_map.bands[0][0::, 0::] = inp_file

        IML.renderISCEXML(inps.output, bands,
                          inps.n_lines, inps.width,
                          data_type, scheme)

        out_img = isceobj.createImage()
        out_img.load(inps.output + '.xml')
        out_img.imageType = data_type
        out_img.renderHdr()
        out_img.renderVRT()
        try:
            out_map.bands[0].base.base.flush()
        except:
            pass

        del out_map

        cmd = 'gdal_translate -of ENVI ' + inps.output + '.vrt ' + inps.output
        os.system(cmd)

    return inps



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

    crop_area = [val for val in inps.bbox.split()]
    if len(crop_area) != 4:
        raise Exception('Bbox should contain 4 floating point values')


    inps.first_row = np.int(crop_area[0])
    inps.last_row = np.int(crop_area[1])
    inps.first_col = np.int(crop_area[2])
    inps.last_col = np.int(crop_area[3])

    inps.n_lines = inps.last_row - inps.first_row
    inps.width = inps.last_col - inps.first_col


    if 'slc' in inps.output:

        inps = cropSLC(inps)
    else:
        inps = cropQualitymap(inps)


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
