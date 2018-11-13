#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################
import numpy as np
import os
import sys
import gdal
import argparse
import isce
import isceobj
from isceobj.Util.ImageUtil import ImageLib as IML
sys.path.insert(0, os.getenv('RSMAS_ISCE'))
import _pysqsar_utilities as squeesar
from rsmas_logging import loglevel
from dataset_template import Template
sys.path.insert(0, os.getenv('SENTINEL_STACK'))
from mergeBursts import multilook

logger  = squeesar.send_logger_squeesar()

##############################################################################
EXAMPLE = """example:
  crop_sentinel.py LombokSenAT156VV.template 
"""


def create_parser():
    """ Creates command line argument parser object. """

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, epilog=EXAMPLE)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('custom_template_file', nargs='?',
                        help='custom template with option settings.\n')

    return parser


def command_line_parse(args):
    """ Parses command line agurments into inps variable. """

    parser = create_parser()
    inps = parser.parse_args(args)

    return inps


def main(iargs=None):
    """
    Crops SLC images from Isce merged/SLC directory.
    """

    inps = command_line_parse(iargs)

    logger.log(loglevel.INFO, os.path.basename(sys.argv[0]) + " " + sys.argv[1])
    inps.template = Template(inps.custom_template_file).get_options()

    project_name = os.path.basename(inps.custom_template_file).partition('.')[0]
    project_dir = os.getenv('SCRATCHDIR')+ '/' + project_name
    slave_dir = project_dir + '/merged/SLC'
    geo_master_dir = project_dir + '/merged/geom_master'
    slc_list = os.listdir(slave_dir)
    

    lon_west = float(inps.template['lon1'])
    lon_east = float(inps.template['lon2'])
    lat_south = float(inps.template['lat1'])
    lat_north = float(inps.template['lat2'])

    azimuth_lks = inps.template['sentinelStack.azimuthLooks']
    range_lks = inps.template['sentinelStack.rangeLooks']
    
    if os.path.isfile(project_dir + '/merged/cropped.npy'):
      print('Already cropped')
    else:
        crop_area = np.array(squeesar.convert_geo2image_coord(geo_master_dir, lat_south, lat_north, lon_west, lon_east))
        first_row = np.int(crop_area[0])
        last_row = np.int(crop_area[1])
        first_col = np.int(crop_area[2])
        last_col = np.int(crop_area[3])

        n_lines = last_row - first_row
        width = last_col - first_col

    
        for slc in slc_list:
            filename = os.path.join(slave_dir, slc, slc + '.slc.full')

            ds = gdal.Open(filename + '.vrt', gdal.GA_ReadOnly)
            inp_file = ds.GetRasterBand(1).ReadAsArray()
            del ds

            out_map = np.memmap(filename, dtype=np.complex64, mode='r+', shape=(n_lines, width))
            out_map[:, :] = inp_file[first_row:last_row, first_col:last_col]

            out_img = isceobj.createSlcImage()
            out_img.setAccessMode('write')
            out_img.setFilename(filename)
            out_img.setWidth(width)
            out_img.setLength(n_lines)
            out_img.renderVRT()
            out_img.renderHdr()

            del out_map
            cmd = 'gdal_translate -of ENVI ' + filename + '.vrt ' + filename
            os.system(cmd)

            list_geo = ['hgt', 'lat', 'lon', 'los', 'shadowMask','incLocal']

        for t in list_geo:
            filename = os.path.join(geo_master_dir, t + '.rdr.full')

            img = isceobj.createImage()
            img.load(filename + '.xml')
            bands = img.bands
            data_type = IML.NUMPY_type(img.dataType)
            scheme = img.scheme

            ds = gdal.Open(filename + '.vrt', gdal.GA_ReadOnly)
            inp_file = ds.GetRasterBand(1).ReadAsArray()
            if bands == 2:
                inp_file2 = ds.GetRasterBand(2).ReadAsArray()
            del ds, img

            out_map = IML.memmap(filename, mode='r+', nchannels=bands,
                            nxx=width, nyy=n_lines, scheme=scheme, dataType=data_type)

            if bands == 2:
                out_map.bands[0][:, :] = inp_file[first_row:last_row, first_col:last_col]
                out_map.bands[1][:, :] = inp_file2[first_row:last_row, first_col:last_col]
            else:
                out_map.bands[0][:, :] = inp_file[first_row:last_row, first_col:last_col]

            IML.renderISCEXML(filename, bands,
                            n_lines, width,
                            data_type, scheme)

            out_img = isceobj.createImage()
            out_img.load(filename + '.xml')
            out_img.imageType = data_type
            out_img.renderHdr()
            try:
                out_map.bands[0].base.base.flush()
            except:
                pass

            cmd = 'gdal_translate -of ENVI ' + filename + '.vrt ' + filename
            os.system(cmd)

            multilook(filename, outname=filename.split('.full')[0],
                      alks=azimuth_lks, rlks=range_lks,
                      multilook_tool='isce', no_data=None)
        np.save(project_dir + '/merged/cropped.npy', 'True')

if __name__ == '__main__':
    '''
    Crop SLCs.
    '''
    main(sys.argv[:])
