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
import time
import glob
from minsar.objects import message_rsmas
from isceobj.Util.ImageUtil import ImageLib as IML
from mergeBursts import multilook
from minsar.utils.process_utilities import create_or_update_template
from minopy_utilities import convert_geo2image_coord, patch_slice
from minsar.objects.auto_defaults import PathFind
import dask

pathObj = PathFind()
##############################################################################


def create_parser():
    """ Creates command line argument parser object. """

    parser = argparse.ArgumentParser(description='Crops the scene given cropping box in lat/lon (from template)')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('customTemplateFile', nargs='?',
                        help='custom template with option settings.\n')
    return parser


def command_line_parse(iargs=None):
    """ Parses command line agurments into inps variable. """

    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    return inps


def cropSLC(data):
    '''crop SLC images'''

    (input_file, output_file) = data

    ds = gdal.Open(input_file + '.vrt', gdal.GA_ReadOnly)
    inp_file = ds.GetRasterBand(1).ReadAsArray()[pathObj.first_row:pathObj.last_row, pathObj.first_col:pathObj.last_col]
    data_type = inp_file.dtype.type
    del ds

    out_map = np.memmap(output_file, dtype=data_type, mode='write', shape=(pathObj.n_lines, pathObj.width))
    out_map[:, :] = inp_file[:, :]

    IML.renderISCEXML(output_file, 1, pathObj.n_lines, pathObj.width, IML.NUMPY_type(str(inp_file.dtype)), 'BIL')

    out_img = isceobj.createSlcImage()
    out_img.load(output_file + '.xml')
    out_img.renderVRT()
    out_img.renderHdr()

    del inp_file, out_map

    return output_file


def cropQualitymap(data):
    '''crop geometry files: lat, lon, ...'''

    (input_file, output_file) = data

    img = isceobj.createImage()
    img.load(input_file + '.xml')
    bands = img.bands
    data_type = IML.NUMPY_type(img.dataType)
    scheme = img.scheme

    ds = gdal.Open(input_file + '.vrt', gdal.GA_ReadOnly)
    inp_file = ds.GetRasterBand(1).ReadAsArray()[pathObj.first_row:pathObj.last_row, pathObj.first_col:pathObj.last_col]

    if bands == 2:
        inp_file2 = ds.GetRasterBand(2).ReadAsArray()[pathObj.first_row:pathObj.last_row, pathObj.first_col:pathObj.last_col]

    del ds, img

    out_map = IML.memmap(output_file, mode='write', nchannels=bands,
                         nxx=pathObj.width, nyy=pathObj.n_lines, scheme=scheme, dataType=data_type)

    if bands == 2:
        out_map.bands[0][:, :] = inp_file[:, :]
        out_map.bands[1][:, :] = inp_file2[:, :]
    else:
        out_map.bands[0][:, :] = inp_file[:, :]

    IML.renderISCEXML(output_file, bands,
                      pathObj.n_lines, pathObj.width,
                      data_type, scheme)

    out_img = isceobj.createImage()
    out_img.load(output_file + '.xml')
    out_img.renderHdr()
    out_img.renderVRT()
    try:
        out_map.bands[0].base.base.flush()
    except:
        pass

    del out_map, inp_file
    if bands == 2:
        del inp_file2

    return output_file


if __name__ == '__main__':
    '''
    Crop SLCs and geometry.
    '''
    message_rsmas.log(os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1::]))
    
    inps = command_line_parse()
    inps = create_or_update_template(inps)
    inps.geo_master = os.path.join(inps.work_dir, pathObj.geomasterdir)

    slc_list = os.listdir(os.path.join(inps.work_dir, pathObj.mergedslcdir))
    slc_list = [os.path.join(inps.work_dir, pathObj.mergedslcdir, x, x + '.slc.full') for x in slc_list]

    meta_data = pathObj.get_geom_master_lists()

    cbox = [val for val in inps.cropbox.split()]
    if len(cbox) != 4:
        raise Exception('Bbox should contain 4 floating point values')

    crop_area = np.array(
        convert_geo2image_coord(inps.geo_master, np.float32(cbox[0]), np.float32(cbox[1]),
                                np.float32(cbox[2]), np.float32(cbox[3])))

    pathObj.first_row = np.int(crop_area[0])
    pathObj.last_row = np.int(crop_area[1])
    pathObj.first_col = np.int(crop_area[2])
    pathObj.last_col = np.int(crop_area[3])

    pathObj.n_lines = pathObj.last_row - pathObj.first_row
    pathObj.width = pathObj.last_col - pathObj.first_col

    run_list_slc = []
    run_list_geo = []

    for item in slc_list:
        run_list_slc.append((item, item.split('.full')[0]))

    for item in meta_data:
        run_list_geo.append((os.path.join(inps.geo_master, item + '.rdr.full'),
                                  os.path.join(inps.geo_master, item + '.rdr')))

    futures = []
    start_time = time.time()

    for item in run_list_slc:
        future = dask.delayed(cropSLC)(item)
        futures.append(future)

    for item in run_list_geo:
        future = dask.delayed(cropQualitymap)(item)
        futures.append(future)

    results = dask.compute(*futures)

    print('Done cropping images.')
