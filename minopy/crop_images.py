#!/usr/bin/env python3
############################################################
# Program is part of MiNoPy                                #
# Author:  Sara Mirzaee                                    #
############################################################
import logging
import warnings

warnings.filterwarnings("ignore")

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import os, sys

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

blockPrint()
import datetime
from isceobj.Util.ImageUtil import ImageLib as IML
from osgeo import gdal
import numpy as np
from mintpy import subset
from minopy.objects.utils import read_attribute, coord_rev
from minopy.objects.arg_parser import MinoPyParser
enablePrint()

#################################################################
datasetName2templateKey = {'slc': 'MINOPY.load.slcFile',
                           'height': 'MINOPY.load.demFile',
                           'latitude': 'MINOPY.load.lookupYFile',
                           'longitude': 'MINOPY.load.lookupXFile',
                           'incidenceAngle': 'MINOPY.load.incAngleFile',
                           'shadowMask': 'MINOPY.load.shadowMaskFile',
                           'waterMask': 'MINOPY.load.waterMaskFile',
                           }


#################################################################

def main(iargs=None):
    Parser = MinoPyParser(iargs, script='crop_images')
    inps = Parser.parse()

    dateStr = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d:%H%M%S')

    if not iargs is None:
        msg = os.path.basename(__file__) + ' ' + ' '.join(iargs[:])
        string = dateStr + " * " + msg
        print(string)
    else:
        msg = os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1::])
        string = dateStr + " * " + msg
        print(string)

    inps.slc_dir = os.path.abspath(inps.slc_dir)
    inps.geometry_dir = os.path.abspath(inps.geometry_dir)

    geometry_files = ['lat', 'lon', 'los', 'hgt', 'shadowMask']
    slc_paths = os.listdir(inps.slc_dir)

    pix_box, geo_box = subset.read_subset_template2box(inps.template_file[0])
    print(pix_box, geo_box)

    geo_file = os.path.join(inps.geometry_dir, 'hgt.rdr.full.xml')
    atr = read_attribute(geo_file.split('.xml')[0], metafile_ext='.xml')
    print(atr)

    # geo_box --> pix_box
    lookupFile = [os.path.join(inps.geometry_dir, 'lat.rdr.full'),
                  os.path.join(inps.geometry_dir, 'lon.rdr.full')]
    coord = coord_rev(atr, lookup_file=lookupFile)
    if geo_box is not None:
        pix_box = coord.bbox_geo2radar(geo_box)
        pix_box = coord.check_box_within_data_coverage(pix_box)
        print('input bounding box of interest in lalo: {}'.format(geo_box))
    print('box to read for datasets in y/x: {}'.format(pix_box))

    if not pix_box and not geo_box:
        print('No subset is given')
        return

    col1 = int(pix_box[0])
    col2 = int(pix_box[2])
    row1 = int(pix_box[1])
    row2 = int(pix_box[3])

    length = int(row2 - row1)
    width = int(col2 - col1)

    os.makedirs(inps.out_dir, exist_ok=True)
    os.makedirs(inps.out_dir + '/SLC', exist_ok=True)
    os.makedirs(inps.out_dir + '/geom_reference', exist_ok=True)

    for slc in slc_paths:
        os.makedirs(inps.out_dir + '/SLC/' + slc, exist_ok=True)
        slc_file = os.path.join(inps.slc_dir, slc, slc + '.slc.full')
        crop_slc_file = os.path.join(inps.out_dir + '/SLC', slc, slc + '.slc.full')
        if not os.path.exists(slc_file + '.xml'):
            print('{} does not exist'.format(slc_file))
        dsSlc = gdal.Open(slc_file + '.vrt', gdal.GA_ReadOnly)
        full_slc = dsSlc.GetRasterBand(1).ReadAsArray(col1, row1, width, length)
        crop_slc = np.memmap(crop_slc_file, dtype='complex64', mode='w+', shape=(length, width))
        crop_slc[:, :] = full_slc[:, :]
        IML.renderISCEXML(crop_slc_file, bands=1, nyy=length, nxx=width, datatype='complex64', scheme='BSQ')

    for geom in geometry_files:
        geo_file = os.path.join(inps.geometry_dir, geom + '.rdr.full')
        crop_geo_file = os.path.join(inps.out_dir + '/geom_reference', geom + '.rdr.full')
        if not os.path.exists(geo_file + '.xml'):
            print('{} does not exist'.format(geo_file))
        dsGeo = gdal.Open(geo_file + '.vrt', gdal.GA_ReadOnly)
        full_geom = dsGeo.GetRasterBand(1).ReadAsArray(col1, row1, width, length)
        dtype = 'float32'
        scheme = 'BIL'
        if geom == 'los':
            scheme = 'BSQ'
        if geom == 'shadowMask':
            dtype = 'byte'
        if dsGeo.RasterCount == 2:
            full_geom2 = dsGeo.GetRasterBand(2).ReadAsArray(col1, row1, width, length)
            crop_geom = np.memmap(crop_geo_file, dtype=dtype, mode='w+', shape=(2, length, width))
            crop_geom[0, :, :] = full_geom[:, :]
            crop_geom[1, :, :] = full_geom2[:, :]
            IML.renderISCEXML(crop_geo_file, bands=2, nyy=length, nxx=width, datatype=dtype,
                              scheme=scheme)
        else:
            crop_geom = np.memmap(crop_geo_file, dtype=dtype, mode='w+', shape=(length, width))
            crop_geom[:, :] = full_geom[:, :]
            IML.renderISCEXML(crop_geo_file, bands=1, nyy=length, nxx=width, datatype=dtype,
                              scheme=scheme)

    return

#################################################################
if __name__ == '__main__':
    """
    loading a stack of InSAR pairs to and HDF5 file
    """
    main()

