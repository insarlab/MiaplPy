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
from pysar.utils import readfile
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

    print(os.path.basename(sys.argv[0]) + " " + sys.argv[1])
    sys.exit(1)
    logger.log(loglevel.INFO, os.path.basename(sys.argv[0]) + " " + sys.argv[1])
    templateContents = readfile.read_template(templateFileString)

    projectName = os.path.basename(templateFileString).partition('.')[0]
    scratchDir = os.getenv('SCRATCHDIR')
    projdir = scratchDir + '/' + projectName
    slavedir = projdir + '/merged/SLC'
    gmasterdir = projdir + '/merged/geom_master'
    slclist = os.listdir(slavedir)
    

    lon1 = float(templateContents['lon1'])
    lon2 = float(templateContents['lon2'])
    lat1 = float(templateContents['lat1'])
    lat2 = float(templateContents['lat2'])

    azlks = templateContents['sentinelStack.azimuthLooks']
    rnlks = templateContents['sentinelStack.rangeLooks']
    
    if os.path.isfile(projdir + '/merged/cropped.npy'):
      print('Already cropped')
    else:
        croparea = np.array(findrowcol(gmasterdir, lat1, lat2, lon1, lon2))
        frow = np.int(croparea[0])
        lrow = np.int(croparea[1])
        fcol = np.int(croparea[2])
        lcol = np.int(croparea[3])

        nLines = lrow - frow
        width = lcol - fcol
        #slclist = []
    
        for t in slclist:
            filename = os.path.join(slavedir, t, t + '.slc.full')

            ds = gdal.Open(filename + '.vrt', gdal.GA_ReadOnly)
            Inpfile = ds.GetRasterBand(1).ReadAsArray()
            del ds

            outMap = np.memmap(filename, dtype=np.complex64, mode='r+', shape=(nLines, width))
            outMap[:, :] = Inpfile[frow:lrow, fcol:lcol]

            oimg = isceobj.createSlcImage()
            oimg.setAccessMode('write')
            oimg.setFilename(filename)
            oimg.setWidth(width)
            oimg.setLength(nLines)
            oimg.renderVRT()
            oimg.renderHdr()

            del outMap
            cmd = 'gdal_translate -of ENVI ' + filename + '.vrt ' + filename
            os.system(cmd)

            listgeo = ['hgt', 'lat', 'lon', 'los', 'shadowMask','incLocal']

        for t in listgeo:
            filename = os.path.join(gmasterdir, t + '.rdr.full')

            img = isceobj.createImage()
            img.load(filename + '.xml')
            bands = img.bands
            dtype = IML.NUMPY_type(img.dataType)
            scheme = img.scheme

            ds = gdal.Open(filename + '.vrt', gdal.GA_ReadOnly)
            Inpfile = ds.GetRasterBand(1).ReadAsArray()
            if bands == 2:
                Inpfile2 = ds.GetRasterBand(2).ReadAsArray()
            del ds, img

            outMap = IML.memmap(filename, mode='r+', nchannels=bands,
                            nxx=width, nyy=nLines, scheme=scheme, dataType=dtype)

            if bands == 2:
                outMap.bands[0][:, :] = Inpfile[frow:lrow, fcol:lcol]
                outMap.bands[1][:, :] = Inpfile2[frow:lrow, fcol:lcol]
            else:
                outMap.bands[0][:, :] = Inpfile[frow:lrow, fcol:lcol]

            IML.renderISCEXML(filename, bands,
                            nLines, width,
                            dtype, scheme)

            oimg = isceobj.createImage()
            oimg.load(filename + '.xml')
            oimg.imageType = dtype
            oimg.renderHdr()
            try:
                outMap.bands[0].base.base.flush()
            except:
                pass

            cmd = 'gdal_translate -of ENVI ' + filename + '.vrt ' + filename
            os.system(cmd)
            
            print(inps.multilook)
            if inps.multilook:
                multilook(inps.outfile+suffix, outname = inps.outfile,
                          alks = inps.numberAzimuthLooks, rlks=inps.numberRangeLooks,
                          multilook_tool=inps.multilookTool, no_data=inps.noData)
            else:
                print('Skipping multi-looking ....')

            multilook(filename, outname=filename.split('.full')[0], alks=azlks, rlks=rnlks)
        np.save(projdir + '/merged/cropped.npy', 'True')

if __name__ == '__main__':
    '''
    Crop SLCs.
    '''
    main(sys.argv[:])
