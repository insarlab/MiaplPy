#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################
import numpy as np
import os
import isce
import isceobj
import datetime
import logging
import argparse
from isceobj.Util.ImageUtil import ImageLib as IML
from isceobj.Util.decorators import use_api
import glob
import sys
import gdal
import subprocess
import cmath
import scipy.io

import logging

import pysar
from pysar.utils import utils
from pysar.utils import readfile
from pysar.utils import writefile

logger = logging.getLogger("process_sentinel")


######################3

def findrowcol(masterdir, lat1, lat2, lon1, lon2):
    objLat = isceobj.createIntImage()
    objLat.load(masterdir + '/lat.rdr.full.xml')
    objLon = isceobj.createIntImage()
    objLon.load(masterdir + '/lon.rdr.full.xml')
    width = objLon.getWidth()
    length = objLon.getLength()
    cw = int(width / 2)
    rl = int(length / 2)
    ds = gdal.Open(masterdir + '/lat.rdr.full.vrt', gdal.GA_ReadOnly)
    lat = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    ds = gdal.Open(masterdir + '/lon.rdr.full.vrt', gdal.GA_ReadOnly)
    lon = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    latc = lat[:, cw]
    lonr = lon[rl, :]
    latmin = latc - lat1
    latmax = latc - lat2
    lonmin = lonr - lon1
    lonmax = lonr - lon2
    frow = [index for index in range(len(latmin)) if np.abs(latmin[index]) == np.min(np.abs(latmin))]
    lrow = [index for index in range(len(latmax)) if np.abs(latmax[index]) == np.min(np.abs(latmax))]
    fcol = [index for index in range(len(lonmin)) if np.abs(lonmin[index]) == np.min(np.abs(lonmin))]
    lcol = [index for index in range(len(lonmax)) if np.abs(lonmax[index]) == np.min(np.abs(lonmax))]
    imcoords = [frow, lrow, fcol, lcol]
    return imcoords


def readim(slcname, frow, lrow, fcol, lcol):
    objSlc = isceobj.createSlcImage()
    objSlc.load(slcname + '.xml')
    ds = gdal.Open(slcname + '.vrt', gdal.GA_ReadOnly)
    Im = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    out = Im[frow:lrow, fcol:lcol]
    return out


def multilook(infile, outname=None, alks=5, rlks=15):
    '''
    Take looks.
    '''

    from mroipac.looks.Looks import Looks

    print('Multilooking {0} ...'.format(infile))

    inimg = isceobj.createImage()
    inimg.load(infile + '.xml')

    if outname is None:
        spl = os.path.splitext(inimg.filename)
        ext = '.{0}alks_{1}rlks'.format(alks, rlks)
        outname = spl[0] + ext + spl[1]

    lkObj = Looks()
    lkObj.setDownLooks(alks)
    lkObj.setAcrossLooks(rlks)
    lkObj.setInputImage(inimg)
    lkObj.setOutputFilename(outname)
    lkObj.looks()

    return outname


#######################
def main(argv):
    try:
        templateFileString = argv[1]
    except:
        print (
            "  ******************************************************************************************************")
        print (
            "  ******************************************************************************************************")
        print (
            "  ******************************************************************************************************")
        print (" ")
        print ("  Replace SLC images with squeezed values")
        print (" ")
        print ("  Usage: crop_sentinel.py templatefile")
        print (" ")
        print ("  Example: ")
        print ("       crop_sentinel.py $TE/NMerapiSenAT127VV.template")
        print (" ")
        print (
            "  ******************************************************************************************************")
        print (
            "  ******************************************************************************************************")
        print (
            "  ******************************************************************************************************")
        sys.exit(1)

    logger.info(os.path.basename(sys.argv[0]) + " " + sys.argv[1])
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

            multilook(filename, outname=filename.split('.full')[0], alks=azlks, rlks=rnlks)
        np.save(projdir + '/merged/cropped.npy', 'True')

if __name__ == '__main__':
    '''
    Crop SLCs.
    '''
    main(sys.argv[:])
