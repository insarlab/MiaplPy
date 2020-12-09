#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import os
import isceobj
from isceobj.Image.IntImage import IntImage
import numpy as np
from minopy.objects.arg_parser import MinoPyParser
import gdal


def main(iargs=None):
    """
        Overwrite filtered SLC images in Isce merged/SLC directory.
    """

    Parser = MinoPyParser(iargs, script='generate_interferograms')
    inps = Parser.parse()
    resampName = run_interferogram(inps)

    resampInt = resampName + '.int'

    filtInt = os.path.dirname(resampInt) + '/filt_fine.int'
    filter_strength = inps.filter_strength
    runFilter(resampInt, filtInt, filter_strength)

    cor_file = os.path.dirname(resampInt) + '/filt_fine.cor'
    estCoherence(filtInt, cor_file)

    return


def run_interferogram(inps):
    if inps.azlooks * inps.rglooks > 1:
        extention = '.ml.slc'
    else:
        extention = '.slc'

    inps.reference = os.path.join(inps.reference, os.path.basename(inps.reference) + extention)
    inps.secondary = os.path.join(inps.secondary, os.path.basename(inps.secondary) + extention)

    ds = gdal.Open(inps.reference + '.vrt', gdal.GA_ReadOnly)
    width = ds.RasterXSize
    length = ds.RasterYSize

    reference = np.memmap(inps.reference, dtype=np.complex64, mode='r', shape=(length, width))
    secondary = np.memmap(inps.secondary, dtype=np.complex64, mode='r', shape=(length, width))

    resampName = inps.out_dir + '/fine'
    resampInt = resampName + '.int'

    ifg = np.memmap(resampInt, dtype=np.complex64, mode='w+', shape=(length, width))

    for kk in range(length):
        ifg[kk, :] = reference[kk, :] * np.conj(secondary[kk, :])

    obj_int = IntImage()
    obj_int.setFilename(resampInt)
    obj_int.setWidth(width)
    obj_int.setLength(length)
    obj_int.setAccessMode('READ')
    obj_int.renderHdr()
    obj_int.renderVRT()

    return resampName


def runFilter(infile, outfile, filterStrength):
    from mroipac.filter.Filter import Filter

    # Initialize the flattened interferogram
    topoflatIntFilename = infile
    intImage = isceobj.createIntImage()
    intImage.load( infile + '.xml')
    intImage.setAccessMode('read')
    intImage.createImage()

    # Create the filtered interferogram
    filtImage = isceobj.createIntImage()
    filtImage.setFilename(outfile)
    filtImage.setWidth(intImage.getWidth())
    filtImage.setAccessMode('write')
    filtImage.createImage()

    objFilter = Filter()
    objFilter.wireInputPort(name='interferogram',object=intImage)
    objFilter.wireOutputPort(name='filtered interferogram',object=filtImage)
    objFilter.goldsteinWerner(alpha=filterStrength)

    intImage.finalizeImage()
    filtImage.finalizeImage()


def estCoherence(outfile, corfile):
    from mroipac.icu.Icu import Icu

    # Create phase sigma correlation file here
    filtImage = isceobj.createIntImage()
    filtImage.load(outfile + '.xml')
    filtImage.setAccessMode('read')
    filtImage.createImage()

    phsigImage = isceobj.createImage()
    phsigImage.dataType = 'FLOAT'
    phsigImage.bands = 1
    phsigImage.setWidth(filtImage.getWidth())
    phsigImage.setFilename(corfile)
    phsigImage.setAccessMode('write')
    phsigImage.createImage()

    icuObj = Icu(name='sentinel_filter_icu')
    icuObj.configure()
    icuObj.unwrappingFlag = False
    icuObj.useAmplitudeFlag = False
    # icuObj.correlationType = 'NOSLOPE'

    icuObj.icu(intImage=filtImage, phsigImage=phsigImage)
    phsigImage.renderHdr()

    filtImage.finalizeImage()
    phsigImage.finalizeImage()


if __name__ == '__main__':
    main()


