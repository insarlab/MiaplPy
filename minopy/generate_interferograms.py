#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import os
import isce
import isceobj
from isceobj.Image.IntImage import IntImage
import numpy as np
from minopy.objects.arg_parser import MinoPyParser
from osgeo import gdal
import h5py


def main(iargs=None):
    """
        Overwrite filtered SLC images in Isce merged/SLC directory.
    """

    Parser = MinoPyParser(iargs, script='generate_interferograms')
    inps = Parser.parse()

    os.makedirs(inps.out_dir, exist_ok=True)

    resampName = inps.out_dir + '/fine'
    resampInt = resampName + '.int'
    filtInt = os.path.dirname(resampInt) + '/filt_fine.int'
    cor_file = os.path.dirname(resampInt) + '/filt_fine.cor'

    #if os.path.exists(cor_file + '.xml'):
    #    return

    run_interferogram(inps, resampName)

    filter_strength = inps.filter_strength
    runFilter(resampInt, filtInt, filter_strength)

    estCoherence(filtInt, cor_file)

    return


def run_interferogram(inps, resampName):
    if inps.azlooks * inps.rglooks > 1:
        extention = '.ml.slc'
    else:
        extention = '.slc'

    with h5py.File(inps.stack_file, 'r') as ds:
        date_list = np.array([x.decode('UTF-8') for x in ds['dates'][:]])
        ref_ind = np.where(date_list==inps.reference)[0]
        sec_ind = np.where(date_list==inps.secondary)[0]

        slcs =  ds['slc']
        length = slcs.shape[1]
        width = slcs.shape[2]

        resampInt = resampName + '.int'

        #ifg_mem = np.zeros([length, width], dtype=np.complex64)
 
        ref_image = slcs[ref_ind, :, :].reshape(length, width)
        sec_image = np.conj(slcs[sec_ind, :, :]).reshape(length, width)

        ifg = np.memmap(resampInt, dtype=np.complex64, mode='write', shape=(length, width))
        ifg[:, :] = ref_image * sec_image

        #for kk in range(length):
        #    ifg[kk, :] = (slcs[ref_ind, kk, :] * np.conj(slcs[sec_ind, kk, :])).reshape(1, -1)


        #ifg[:, :] = ifg_mem[:, :]

        obj_int = IntImage()
        obj_int.setFilename(resampInt)
        obj_int.setWidth(width)
        obj_int.setLength(length)
        obj_int.setAccessMode('READ')
        obj_int.renderHdr()
        obj_int.renderVRT()


        intImage = isceobj.createIntImage()
        intImage.load(resampInt + '.xml')
        intImage.setAccessMode('read')
        intImage.createImage()
        intImage.finalizeImage()

    return resampName

def runFilter(infile, outfile, filterStrength):
    from mroipac.filter.Filter import Filter

    # Initialize the flattened interferogram
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


