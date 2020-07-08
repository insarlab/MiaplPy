#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import sys
import os
import isceobj
from minopy.objects.IsceImage import create_slc_image
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from components.stdproc.stdproc import crossmul
from isceobj.Image.IntImage import IntImage
import numpy as np
from minopy.objects.arg_parser import MinoPyParser


def main(iargs=None):
    """
        Overwrite filtered SLC images in Isce merged/SLC directory.
    """

    Parser = MinoPyParser(iargs, script='generate_interferograms')
    inps = Parser.parse()

    isce_stack_path = os.getenv('ISCE_STACK')
    system_path = os.getenv('PATH')

    if inps.prefix == 'tops':
        sys.path.append(os.path.join(os.getenv('ISCE_STACK'), 'topsStack'))
        '''
        if 'stripmapStack' in system_path:
            system_path_2 = system_path.replace('stripmapStack', 'topsStack')
            os.environ['PATH'] = system_path_2
        elif not 'topsStack' in system_path:
            os.environ['PATH'] = isce_stack_path + '/topsStack/:' + system_path
        '''
        resampName = run_interferogram_tops(inps)

    if inps.prefix == 'stripmap':
        sys.path.append(os.path.join(os.getenv('ISCE_STACK'), 'stripmapStack'))
        '''
        if 'topsStack' in system_path:
            system_path_2 = system_path.replace('topsStack', 'stripmapStack')
            os.environ['PATH'] = system_path_2
        elif not 'stripmapStack' in system_path:
            os.environ['PATH'] = isce_stack_path + '/stripmapStack/:' + system_path
        '''
        resampName = run_interferogram_stripmap(inps)

    ##
    from FilterAndCoherence import estCoherence, runFilter

    cor_file = resampName + '.cor'
    resampInt = resampName + '.int'
    estCoherence(resampInt, cor_file)

    filtInt = os.path.dirname(resampInt) + '/filt_' + os.path.basename(resampInt)
    filter_strength = 0.1
    runFilter(resampInt, filtInt, filter_strength)

    return #reset_path(system_path)

def run_interferogram_tops(inps):

    imageSlc = isceobj.createImage()
    imageSlc.load(inps.slc_file + '.xml')

    width = int(imageSlc.getWidth())
    lines = int(imageSlc.getLength())
    bands = int(imageSlc.bands)

    resampName = inps.ifg_dir + '/fine'
    resampInt = resampName + '.int'

    rslc = np.memmap(inps.slc_file, dtype=np.complex64, mode='r', shape=(bands, lines, width))

    ifg = np.memmap(resampInt, dtype=np.complex64, mode='w+', shape=(lines, width))

    for kk in range(0, lines):
        ifg[kk, :] = (rslc[inps.band_master, kk, :] * np.conj(rslc[inps.band_slave, kk, :])).reshape(1, -1)

    obj_int = IntImage()
    obj_int.setFilename(resampInt)
    obj_int.setWidth(width)
    obj_int.setLength(lines)
    obj_int.setAccessMode('READ')
    obj_int.renderHdr()
    obj_int.renderVRT()

    return resampName

def run_interferogram_stripmap(inps):

    imageSlc = isceobj.createImage()
    imageSlc.load(inps.slc_file + '.xml')

    objSlc1 = create_slc_image()
    IU.copyAttributes(imageSlc, objSlc1)
    objSlc1.set_bands = inps.band_master
    objSlc1.setAccessMode('read')
    objSlc1.createImage()

    objSlc2 = create_slc_image()
    IU.copyAttributes(imageSlc, objSlc2)
    objSlc2.set_band = inps.band_slave
    objSlc2.setAccessMode('read')
    objSlc2.createImage()

    width = int(imageSlc.getWidth())
    lines = int(imageSlc.getLength())

    resampName = inps.ifg_dir + '/fine'
    resampAmp = resampName + '.amp'
    resampInt = resampName + '.int'

    objInt = isceobj.createIntImage()
    objInt.setFilename(resampInt)
    objInt.setWidth(width)
    imageInt = isceobj.createIntImage()
    IU.copyAttributes(objInt, imageInt)
    objInt.setAccessMode('write')
    objInt.createImage()

    objAmp = isceobj.createAmpImage()
    objAmp.setFilename(resampAmp)
    objAmp.setWidth(width)
    imageAmp = isceobj.createAmpImage()
    IU.copyAttributes(objAmp, imageAmp)
    objAmp.setAccessMode('write')
    objAmp.createImage()

    objCrossmul = crossmul.createcrossmul()
    objCrossmul.width = width
    objCrossmul.length = lines
    objCrossmul.LooksDown = 1      # azLooks
    objCrossmul.LooksAcross = 1    # rgLooks

    objCrossmul.crossmul(objSlc1, objSlc2, objInt, objAmp)

    for obj in [objInt, objAmp, objSlc1, objSlc2]:
        obj.finalizeImage()

    return resampName

#def reset_path(system_path):
#    os.environ['PATH'] = system_path
#    return


if __name__ == '__main__':
    """
        Overwrite filtered SLC images.
    """

    main()


