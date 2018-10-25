#!/usr/bin/env python3

# Author: Piyush Agram
# Heresh Fattahi: Adopted for stack
# Modified for SqueeSAR by: Sara Mirzaee


import isce
import isceobj
import numpy as np
import argparse
import os
import gdal


def createParser():
    parser = argparse.ArgumentParser( description='Use polynomial offsets and create burst by burst interferograms')

    parser.add_argument('-m', '--master', dest='master', type=str, required=True,
            help='Directory with master acquisition')

    parser.add_argument('-s', '--slave', dest='slave', type=str, required=True,
            help='Directory with slave acquisition')

    parser.add_argument('-f', '--flatten', dest='flatten', action='store_true', default=False,
            help='Flatten the interferograms with offsets if needed')

    parser.add_argument('-i', '--interferogram', dest='interferogram', type=str, default='interferograms',
            help='Path for the interferogram')

    parser.add_argument('-p', '--interferogram_prefix', dest='intprefix', type=str, default='int',
            help='Prefix for the interferogram')

    parser.add_argument('-v', '--overlap', dest='overlap', action='store_true', default=False,
            help='Flatten the interferograms with offsets if needed')

    parser.add_argument('-l', '--multilook', action='store_true', dest='multilook', default=False,
                    help = 'Multilook the merged products. True or False')

    parser.add_argument('-A', '--azimuth_looks', type=str, dest='numberAzimuthLooks', default=3,
            help = 'azimuth looks')

    parser.add_argument('-R', '--range_looks', type=str, dest='numberRangeLooks', default=9,
            help = 'range looks')


    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)



def multiply(masname, slvname, outname, flatten=False):

    print('multiply')
    masImg = isceobj.createSlcImage()
    masImg.load( masname + '.xml')

    width = masImg.getWidth()
    length = masImg.getLength()

    ds = gdal.Open(masname + '.vrt', gdal.GA_ReadOnly)
    master = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    ds = gdal.Open(slvname + '.vrt', gdal.GA_ReadOnly)
    slave = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    print('read') 
    #master = np.memmap(masname, dtype=np.complex64, mode='r', shape=(length,width))
    #slave = np.memmap(slvname, dtype=np.complex64, mode='r', shape=(length, width))
    
    rng1 = np.zeros((length,width))
    rng2 = np.zeros((length,width))
    rng12 = rng2 - rng1

    cJ = np.complex64(-1j)
    fact = 4 * np.pi * 2.329562114715323 / 0.05546576
    #Zero out anytging outside the valid region:
    ifg = np.memmap(outname, dtype=np.complex64, mode='w+', shape=(length,width))
    for kk in range(0,length):
        ifg[kk,0:width+1] = master[kk,0:width+1] * np.conj(slave[kk,0:width+1])
        if flatten:
            phs = np.exp(cJ*fact*rng12[kk,0:width + 1])
            ifg[kk,0:width + 1] *= phs


    ####
    master=None
    slave=None
    ifg = None

    objInt = isceobj.createIntImage()
    objInt.setFilename(outname)
    objInt.setWidth(width)
    objInt.setLength(length)
    objInt.setAccessMode('READ')
    #objInt.createImage()
    #objInt.finalizeImage()
    objInt.renderHdr()
    objInt.renderVRT()
    return objInt

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


def main(iargs=None):
    '''Create overlap interferograms.
    '''
    inps=cmdLineParse(iargs)
    
    ifgdir = inps.interferogram
            
    if not os.path.exists(ifgdir):
           os.makedirs(ifgdir)

    ####Load relevant products
    sm = inps.master.split('/')[-1]
    mastername = os.path.join(inps.master, sm + '.slc.full')
    #mastername = os.path.join(inps.master, sm + '.slc')
    print(mastername)             
    sl = inps.slave.split('/')[-1]
    slavename = os.path.join(inps.slave, sl +'.slc.full')
    #slavename = os.path.join(inps.slave, sl + '.slc')
    print(slavename)
    intname = os.path.join(ifgdir, 'fine.int')
    
    #fact = 4 * np.pi * slave.rangePixelSize / slave.radarWavelength
                      
    #intimage = multiply(mastername, slavename, intname, fact, master, flatten=inps.flatten)

    suffix = '.full'
    if (inps.numberRangeLooks == 1) and (inps.numberAzimuthLooks == 1):
        suffix = ''
    
    intimage = multiply(mastername, slavename, intname + suffix, flatten=inps.flatten)


    print(inps.multilook)
    if inps.multilook:
        multilook(intname + suffix, intname, alks = inps.numberAzimuthLooks, rlks=inps.numberRangeLooks)


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()


