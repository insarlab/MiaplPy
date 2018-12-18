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
sys.path.insert(0, os.getenv('SENTINEL_STACK'))
from mergeBursts import multilook



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


def cmdLineParse(iargs=None):
    parser = createParser()
    return parser.parse_args(args=iargs)



def multiply(master_name, slave_name, out_name, flatten=False):

    print('multiply')
    mas_img = isceobj.createSlcImage()
    mas_img.load( master_name + '.xml')

    width = mas_img.getWidth()
    length = mas_img.getLength()

    ds = gdal.Open(master_name + '.vrt', gdal.GA_ReadOnly)
    master = ds.GetRasterBand(1).ReadAsArray()
    del ds
    ds = gdal.Open(slave_name + '.vrt', gdal.GA_ReadOnly)
    slave = ds.GetRasterBand(1).ReadAsArray()
    del ds
    print('read') 


    rng12 = np.zeros((length,width))

    cJ = np.complex64(-1j)
    fact = 4 * np.pi * 2.329562114715323 / 0.05546576

    ifg = np.memmap(out_name, dtype=np.complex64, mode='w+', shape=(length,width))
    for kk in range(0,length):
        ifg[kk,0:width+1] = master[kk,0:width+1] * np.conj(slave[kk,0:width+1])
        if flatten:
            phs = np.exp(cJ*fact*rng12[kk,0:width + 1])
            ifg[kk,0:width + 1] *= phs

    del master, slave, ifg

    obj_int = isceobj.createIntImage()
    obj_int.setFilename(out_name)
    obj_int.setWidth(width)
    obj_int.setLength(length)
    obj_int.setAccessMode('READ')
    obj_int.renderHdr()
    obj_int.renderVRT()
    
    return None


def main(iargs=None):
    '''Create interferograms.
    '''

    inps = cmdLineParse(iargs)
    
    ifg_dir = inps.interferogram
            
    if not os.path.exists(ifg_dir):
           os.makedirs(ifg_dir)

    ####Load relevant products
    sm = inps.master.split('/')[-1]
    master_name = os.path.join(inps.master, sm + '.slc.full')

    sl = inps.slave.split('/')[-1]
    slave_name = os.path.join(inps.slave, sl +'.slc.full')


    int_name = os.path.join(ifg_dir, 'fine.int')


    suffix = '.full'
    if (inps.numberRangeLooks == 1) and (inps.numberAzimuthLooks == 1):
        suffix = ''
    
    multiply(master_name, slave_name, int_name + suffix, flatten=inps.flatten)


    print(inps.multilook)
    inps.numberAzimuthLooks = 2
    inps.numberRangeLooks = 6
    if inps.multilook:
        multilook(int_name + suffix, int_name,
                  alks=inps.numberAzimuthLooks, rlks=inps.numberRangeLooks,
                  multilook_tool='isce', no_data=None)


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()


