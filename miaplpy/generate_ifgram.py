#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import sys

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

blockPrint()
import datetime
import isceobj
import numpy as np
from miaplpy.objects.arg_parser import MiaplPyParser
import h5py
from math import sqrt, exp

enablePrint()


def main(iargs=None):
    """
        Overwrite filtered SLC images in Isce merged/SLC directory.
    """

    Parser = MiaplPyParser(iargs, script='generate_interferograms')
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

    print(inps.out_dir)
    os.makedirs(inps.out_dir, exist_ok=True)

    resampName = inps.out_dir + '/fine'
    resampInt = resampName + '.int'
    filtInt = os.path.dirname(resampInt) + '/filt_fine.int'
    cor_file = os.path.dirname(resampInt) + '/filt_fine.cor'

    if os.path.exists(cor_file + '.xml'):
        return

    length, width = run_interferogram(inps, resampName)

    filter_strength = inps.filter_strength
    runFilter(resampInt, filtInt, filter_strength)

    estCoherence(filtInt, cor_file)
    #run_interpolation(filtInt, inps.stack_file, length, width)

    return


def run_interferogram(inps, resampName):
    if inps.azlooks * inps.rglooks > 1:
        extention = '.ml.slc'
    else:
        extention = '.slc'

    with h5py.File(inps.stack_file, 'r') as ds:
        date_list = np.array([x.decode('UTF-8') for x in ds['date'][:]])
        ref_ind = np.where(date_list==inps.reference)[0]
        sec_ind = np.where(date_list==inps.secondary)[0]
        phase_series = ds['phase']
        amplitudes = ds['amplitude']

        length = phase_series.shape[1]
        width = phase_series.shape[2]

        resampInt = resampName + '.int'

        intImage = isceobj.createIntImage()
        intImage.setFilename(resampInt)
        intImage.setAccessMode('write')
        intImage.setWidth(width)
        intImage.setLength(length)
        intImage.createImage()

        out_ifg = intImage.asMemMap(resampInt)
        box_size = 3000
        num_row = int(np.ceil(length / box_size))
        num_col = int(np.ceil(width / box_size))
        for i in range(num_row):
            for k in range(num_col):
                row_1 = i * box_size
                row_2 = i * box_size + box_size
                col_1 = k * box_size
                col_2 = k * box_size + box_size
                if row_2 > length:
                    row_2 = length
                if col_2 > width:
                    col_2 = width

                ref_phase = phase_series[ref_ind, row_1:row_2, col_1:col_2].reshape(row_2 - row_1, col_2 - col_1)
                sec_phase = phase_series[sec_ind, row_1:row_2, col_1:col_2].reshape(row_2 - row_1, col_2 - col_1)
                ref_amplitude = amplitudes[ref_ind, row_1:row_2, col_1:col_2].reshape(row_2 - row_1, col_2 - col_1)
                sec_amplitude = amplitudes[sec_ind, row_1:row_2, col_1:col_2].reshape(row_2 - row_1, col_2 - col_1)

                ifg = (ref_amplitude * sec_amplitude) * np.exp(1j * (ref_phase - sec_phase))

                out_ifg[row_1:row_2, col_1:col_2, 0] = ifg[:, :]

        intImage.renderHdr()
        intImage.finalizeImage()

    return length, width



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
    objFilter.wireOutputPort(name='filtered interferogram', object=filtImage)
    objFilter.goldsteinWerner(alpha=filterStrength)

    intImage.finalizeImage()
    filtImage.finalizeImage()

def runFilterG(infile, outfile, filterStrength):

    # Initialize the flattened interferogram
    intImage = isceobj.createIntImage()
    intImage.load(infile + '.xml')
    intImage.setAccessMode('read')
    intImage.createImage()

    # Create the filtered interferogram
    filtImage = isceobj.createIntImage()
    filtImage.setFilename(outfile)
    filtImage.setWidth(intImage.getWidth())
    filtImage.setLength(intImage.getLength())
    filtImage.setAccessMode('write')
    filtImage.createImage()

    img = intImage.memMap(mode='r', band=0)
    original = np.fft.fft2(img[:, :])
    center = np.fft.fftshift(original)
    LowPassCenter = center * gaussianLP(100, img.shape)
    LowPass = np.fft.ifftshift(LowPassCenter)
    inverse_LowPass = np.fft.ifft2(LowPass)

    out_filtered = filtImage.asMemMap(outfile)
    out_filtered[:, :, 0] = inverse_LowPass[:, :]

    filtImage.renderHdr()

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

def run_interpolation(filtifg, tcoh_file, length, width):
    from scipy.spatial import Delaunay
    from scipy.interpolate import LinearNDInterpolator

    ifg = np.memmap(filtifg, dtype=np.complex64, mode='r+', shape=(length, width))
    with h5py.File(tcoh_file, 'r') as ds:
        tcoh_ds = ds['temporalCoherence']

        box_size = 3000
        num_row = int(np.ceil(length / box_size))
        num_col = int(np.ceil(width / box_size))
        for i in range(num_row):
            for k in range(num_col):
                row_1 = i * box_size
                row_2 = i * box_size + box_size
                col_1 = k * box_size
                col_2 = k * box_size + box_size
                if row_2 > length:
                    row_2 = length
                if col_2 > width:
                    col_2 = width

                ifg_sub = ifg[row_1:row_2, col_1:col_2]
                tcoh = tcoh_ds[0, row_1:row_2, col_1:col_2]
                mask = np.zeros((tcoh.shape[0], tcoh.shape[1]))
                mask[tcoh >= 0.5] = 1

                y, x = np.where(mask == 1)
                yi, xi = np.where(mask == 0)
                zifg = np.angle(ifg_sub[y, x])
                points = np.array([[r, c] for r, c in zip(y, x)])
                tri = Delaunay(points)
                func = LinearNDInterpolator(tri, zifg, fill_value=0)
                interp_points = np.array([[r, c] for r, c in zip(yi.flatten(), xi.flatten())])
                res = np.exp(1j * func(interp_points)) * (np.abs(ifg_sub[yi, xi]).flatten())
                ifg[y + row_1, x + col_1] = ifg_sub[y, x]
                ifg[yi + row_1, xi + col_1] = res

    del ifg
    return


def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

if __name__ == '__main__':
    main()


