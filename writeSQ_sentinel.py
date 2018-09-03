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


#######################################
def comp_matr(x, y):
    a1 = np.size(x, axis=0)
    a2 = np.size(x, axis=1)
    out = np.empty((a1, a2), dtype=complex)
    for a in range(a1):
        for b in range(a2):
            out[a, b] = cmath.rect(x[a, b], y[a, b])
    return out


########################################
def main(argv):
    try:
        templateFileString = argv[1]
        slcname = argv[2]
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
        print ("  Usage: writeSQ_sentinel.py templatefile slcfile")
        print (" ")
        print ("  Example: ")
        print ("       writeSQ_sentinel.py $TE/NMerapiSenAT127VV.template 20180402/20180402.slc.full")
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
    sqdir = projdir + '/SqueeSAR'
    patchlistfile = sqdir + "/run_PSQ_sentinel"
    slclist = os.listdir(slavedir)

    wra = int(templateContents['squeesar.wsizerange'])
    waz = int(templateContents['squeesar.wsizeazimuth'])

    plist = []

    f = open(patchlistfile, "r")
    while 1:
        line = f.readline()
        if not line: break
        plist.append(line.split(" ")[-2].split('\t')[1])
    f.close()

    pr0 = np.load(sqdir + '/rowpatch.npy')
    pc0 = np.load(sqdir + '/colpatch.npy')

    pr = np.load(sqdir + '/rowpatch.npy')
    pr[1, 0, 0] = pr[1, 0, 0] - waz + 1
    pr[0, 0, 1::] = pr[0, 0, 1::] + waz + 1
    pr[1, 0, 1::] = pr[1, 0, 1::] - waz + 1
    pr[1, 0, -1] = pr[1, 0, -1] + waz - 1

    pc = np.load(sqdir + '/colpatch.npy')
    pc[1, 0, 0] = pc[1, 0, 0] - wra + 1
    pc[0, 0, 1::] = pc[0, 0, 1::] + wra + 1
    pc[1, 0, 1::] = pc[1, 0, 1::] - wra + 1
    pc[1, 0, -1] = pc[1, 0, -1] + wra - 1

    frow = pr[0, 0, 0]
    lrow = pr[1, 0, -1]
    fcol = pc[0, 0, 0]
    lcol = pc[1, 0, -1]

    lin = lrow - frow
    sam = lcol - fcol
    print(lin, sam)
    g = slcname
    imind = [i for (i, val) in enumerate(slclist) if val == g.split('/')[0]]

    while g == slcname:

        RSLCamp = np.zeros([np.int(lin), np.int(sam)])
        RSLCphase = np.zeros([np.int(lin), np.int(sam)])

        for t in plist:

            r = int(t.split('PATCH')[-1].split('_')[0])
            c = int(t.split('PATCH')[-1].split('_')[1])
            r1 = pr[0, 0, r]
            r2 = pr[1, 0, r]
            c1 = pc[0, 0, c]
            c2 = pc[1, 0, c]
            amp = np.load(sqdir + '/' + t + '/Amplitude_ref.npy')
            ph = np.load(sqdir + '/' + t + '/Phase_ref.npy')

            fr = (r1 - pr0[0, 0, r])
            lr = np.size(amp, axis=1) - (pr0[1, 0, r] - r2)
            fc = (c1 - pc0[0, 0, c])
            lc = np.size(amp, axis=2) - (pc0[1, 0, c] - c2)

            RSLCamp[r1:r2 + 1, c1:c2 + 1] = amp[imind, fr:lr + 1, fc:lc + 1]  # ampw.reshape(s1,s2)
            RSLCphase[r1:r2 + 1, c1:c2 + 1] = ph[imind, fr:lr + 1, fc:lc + 1]  # phw.reshape(s1,s2)

        data = comp_matr(RSLCamp, RSLCphase)
        slcf = slavedir + '/' + slcname

        with open(slcf + '.xml', 'r') as fid:
            vrtf = fid.readlines()

        nLines = lin
        width = sam

        outMap = np.memmap(slcf, dtype=np.complex64, mode='r+', shape=(nLines, width))
        outMap[:, :] = data

        oimg = isceobj.createSlcImage()
        oimg.setAccessMode('write')
        oimg.setFilename(slcf)
        oimg.setWidth(width)
        oimg.setLength(nLines)
        oimg.renderVRT()
        oimg.renderHdr()

        del outMap

        with open(slcf + '.xml', 'w') as fid:
            for line in vrtf:
                fid.write(line)
        fid.close()

        g = 0


if __name__ == '__main__':
    main(sys.argv[:])




