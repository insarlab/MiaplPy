#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import sys
import glob
import time
import matplotlib.pyplot as plt
import logging

import pysar
from pysar.utils import utils
from pysar.utils import readfile
import subprocess
import cmath
from scipy.stats import ks_2samp, norm, ttest_ind
from skimage.measure import label
from scipy import linalg
import numpy as np
from numpy.linalg import pinv
import multiprocessing
from scipy.optimize import minimize

logger = logging.getLogger("process_sentinel")


###############


def comp_matr(x, y):
    a1 = np.size(x, axis=0)
    a2 = np.size(x, axis=1)
    out = np.empty((a1, a2), dtype=complex)
    for a in range(a1):
        for b in range(a2):
            out[a, b] = cmath.rect(x[a, b], y[a, b])
    return out


def is_semi_pos_def_chol(x):
    try:
        np.linalg.cholesky(x)
        return True
    except np.linalg.linalg.LinAlgError:
        return False


def corrcov(x):
    D = np.diag(1 / np.sqrt(np.diag(x)))
    y = np.matmul(D, x)
    corr = np.matmul(y, np.transpose(D))
    return corr


def gam_pta_f(g1, g2):
    n1 = g1.shape[0]
    [r, c] = np.where(g1 != 0)
    g11 = g1[r,c].reshape(len(r))
    g22 = g2[r,c].reshape(len(r))
    gam = np.real(np.dot(np.exp(1j * g11), np.exp(-1j * g22))) * 2 / (n1 ** 2 - n1)
    return gam


def trwin(x):
    n1 = np.size(x, axis=0)
    n2 = np.size(x, axis=1)
    n3 = np.size(x, axis=2)
    x1 = np.zeros((n1, n3, n2))
    for t in range(n1):
        x1[t, :, :] = x[t, :, :].transpose()
    return x1


def optphase(xm):
    global igam_c
    global fval
    n = max(np.shape(xm))  # max([np.size(xm,axis=0),np.size(xm,axis=1)])
    x = np.matrix(np.zeros((n + 1, 1)))
    x = np.matrix(np.exp(1j * x))
    x[1::, 0] = np.matrix(np.exp(1j * xm)).reshape(n, 1)
    x[0, 0] = fval
    y = np.matmul(x.getH(), np.matrix(igam_c))
    f = np.real(np.log(np.matmul(y, x)))
    return f


###################################


def main(argv):
    try:
        templateFileString = argv[1]
        patchDir = argv[2]
    except:
        print (
            "  ******************************************************************************************************")
        print (
            "  ******************************************************************************************************")
        print (
            "  ******************************************************************************************************")
        print (" ")
        print ("  Run SqueeSAR on coregistered SLCs in Process directory based on Gamma outputs")
        print (" ")
        print ("  Usage: PSQ_gamma.py ProjectName patchfolder")
        print (" ")
        print ("  Example: ")
        print ("       PSQ_gamma.py $TE/PichinchaSMT51TsxD PATCH0_0")
        print (" ")
        print (
            "  ******************************************************************************************************")
        print (
            "  ******************************************************************************************************")
        print (
            "  ******************************************************************************************************")
        sys.exit(1)

    global igam_c
    global fval
    global lin
    global sam
    global nimage
    global SN
    global SHP
    global w
    global t1, r, refr

    logger.info(os.path.basename(sys.argv[0]) + " " + sys.argv[1])
    templateContents = readfile.read_template(templateFileString)
    projectName = os.path.basename(templateFileString).partition('.')[0]
    scratchDir = os.getenv('SCRATCHDIR')

    projdir = scratchDir + '/' + projectName
    slavedir = projdir + '/merged/SLC'
    workDir = projdir + "/SqueeSAR/" + patchDir
    slclist = os.listdir(slavedir)


    nimage = len(slclist)

    RSLCamp = np.load(workDir + '/Amplitude.npy')
    RSLCphase = np.load(workDir + '/Phase.npy')

    lin = np.size(RSLCamp, axis=1)
    sam = np.size(RSLCamp, axis=2)

    wra = int(templateContents['squeesar.wsizerange'])
    waz = int(templateContents['squeesar.wsizeazimuth'])
    ####
    if os.path.isfile(workDir + '/SHP.npy'):
        SHP = np.load(workDir + '/SHP.npy')
        SHP = SHP.tolist()
        t1 = np.load(workDir + '/ind.npy')
    else:
        SHP = []
        t1 = 0
    while t1 < lin:  # rows
        r = np.ogrid[t1 - ((waz - 1) / 2):t1 + ((waz - 1) / 2) + 1]
        refr = np.array([(waz - 1) / 2])
        r = r[r >= 0]
        r = r[r < lin]
        refr = refr - (waz - len(r))
        t2i = 0
        while t2i < sam:
            time0 = time.time()
            c = np.ogrid[t2i - ((wra - 1) / 2):t2i + ((wra - 1) / 2) + 1]
            refc = np.array([(wra - 1) / 2])
            c = c[c >= 0]
            c = c[c < sam]
            refc = refc - (wra - len(c))
            x, y = np.meshgrid(r.astype(int), c.astype(int), sparse=True)
            win = RSLCamp[:, x, y]
            win = trwin(win)
            testvec = win.reshape(nimage, len(r) * len(c))
            ksres = np.ones(len(r) * len(c))
            S1 = RSLCamp[:, t1, t2i]
            S1 = S1.reshape(nimage, 1)
            for m in range(len(testvec[0])):
                S2 = testvec[:, m]
                S2 = S2.reshape(nimage, 1)
                test = ttest_ind(S1, S2, equal_var=False)
                if test[1] > 0.05:
                    ksres[m] = 1
                else:
                    ksres[m] = 0

            ks_res = ksres.reshape(len(r), len(c))
            ks_label = label(ks_res, background=False, connectivity=2)
            reflabel = ks_label[refr.astype(int), refc.astype(int)]

            rr, cc = np.where(ks_label == reflabel)

            rr = rr + r[0]
            cc = cc + c[0]
            if len(rr) > 20:
                shp = {'name': str(t1) + '_' + str(t2i), 'row': rr, 'col': cc}
                shp = np.array(shp)
                SHP.append(shp)
                timep = time.time() - time0
            t2i = t2i + 1
        print ('save row ' + str(t1) + ',  Elapsed time: ' + str(timep) + ' s')
        np.save(workDir + '/ind.npy', t1)
        np.save(workDir + '/SHP.npy', SHP)
        t1 = t1 + 1

    print('SHP created ...')

    ####
    #################################################
    if os.path.isfile(workDir + '/Phase_ref.npy'):
        RSLCamp_ref = np.load(workDir + '/Amplitude_ref.npy')
        RSLCphase_ref = np.load(workDir + '/Phase_ref.npy')
        t = np.load(workDir + '/indf.npy')
        shpn = len(SHP)
    else:
        RSLCamp_ref = np.zeros([nimage, lin, sam])
        RSLCphase_ref = np.zeros([nimage, lin, sam])
        shpn = len(SHP)
        t = 0
        np.save(workDir + '/Amplitude_ref.npy', RSLCamp_ref)
        np.save(workDir + '/Phase_ref.npy', RSLCphase_ref)
        np.save(workDir + '/indf.npy', t)
    while t < shpn:
        pix = SHP[t]
        pix = pix.tolist()
        if pix != None:
            rr = pix['row']
            t1 = int(pix['name'].split('_')[0])  # row
            t2 = int(pix['name'].split('_')[1])  # col
            cc = pix['col']
            dp = np.matrix(1.0 * np.arange(nimage * len(rr)).reshape((nimage, len(rr))))
            dp = np.exp(1j * dp)
            for q in range(len(rr)):
                Am = RSLCamp[:, rr[q].astype(int), cc[q].astype(int)]
                dpamp = np.matrix(1.0 * np.arange(nimage).reshape((nimage, 1)))
                dpamp[:, 0] = Am.reshape(nimage, 1)
                Ph = RSLCphase[:, rr[q].astype(int), cc[q].astype(int)]
                dpph = np.matrix(1.0 * np.arange(nimage).reshape((nimage, 1)))
                dpph[:, 0] = Ph.reshape(nimage, 1)
                dp[:, q] = comp_matr(dpamp, dpph)

            cov_m = np.matmul(dp, dp.getH()) / (len(rr))
            phi = np.angle(cov_m)
            abs_cov = np.abs(cov_m)
            if is_semi_pos_def_chol(abs_cov):
                print('yes it is semi pos def')
                coh = corrcov(abs_cov)
                gam_c = np.multiply(coh, np.exp(1j * phi))
                igam_c = np.multiply(pinv(np.abs(gam_c)), gam_c)
                x0 = RSLCphase[:, t1, t2].reshape(nimage, 1)
                fval = 0
                xm = x0[1::, 0]

                try:
                    res = minimize(optphase, xm, method='L-BFGS-B', tol=1e-5, options={'gtol': 1e-5, 'disp': False})
                    xn = np.zeros((nimage, 1))
                    xn[1::, 0] = res.x
                    xn[0, 0] = fval
                except:
                    xn = x0
                xn0 = np.exp(1j * xn)
                xn = np.angle(xn0)
                xn = np.matrix(xn.reshape(nimage, 1))
                # print('xn:',xn)
                ampn = np.sqrt(np.abs(np.diag(cov_m)))
                x0 = x0.reshape(nimage, 1)
                x0 = np.matrix(x0)
                g1 = np.triu(phi)
                g2 = np.matmul(np.exp(1j * xn), (np.exp(1j * xn)).getH())
                g2 = np.triu(np.angle(g2), 1)
                gam_pta = gam_pta_f(g1, g2)
                print ([t, gam_pta])
                if gam_pta > 0.5 and gam_pta <= 1:
                    RSLCamp_ref[:, t1:t1 + 1, t2:t2 + 1] = np.array(ampn).reshape(nimage, 1, 1)
                    RSLCphase_ref[:, t1:t1 + 1, t2:t2 + 1] = np.array(xn).reshape(nimage, 1, 1)

            else:
                print ('Warning: Coherence matrix is not semi positive definite')
                RSLCamp_ref[:, t1:t1 + 1, t2:t2 + 1] = RSLCamp[:, t1:t1 + 1, t2:t2 + 1]
                RSLCphase_ref[:, t1:t1 + 1, t2:t2 + 1] = RSLCphase[:, t1:t1 + 1, t2:t2 + 1]
            if t % 1000 == 0:
                np.save(workDir + '/Amplitude_ref.npy', RSLCamp_ref)
                np.save(workDir + '/Phase_ref.npy', RSLCphase_ref)
                np.save(workDir + '/indf.npy', t)
        else:

            print ('Warning: shp invalid')
        t += 1

    tt, vv = np.where(RSLCamp_ref[0, :, :] == 0)
    tt = tt.astype(int)
    vv = vv.astype(int)
    RSLCamp_ref[:, tt, vv] = RSLCamp[:, tt, vv]
    RSLCphase_ref[:, tt, vv] = RSLCphase[:, tt, vv]

    np.save(workDir + '/Amplitude_ref.npy', RSLCamp_ref)
    np.save(workDir + '/Phase_ref.npy', RSLCphase_ref)
    np.save(workDir + '/indf.npy', t)
    np.save(workDir + '/endflag.npy', 'True')


#################################################


if __name__ == '__main__':
    main(sys.argv[:])








