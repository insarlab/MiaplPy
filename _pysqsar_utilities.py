#! /usr/bin/env python3
###############################################################################
#
# Project: Utilitiels for pysqsar
# Author: Sara Mirzaee
# Created: 10/2018
#
###############################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy import linalg as LA
from scipy.optimize import minimize
from scipy.stats import anderson_ksamp
from skimage.measure import label
import time
from dask import compute, delayed
import dask.multiprocessing

def readim(slcname):
    ds = gdal.Open(slcname + '.vrt', gdal.GA_ReadOnly)
    Im = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    return Im


class Node():
    #cols = 3, rows = 4, nodes = [[Node(i,j) for j in range(cols)] for i in range(rows)]
      def __init__(self, i,j):
        self.name = "%s_%s" % (str(i),str(j))
        self.refp = [i,j]
        return None
      def __repr__(self):
        return self.name  

def corr2cov(A = [],sigma = []):
    D = np.diag(sigma)
    cov = D*A*D
    return cov
    
def cov2corr(x):                  
    D = np.diag(1 / np.sqrt(np.diag(x)))
    y = np.matmul(D, x)
    corr = np.matmul(y, np.transpose(D))
    return corr

def is_semi_pos_def_chol(x):
    try:
        np.linalg.cholesky(x)
        return True
    except np.linalg.linalg.LinAlgError:
        return False
    

def L2norm_rowwis(M):
    s = np.shape(M)
    return (LA.norm(M, axis=1)).reshape(s[0],1)


def regularize_matrix(M):
    sh = np.shape(M)
    N = np.zeros([sh[0], sh[1]])
    N[:,:] = M[:,:]
    en = 1e-6
    t = 0
    while t<500:
        if is_semi_pos_def_chol(N):
            return N 
        else:
            N[:,:] = N[:,:] + en*np.identity(len(N))
            en = 2*en
            t = t+1
    return 0

def gam_pta_f(g1, g2):
    n1 = g1.shape[0]
    g11 = g1.reshape(n1 * n1, 1)
    g22 = g2.reshape(n1 * n1, 1)
    [r, c] = np.where(g11 != 0)
    gam = np.real(np.dot(np.exp(1j * g11[r, 0]), np.exp(-1j * g22[r, 0]))) * 2 / (n1 ** 2 - n1)
    return gam

def optphase(x0, igam_c):
    n = len(x0)
    x = np.ones([n+1,1])+0j
    x[1::,0] = np.exp(1j*x0[:])#.reshape(n,1)
    x = np.matrix(x)
    y = np.matmul(-x.getH(), igam_c)
    f = np.abs(np.log(np.matmul(y, x)))
    return f

def PTA_L_BFGS(xm): 
    n = len(xm)
    x0 = np.zeros([n-1,1])
    x0[:,0] = np.real(xm[1::,0])
    coh = 1j*np.zeros([n,n])
    coh[:,:] = xm[:,1::]
    abscoh = regularize_matrix(np.abs(coh))
    if np.size(abscoh) == np.size(coh):
        igam_c = np.matrix(np.multiply(LA.pinv(abscoh),coh))
        res = minimize(optphase, x0, args = igam_c, method='L-BFGS-B', tol=None, options={'gtol': 1e-6, 'disp': True})
        out = np.zeros([n,1])
        out[1::,0] = -res.x
        return out
    else:
        print('warning: coherence matrix not positive semidifinite, It is switched from PTA to EVD')
        return EVD_phase_estimation(coh)


def EVD_phase_estimation(coh0):
    w,v = LA.eig(coh0)
    f = np.where(w == np.sort(w)[len(coh0)-1])
    x0 = np.reshape(-np.angle(v[:,f]),[len(coh0),1])
    return x0


def EMI_phase_estimation(coh0):
        abscoh = regularize_matrix(np.abs(coh0))
        if np.size(abscoh) == np.size(coh0):
            M = np.multiply(LA.pinv(abscoh),coh0)
            w,v = LA.eig(M)
            f = np.where(w == np.sort(w)[0])
            x0 = np.reshape((v[:,f]),[len(v),1])
            return -np.angle(x0)
        else:
            print('warning: coherence matric not positive semidifinite, It is switched from EMI to EVD')
            return EVD_phase_estimation(coh0)
            

def CRLB_cov(gama, L):
    Btheta = np.zeros([len(gama),len(gama)-1])
    Btheta[1::,:] = np.identity(len(gama)-1)
    X = 2*L*(np.multiply(np.abs(gama),LA.pinv(np.abs(gama)))-np.identity(len(gama)))
    cov_out = LA.pinv(np.matmul(np.matmul(Btheta.T,(X+np.identity(len(X)))),Btheta))
    return cov_out

    
def daysmat(n_img,tmp_bl):
    ddays = np.matrix(np.exp(1j*np.arange(n_img)*tmp_bl/10000))
    days_mat = np.round((np.angle(np.matmul(ddays.getH(),ddays)))*10000)
    return days_mat

def simulate_phase(n_img=100,tmp_bl=6,deformation_rate=1,lamda=56):
    days_mat = daysmat(n_img,tmp_bl)
    Ip = days_mat*4*np.pi*deformation_rate/(lamda*365)
    np.fill_diagonal(Ip, 0)
    return Ip, days_mat

def simulate_corr(Ip, days_mat,gam0=0.6,gamf=0.2,decorr_days=50):
    corr_mat = np.multiply((gam0-gamf)*np.exp(-np.abs(days_mat/decorr_days))+gamf,np.exp(1j*Ip))
    return corr_mat
    
def est_corr(CCGsam):
    #corr_mat = np.matmul(np.conj(CCGsam),CCGsam.T)\
    #/np.sqrt(np.matmul(L2norm_rowwis(CCGsam)**2,(L2norm_rowwis(CCGsam)**2).T))
    CCGS = np.matrix(CCGsam)
    corr_mat = np.matmul(CCGS,CCGS.getH())/CCGS.shape[1]
    f = np.angle(corr_mat)
    corr_mat = np.multiply(np.abs(corr_mat),np.exp(-1j*f))
    return corr_mat
    

def custom_cmap(vmin=0,vmax=1):
    from spectrumRGB import rgb
    ww=np.arange(380.,781.)
    rgb=rgb()
    import matplotlib as mpl
    cmap = mpl.colors.ListedColormap(rgb)
    norm = mpl.colors.Normalize(vmin, vmax)
    return cmap, norm
   

def EST_rms(x):
    z = np.matrix(np.reshape(x,[len(x),1]))
    out = (np.matmul(z,z.getH()))/len(x)
    return out
          
    
def trwin(x):
    n1 = np.size(x, axis=0)
    n2 = np.size(x, axis=1)
    n3 = np.size(x, axis=2)
    x1 = np.zeros((n1, n3, n2))
    for t in range(n1):
        x1[t, :, :] = x[t, :, :].transpose()
    return x1

def shpobj(df):
    n = df.shape
    for i in range(n[0]):
        for j in range(n[1]):
            df.at[i,j] = Node(i,j)
    return df

def win_loc(mydf, wra=21, waz=15, nimage=54, lin=330, sam=342):
    r0 = mydf.refp[0]
    c0 = mydf.refp[1]
    r = np.ogrid[r0 - ((waz - 1) / 2):r0 + ((waz - 1) / 2) + 1]
    refr = np.array([(waz - 1) / 2]) 
    r = r[r >= 0]
    r = r[r < lin]
    refr = refr - (waz - len(r))
    c = np.ogrid[c0 - ((wra - 1) / 2):c0 + ((wra - 1) / 2) + 1]
    refc = np.array([(wra - 1) / 2])
    c = c[c >= 0]
    c = c[c < sam]
    refc = refc - (wra - len(c))
    mydf.refp0 = [refr,refc]
    mydf.rows = r
    mydf.cols = c
    return mydf
    
    
def shp_loc(df, pixelsdict=dict):
    amp = pixelsdict['amp']
    nimage = amp.shape[0]
    s = df.shape
    for q in range(s[0]):
        mydf = df[q]
        print('mydf:',mydf)
        r = mydf.rows.astype(int)
        c = mydf.cols.astype(int)
        x, y = np.meshgrid(r, c, sparse=True)
        refr = mydf.refp[0] 
        refc = mydf.refp[1] 
        refr0 = mydf.refp0[0] 
        refc0 = mydf.refp0[1] 
        win = amp[:, x, y]
        win = trwin(win)
        testvec = win.reshape(nimage, len(r) * len(c))
        ksres = np.ones(len(r) * len(c))
        S1 = amp[:, refr, refc]
        for m in range(len(testvec[0])):
            S2 = testvec[:, m]
            try:
                test = anderson_ksamp([S1, S2])
                if test.significance_level > 0.05:
                    ksres[m] = 1
                else:
                    ksres[m] = 0
            except:
                ksres[m] = 0
        ks_res = ksres.reshape(len(r), len(c))
        ks_label = label(ks_res, background=False, connectivity=2)
        reflabel = ks_label[refr0.astype(int), refc0.astype(int)]
        rr, cc = np.where(ks_label == reflabel)
        rr = rr + r[0]
        cc = cc + c[0]
        mydf.rows = rr
        mydf.cols = cc
        if len(rr)>20:
            mydf.scatterer = 'DS'
        else:
            mydf.scatterer = 'PS'
    return df  


def patch_slice(lin,sam,waz,wra):
    pr1 = np.ogrid[0:lin-50:300]
    pr2 = pr1+300
    pr2[-1] = lin
    pr1[1::] = pr1[1::] - 2*waz

    pc1 = np.ogrid[0:sam-50:300]
    pc2 = pc1+300
    pc2[-1] = sam
    pc1[1::] = pc1[1::] - 2*wra
    pr = [[pr1], [pr2]]
    pc = [[pc1], [pc2]]
    patchlist = []
    for n1 in range(len(pr1)):
        lin1 = pr2[n1] - pr1[n1]
        for n2 in range(len(pc1)):
            sam1 = pc2[n2] - pc1[n2]
            patchlist.append(str(n1) + '_' + str(n2))
    return pr,pc,patchlist

                    
def comp_matr(x, y):
    a1 = np.size(x, axis=0)
    a2 = np.size(x, axis=1)
    out = np.empty((a1, a2), dtype=complex)
    for a in range(a1):
        for b in range(a2):
            out[a, b] = cmath.rect(x[a, b], y[a, b])
    return out


def phase_link(df, pixelsdict=dict):          
    nimage = pixelsdict['amp'].shape[0]
    s = df.shape
    for q in range(s[0]):
        mydf = df[q]
        if mydf.scatterer == 'DS':
            rr = mydf.rows.astype(int)
            cc = mydf.cols.astype(int)
            refr = mydf.refp[0].astype(int)
            refc = mydf.refp[1].astype(int)
            dp = np.matrix(1.0 * np.arange(nimage * len(rr)).reshape((nimage, len(rr))))
            dp = np.exp(1j * dp)
            dpamp = pixelsdict['amp'][:, rr, cc]
            dpph = pixelsdict['ph'][:, rr, cc]
            dp = np.matrix(comp_matr(dpamp, dpph)) 
            cov_m = np.matmul(dp, dp.getH()) / (len(rr))
            phi = np.angle(cov_m)
            abs_cov = np.abs(cov_m)
            if is_semi_pos_def_chol(abs_cov):
                coh = corrcov(abs_cov)
                gam_c = np.multiply(coh, np.exp(1j * phi))
                try:
                    ph0 = EMI_phase_estimation(gam_c)
                    ph0 = ph0 - ph0[0]
                    xm = np.zeros([len(ph0),len(ph0)+1])+0j
                    xm[:,0:1] = np.reshape(ph0,[len(ph0),1])
                    xm[:,1::] = cov_m[:,:]
                    res_PTA = psq.PTA_L_BFGS(xm)
                    ph_PTA = np.reshape(res_PTA,[len(res_PTA),1])
                    xn = np.matrix(ph_PTA.reshape(nimage, 1))
                except:
                    xn = np.matrix(pixelsdict['ph'][:, refr, refc].reshape(nimage, 1))
                    xn = xn - xn[0,0]
            ampn = np.sqrt(np.abs(np.diag(cov_m))) 
            g1 = np.triu(phi)
            g2 = np.matmul(np.exp(1j * xn), (np.exp(1j * xn)).getH())
            g2 = np.triu(np.angle(g2), 1)
            gam_pta = gam_pta_f(g1, g2)
            if gam_pta > 0.5 and gam_pta <= 1:
                pixelsdict['amp_ref'][:, refr:refr + 1, refc:refc + 1] = np.array(ampn).reshape(nimage, 1, 1)
                pixelsdict['ph_ref'][:, refr:refr + 1, refc:refc + 1] = np.array(xn).reshape(nimage, 1, 1)
    
    tt, vv = np.where(pixelsdict['amp_ref'][0, :, :] == 0)
    tt = tt.astype(int)
    vv = vv.astype(int)
    pixelsdict['amp_ref'][:, tt, vv] = pixelsdict['amp'][:, tt, vv]
    pixelsdict['ph_ref'][:, tt, vv] = pixelsdict['ph'][:, tt, vv]
                    
    np.save(pixelsdict['work_dir'] + '/Amplitude_ref.npy', pixelsdict['amp_ref'])
    np.save(pixelsdict['work_dir'] + '/Phase_ref.npy', pixelsdict['ph_ref'])
    return df   
       
        
                    
                    
                    
                    
                    
                    
                    
                    
                    
