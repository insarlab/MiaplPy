#! /usr/bin/env python3
###############################################################################
#
# Project: Utilitiels for pysqsar
# Author: Sara Mirzaee
# Created: 10/2018
#
###############################################################################

import numpy as np
import sys, os
import cmath
from numpy import linalg as LA
from scipy.optimize import minimize
from scipy.stats import anderson_ksamp
from skimage.measure import label
import gdal
import isce
import isceobj
sys.path.insert(0, os.getenv('RSMAS_ISCE'))
from rsmas_logging import rsmas_logger, loglevel

logfile_name = os.getenv('OPERATIONS') + '/LOGS/squeesar.log'
logger = rsmas_logger(file_name=logfile_name)


def send_logger_squeesar():
    return logger

######################################################################################


def convert_geo2image_coord(geo_master_dir, lat_south, lat_north, lon_west, lon_east):
    """ Finds the corresponding line and sample based on geographical coordinates. """

    obj_lat = isceobj.createIntImage()
    obj_lat.load(geo_master_dir + '/lat.rdr.full.xml')
    obj_lon = isceobj.createIntImage()
    obj_lon.load(geo_master_dir + '/lon.rdr.full.xml')
    width = obj_lon.getWidth()
    length = obj_lon.getLength()
    center_sample = int(width / 2)
    center_line = int(length / 2)
    ds = gdal.Open(geo_master_dir + '/lat.rdr.full.vrt', gdal.GA_ReadOnly)
    lat = ds.GetRasterBand(1).ReadAsArray()
    del ds
    ds = gdal.Open(geo_master_dir + "/lon.rdr.full.vrt", gdal.GA_ReadOnly)
    lon = ds.GetRasterBand(1).ReadAsArray()
    del ds
    lat_center_sample = lat[:, center_sample]
    lon_center_line = lon[center_line, :]
    lat_min = lat_center_sample - lat_south
    lat_max = lat_center_sample - lat_north
    lon_min = lon_center_line - lon_west
    lon_max = lon_center_line - lon_east
    first_row = [index for index in range(len(lat_min)) if np.abs(lat_min[index]) == np.min(np.abs(lat_min))]
    last_row = [index for index in range(len(lat_max)) if np.abs(lat_max[index]) == np.min(np.abs(lat_max))]
    first_col = [index for index in range(len(lon_min)) if np.abs(lon_min[index]) == np.min(np.abs(lon_min))]
    last_col = [index for index in range(len(lon_max)) if np.abs(lon_max[index]) == np.min(np.abs(lon_max))]
    image_coord = [first_row, last_row, first_col, last_col]

    return image_coord

################################################################################


def read_slc_and_crop(slc_file, first_row, last_row, first_col, last_col):
    """ Read SLC file and return crop. """

    obj_slc = isceobj.createSlcImage()
    obj_slc.load(slc_file + '.xml')
    ds = gdal.Open(slc_file + '.vrt', gdal.GA_ReadOnly)
    slc_image = ds.GetRasterBand(1).ReadAsArray()
    del ds
    out = slc_image[first_row:last_row, first_col:last_col]
    return out

################################################################################


def read_image(image_file):
    """ Reads images from isce. """
    
    ds = gdal.Open(image_file + '.vrt', gdal.GA_ReadOnly)
    image = ds.GetRasterBand(1).ReadAsArray()
    del ds
    
    return image

###############################################################################


class Node():
    """ Creates an object for each pixel. """
    #cols = 3, rows = 4, nodes = [[Node(i,j) for j in range(cols)] for i in range(rows)]
    def __init__(self, i,j):
        self.name = "%s_%s" % (str(i),str(j))
        self.ref_pixel = [i,j]
        return None
    def __repr__(self):
        return self.name

###############################################################################
    
    
def corr2cov(corr_matrix = [],sigma = []):
    """ Converts correlation matrix to covariance matrix if std of variables are known. """
    
    D = np.diagflat(sigma)
    cov_matrix = D*corr_matrix*D
    
    return cov_matrix

###############################################################################
    
    
def cov2corr(cov_matrix):
    """ Converts covariance matrix to correlation/coherence matrix. """
    
    D = np.diagflat(1 / np.sqrt(np.diag(cov_matrix)))
    y = np.matmul(D, cov_matrix)
    corr_matrix = np.matmul(y, np.transpose(D))
    
    return corr_matrix

###############################################################################


def is_semi_pos_def_chol(x):
    """ Checks the positive semi definitness of a matrix. """
    
    try:
        np.linalg.cholesky(x)
        return True
    except np.linalg.linalg.LinAlgError:
        
        return False
    
###############################################################################    


def L2norm_rowwis(M):
    """ Returns L2 norm of a matrix rowwise. """
    
    s = np.shape(M)
    return (LA.norm(M, axis=1)).reshape(s[0],1)

###############################################################################


def regularize_matrix(M):
    """ Regularizes a matrix to make it positive semi definite. """ 
    
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

###############################################################################


def gam_pta_f(g1, g2):
    """ Returns squeesar PTA coherence between the initial and estimated phase vectors. """
    
    n1 = g1.shape[0]
    [r, c] = np.where(g1 != 0)
    g11 = g1[r,c].reshape(len(r))
    g22 = g2[r,c].reshape(len(r))
    gam = np.real(np.dot(np.exp(1j * g11), np.exp(-1j * g22))) * 2 / (n1 ** 2 - n1)
    
    return gam

###############################################################################


def optphase(x0, igam_c):
    """ Returns the PTA maximum likelihood function value. """ 
    
    n = len(x0)
    x = np.ones([n+1,1])+0j
    x[1::,0] = np.exp(1j*x0[:])#.reshape(n,1)
    x = np.matrix(x)
    y = np.matmul(-x.getH(), igam_c)
    f = np.abs(np.log(np.matmul(y, x)))
    
    return f

###############################################################################


def PTA_L_BFGS(xm): 
    """ Uses L-BFGS method to optimize PTA function and estimate phase values. """ 
    
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

###############################################################################


def EVD_phase_estimation(coh0):
    """ Estimates the phase values based on eigen value decomosition """
    
    w,v = LA.eig(coh0)
    f = np.where(w == np.sort(w)[len(coh0)-1])
    x0 = np.reshape(-np.angle(v[:,f]),[len(coh0),1])
    x0 = x0 - x0[0,0]
    
    return x0

###############################################################################


def EMI_phase_estimation(coh0):
    """ Estimates the phase values based on EMI decomosition (Homa Ansari paper) """
    
    abscoh = regularize_matrix(np.abs(coh0))
    if np.size(abscoh) == np.size(coh0):
        M = np.multiply(LA.pinv(abscoh),coh0)
        w,v = LA.eig(M)
        f = np.where(w == np.sort(w)[0])
        x0 = np.reshape((v[:,f]),[len(v),1])
        out = -np.angle(x0)
        out = out - out[0,0]
        return out
    else:
        print('warning: coherence matric not positive semidifinite, It is switched from EMI to EVD')
        return EVD_phase_estimation(coh0)

###############################################################################


def CRLB_cov(gama, L):
    """ Estimates the Cramer Rao Lowe Bound based on coherence=gam and ensemble size = L """
    
    Btheta = np.zeros([len(gama),len(gama)-1])
    Btheta[1::,:] = np.identity(len(gama)-1)
    X = 2*L*(np.multiply(np.abs(gama),LA.pinv(np.abs(gama)))-np.identity(len(gama)))
    cov_out = LA.pinv(np.matmul(np.matmul(Btheta.T,(X+np.identity(len(X)))),Btheta))
    
    return cov_out

###############################################################################


def daysmat(n_img,tmp_bl):
    """ Builds a temporal baseline matrix """
    
    ddays = np.matrix(np.exp(1j*np.arange(n_img)*tmp_bl/10000))
    days_mat = np.round((np.angle(np.matmul(ddays.getH(),ddays)))*10000)
    
    return days_mat

###############################################################################


def simulate_phase(n_img=100,tmp_bl=6,deformation_rate=1,lamda=56):
    """ Simulate Interferogram with constant velocity deformation rate """
    
    days_mat = daysmat(n_img,tmp_bl)
    Ip = days_mat*4*np.pi*deformation_rate/(lamda*365)
    np.fill_diagonal(Ip, 0)
    
    return Ip, days_mat

###############################################################################


def simulate_corr(Ip, days_mat,gam0=0.6,gamf=0.2,decorr_days=50):
    """ Simulate Correlation matrix."""
    
    corr_mat = np.multiply((gam0-gamf)*np.exp(-np.abs(days_mat/decorr_days))+gamf,np.exp(1j*Ip))
    return corr_mat

###############################################################################


def est_corr(CCGsam):
    """ Estimate Correlation matrix from an ensemble."""
        
    CCGS = np.matrix(CCGsam)
    corr_mat = np.matmul(CCGS,CCGS.getH())/CCGS.shape[1]
    f = np.angle(corr_mat)
    corr_mat = np.multiply(np.abs(corr_mat),np.exp(-1j*f))
    
    return corr_mat
    
###############################################################################


def custom_cmap(vmin=0,vmax=1):
    """ create a custom colormap based on visible portion of electromagnetive wave."""
    
    from spectrumRGB import rgb
    rgb=rgb()
    import matplotlib as mpl
    cmap = mpl.colors.ListedColormap(rgb)
    norm = mpl.colors.Normalize(vmin, vmax)
    
    return cmap, norm
   
###############################################################################


def EST_rms(x):
    """ Estimate Root mean square error."""
    
    z = np.matrix(np.reshape(x,[len(x),1]))
    out = (np.matmul(z,z.getH()))/len(x)
    
    return out
          
###############################################################################


def trwin(x):
    """ Transpose each layer of a 3 dimentional array."""
    
    n1 = np.size(x, axis=0)
    n2 = np.size(x, axis=1)
    n3 = np.size(x, axis=2)
    x1 = np.zeros((n1, n3, n2))
    for t in range(n1):
        x1[t, :, :] = x[t, :, :].transpose()
        
    return x1

###############################################################################


def shpobj(df):
    """ create an object for each pixel in a dataframe."""
    
    n = df.shape
    for i in range(n[0]):
        for j in range(n[1]):
            df.at[i,j] = Node(i,j)
            
    return df

###############################################################################


def win_loc(mydf, wra=21, waz=15, nimage=54, lin=330, sam=342):
    """ Extract the pixels in multilook window based on image dimensions and pixel location."""
    
    r0 = mydf.ref_pixel[0]
    c0 = mydf.ref_pixel[1]
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
    mydf.ref_pixel_in_window = [refr,refc]
    mydf.rows = r
    mydf.cols = c
    
    return mydf
    
###############################################################################


def shp_loc(df, pixels_dict=dict):
    """ Find statistical homogeneous pixels in a window based on Anderson Darling similarity test."""
    
    amp = pixels_dict['amp']
    nimage = amp.shape[0]
    s = df.shape
    for q in range(s[0]):
        mydf = df[q]
        r = mydf.rows.astype(int)
        c = mydf.cols.astype(int)
        x, y = np.meshgrid(r, c, sparse=True)
        refr = mydf.ref_pixel[0]
        refc = mydf.ref_pixel[1]
        refr0 = mydf.ref_pixel_in_window[0]
        refc0 = mydf.ref_pixel_in_window[1]
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

###############################################################################


def patch_slice(lin,sam,waz,wra):
    """ Devides an image into patches of size 300 by 300 by considering the overlay of the size of multilook window."""
    
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
        for n2 in range(len(pc1)):
            patchlist.append(str(n1) + '_' + str(n2))
            
    return pr,pc,patchlist

###############################################################################


def comp_matr(x, y):
    """ Returns a complex matrix given the amplitude and phase."""
    
    a1 = np.size(x, axis=0)
    a2 = np.size(x, axis=1)
    out = np.empty((a1, a2), dtype=complex)
    for a in range(a1):
        for b in range(a2):
            out[a, b] = cmath.rect(x[a, b], y[a, b])
            
    return out

###############################################################################


def phase_link(df, pixels_dict=dict):
    """ Runs the phase linking algorithm over each DS.""" 
    
    nimage = pixels_dict['amp'].shape[0]
    s = df.shape
    for q in range(s[0]):
        mydf = df[q]
        if mydf.scatterer == 'DS':
            rr = mydf.rows.astype(int)
            cc = mydf.cols.astype(int)
            refr = mydf.ref_pixel[0]
            refc = mydf.ref_pixel[1]
            dp = np.matrix(1.0 * np.arange(nimage * len(rr)).reshape((nimage, len(rr))))
            dp = np.exp(1j * dp)
            dpamp = pixels_dict['amp'][:, rr, cc]
            dpph = pixels_dict['ph'][:, rr, cc]
            dp = np.matrix(comp_matr(dpamp, dpph)) 
            cov_m = np.matmul(dp, dp.getH()) / (len(rr))
            phi = np.angle(cov_m)
            abs_cov = np.abs(cov_m)
            coh = cov2corr(abs_cov)
            gam_c = np.multiply(coh, np.exp(1j * phi))
            try:
                ph0 = EVD_phase_estimation(gam_c)
                xm = np.zeros([len(ph0),len(ph0)+1])+0j
                xm[:,0:1] = np.reshape(ph0,[len(ph0),1])
                xm[:,1::] = cov_m[:,:]
                res_PTA = psq.PTA_L_BFGS(xm)
                ph_PTA = np.reshape(res_PTA,[len(res_PTA),1])
                xn = np.matrix(ph_PTA.reshape(nimage, 1))
                xn = np.matrix(ph0.reshape(nimage, 1))
            except:
                xn = np.matrix(pixels_dict['ph'][:, refr, refc].reshape(nimage, 1))
                xn = xn - xn[0,0]
            ampn = np.sqrt(np.abs(np.diag(cov_m))) 
            g1 = np.triu(phi,1)
            g2 = np.matmul(np.exp(-1j * xn), (np.exp(-1j * xn)).getH())
            g2 = np.triu(np.angle(g2), 1)
            gam_pta = gam_pta_f(g1, g2)
            if gam_pta > 0.4 and gam_pta <= 1:
                mydf.ampref = np.array(ampn).reshape(nimage, 1, 1)
                mydf.phref = np.array(xn).reshape(nimage, 1, 1)
            else:
                mydf.ampref = pixels_dict['amp'][:, refr, refc].reshape(nimage, 1, 1)
                xn = np.matrix(pixels_dict['ph'][:, refr, refc].reshape(nimage, 1))
                xn = xn - xn[0,0]
                mydf.phref = np.array(xn).reshape(nimage, 1, 1)
    return df 

