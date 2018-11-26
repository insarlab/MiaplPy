#! /usr/bin/env python3
###############################################################################
# Project: Utilitiels for pysqsar
# Author: Sara Mirzaee
# Created: 10/2018
###############################################################################
import sys, os
import numpy as np
import cmath
from numpy import linalg as LA
from scipy.optimize import minimize, Bounds
import gdal
import isce
import isceobj
sys.path.insert(0, os.getenv('RSMAS_ISCE'))
from rsmas_logging import rsmas_logger, loglevel

logfile_name = os.getenv('OPERATIONS') + '/LOGS/squeesar.log'
logger_pysq = rsmas_logger(file_name=logfile_name)


def send_logger_squeesar():
    return logger_pysq

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


def corr2cov(corr_matrix = [],sigma = []):
    """ Converts correlation matrix to covariance matrix if std of variables are known. """
    
    D = np.diagflat(sigma)
    cov_matrix = D*corr_matrix*D
    
    return cov_matrix

###############################################################################
    
    
def cov2corr(cov_matrix):
    """ Converts covariance matrix to correlation/coherence matrix. """
    
    D = LA.pinv(np.diagflat(np.sqrt(np.diag(cov_matrix))))
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


def optphase(x0, inverse_gam):
    """ Returns the PTA maximum likelihood function value. """ 
    
    n = len(x0)
    x = np.ones([n+1,1])+0j
    x[1::,0] = np.exp(1j*x0[:])#.reshape(n,1)
    x = np.matrix(x)
    y = -np.matmul(x.getH(), inverse_gam)
    y = np.matmul(y, x)
    f = np.abs(np.log(y))
    
    return f

###############################################################################


def PTA_L_BFGS(xm): 
    """ Uses L-BFGS method to optimize PTA function and estimate phase values. """ 
    
    n = len(xm)
    x0 = np.zeros([n-1,1])
    x0[:,0] = np.real(xm[1::,0])
    coh = 1j*np.zeros([n,n])
    coh[:,:] = xm[:,1::]
    abs_coh = regularize_matrix(np.abs(coh))
    if np.size(abs_coh) == np.size(coh):
        inverse_gam = np.matrix(np.multiply(LA.pinv(abs_coh),coh))
        res = minimize(optphase, x0, args = inverse_gam, method='L-BFGS-B', 
                       bounds=Bounds(-100, 100, keep_feasible=False), 
                       tol=None, options={'gtol': 1e-6, 'disp': False})
        
        out = np.zeros([n,1])
        out[1::,0] = res.x
        out = np.unwrap(out,np.pi,axis=0)
        return out
    else:
        print('warning: coherence matrix not positive semidifinite, It is switched from PTA to EVD')
        return EVD_phase_estimation(coh)

###############################################################################


def EVD_phase_estimation(coh0):
    """ Estimates the phase values based on eigen value decomosition """
    
    w,v = LA.eigh(coh0)
    f = np.where(np.abs(w) == np.sort(np.abs(w))[len(coh0)-1])
    vec = v[:,f].reshape(len(w),1)
    x0 = np.angle(vec)
    x0 = x0 - x0[0,0]
    x0 = np.unwrap(x0,np.pi,axis=0)
    
    return x0

###############################################################################


def EMI_phase_estimation(coh0):
    """ Estimates the phase values based on EMI decomosition (Homa Ansari paper) """
    
    abscoh = regularize_matrix(np.abs(coh0))
    if np.size(abscoh) == np.size(coh0):
        M = np.multiply(LA.pinv(abscoh),coh0)
        w,v = LA.eigh(M)
        f = np.where(np.abs(w) == np.sort(np.abs(w))[0])
        #vec = LA.pinv(v[:,f].reshape(len(w),1)*np.sqrt(len(coh0)))
        vec = v[:,f].reshape(len(w),1)
        x0 = np.angle(vec).reshape(len(w),1)
        x0 = x0 - x0[0,0]
        x0 = np.unwrap(x0,np.pi,axis=0)
        return x0
    else:
        print('warning: coherence matric not positive semidifinite, It is switched from EMI to EVD')
        return EVD_phase_estimation(coh0)

###############################################################################


def CRLB_cov(gama, L):
    """ Estimates the Cramer Rao Lowe Bound based on coherence=gam and ensemble size = L """
    
    B_theta = np.zeros([len(gama),len(gama)-1])
    B_theta[1::,:] = np.identity(len(gama)-1)
    X = 2*L*(np.multiply(np.abs(gama),LA.pinv(np.abs(gama)))-np.identity(len(gama)))
    cov_out = LA.pinv(np.matmul(np.matmul(B_theta.T,(X+np.identity(len(X)))),B_theta))
    
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
    
    coh = np.multiply(cov2corr(np.abs(corr_mat)),np.angle(corr_mat))
    
    return coh
    
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
    
    out = np.sqrt(np.sum(x**2,axis=1)/(np.shape(x)[1]-1))
    
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



