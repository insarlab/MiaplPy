#! /usr/bin/env python3
###############################################################################
# Project: Utilitiels for minopy
# Author: Sara Mirzaee
# Created: 10/2018
###############################################################################

import os
import numpy as np
import cmath
import datetime
from scipy import linalg as LA
from scipy.optimize import minimize
from scipy.stats import ks_2samp, anderson_ksamp, ttest_ind
import gdal
import isceobj
from mroipac.looks.Looks import Looks
import glob
import shutil
import warnings
import h5py
from mintpy.objects import timeseries, ifgramStack
from mintpy.ifgram_inversion import split2boxes, read_unwrap_phase, mask_unwrap_phase
################################################################################


def log_message(logdir, msg):
    f = open(os.path.join(logdir, 'log'), 'a+')
    dateStr = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d:%H%M%S')
    string = dateStr + " * " + msg
    print(string)
    f.write(string + "\n")
    f.close()
    return

################################################################################


def convert_geo2image_coord(geo_reference_dir, lat_south, lat_north, lon_west, lon_east):
    """ Finds the corresponding line and sample based on geographical coordinates. """

    ds = gdal.Open(geo_reference_dir + '/lat.rdr.full.vrt', gdal.GA_ReadOnly)
    lat_lut = ds.GetRasterBand(1).ReadAsArray()
    del ds

    ds = gdal.Open(geo_reference_dir + "/lon.rdr.full.vrt", gdal.GA_ReadOnly)
    lon_lut = ds.GetRasterBand(1).ReadAsArray()
    del ds

    mask_y = np.multiply(lat_lut >= lat_south, lat_lut <= lat_north)
    mask_x = np.multiply(lon_lut >= lon_west, lon_lut <= lon_east)
    mask_yx = np.multiply(mask_y, mask_x)

    rows, cols = np.where(mask_yx)

    first_row = np.rint(np.min(rows)).astype(int) - 10
    last_row = np.rint(np.max(rows)).astype(int) + 10
    first_col = np.rint(np.min(cols)).astype(int) - 10
    last_col = np.rint(np.max(cols)).astype(int) + 10

    image_coord = [first_row, last_row, first_col, last_col]

    return image_coord

##############################################################################


def patch_slice(lines, samples, azimuth_window, range_window, patch_size=200):
    """ Devides an image into patches of size 200 by 200 by considering the overlay of the size of multilook window."""

    patch_row_1 = np.ogrid[0:lines-50:patch_size]
    patch_row_2 = patch_row_1+patch_size
    patch_row_2[-1] = lines
    patch_row_1[1::] = patch_row_1[1::] - 2*azimuth_window

    patch_col_1 = np.ogrid[0:samples-50:patch_size]
    patch_col_2 = patch_col_1+patch_size
    patch_col_2[-1] = samples
    patch_col_1[1::] = patch_col_1[1::] - 2*range_window
    patch_rows = [[patch_row_1], [patch_row_2]]
    patch_cols = [[patch_col_1], [patch_col_2]]
    patchlist = []

    for row in range(len(patch_row_1)):
        for col in range(len(patch_col_1)):
            patchlist.append(str(row) + '_' + str(col))

    return patch_rows, patch_cols, patchlist

##############################################################################


def read_slc_and_crop(slc_file, first_row, last_row, first_col, last_col):
    """ Read SLC file and return crop. """
    import isceobj
    obj_slc = isceobj.createSlcImage()
    obj_slc.load(slc_file + '.xml')
    ds = gdal.Open(slc_file + '.vrt', gdal.GA_ReadOnly)
    slc_image = ds.GetRasterBand(1).ReadAsArray()
    del ds
    out = slc_image[first_row:last_row, first_col:last_col]
    return out

###############################################################################


def write_SLC(date_list, slc_dir, patch_dir, range_win, azimuth_win):
    import isceobj
    merge_dir = slc_dir.split('minopy')[0] + '/merged/SLC'
    if not os.path.exists(slc_dir):
        os.mkdir(slc_dir)
    patch_list = glob.glob(patch_dir + '/patch*')
    patch_rows = np.load(patch_dir + '/rowpatch.npy')
    patch_cols = np.load(patch_dir + '/colpatch.npy')
    patch_rows_overlap = np.zeros(np.shape(patch_rows), dtype=int)
    patch_rows_overlap[:, :, :] = patch_rows[:, :, :]
    patch_rows_overlap[1, 0, 0] = patch_rows_overlap[1, 0, 0] - azimuth_win + 1
    patch_rows_overlap[0, 0, 1::] = patch_rows_overlap[0, 0, 1::] + azimuth_win + 1
    patch_rows_overlap[1, 0, 1::] = patch_rows_overlap[1, 0, 1::] - azimuth_win + 1
    patch_rows_overlap[1, 0, -1] = patch_rows_overlap[1, 0, -1] + azimuth_win - 1

    patch_cols_overlap = np.zeros(np.shape(patch_cols), dtype=int)
    patch_cols_overlap[:, :, :] = patch_cols[:, :, :]
    patch_cols_overlap[1, 0, 0] = patch_cols_overlap[1, 0, 0] - range_win + 1
    patch_cols_overlap[0, 0, 1::] = patch_cols_overlap[0, 0, 1::] + range_win + 1
    patch_cols_overlap[1, 0, 1::] = patch_cols_overlap[1, 0, 1::] - range_win + 1
    patch_cols_overlap[1, 0, -1] = patch_cols_overlap[1, 0, -1] + range_win - 1

    first_row = patch_rows_overlap[0, 0, 0]
    last_row = patch_rows_overlap[1, 0, -1]
    first_col = patch_cols_overlap[0, 0, 0]
    last_col = patch_cols_overlap[1, 0, -1]

    n_line = last_row - first_row
    width = last_col - first_col
    n_image = len(date_list)

    for date_ind, date in enumerate(date_list):
        print(date)
        date_dir = os.path.join(slc_dir, date)
        if not os.path.exists(date_dir):
            os.mkdir(date_dir)
        out_slc = os.path.join(date_dir, date + '.slc')
        if os.path.exists(out_slc + '.xml'):
            continue
        slc = np.memmap(out_slc, dtype=np.complex64, mode='w+', shape=(n_line, width))
        for patch in patch_list:
            row = int(patch.split('patch')[-1].split('_')[0])
            col = int(patch.split('patch')[-1].split('_')[1])
            row1 = patch_rows_overlap[0, 0, row]
            row2 = patch_rows_overlap[1, 0, row]
            col1 = patch_cols_overlap[0, 0, col]
            col2 = patch_cols_overlap[1, 0, col]

            patch_lines = patch_rows[1, 0, row] - patch_rows[0, 0, row]
            patch_samples = patch_cols[1, 0, col] - patch_cols[0, 0, col]

            f_row = row1 - patch_rows[0, 0, row]
            l_row = row2 - patch_rows[0, 0, row]
            f_col = col1 - patch_cols[0, 0, col]
            l_col = col2 - patch_cols[0, 0, col]

            rslc_patch = np.memmap(patch + '/rslc_ref',
                                   dtype=np.complex64, mode='r', shape=(np.int(n_image), patch_lines, patch_samples))
            slc[row1:row2 + 1, col1:col2 + 1] = rslc_patch[date_ind, f_row:l_row + 1, f_col:l_col + 1]

        obj_slc = isceobj.createSlcImage()
        obj_slc.setFilename(out_slc)
        obj_slc.setWidth(width)
        obj_slc.setLength(n_line)
        obj_slc.bands = 1
        obj_slc.scheme = 'BIL'
        obj_slc.dataType = 'CFLOAT'
        obj_slc.setAccessMode('read')
        obj_slc.renderHdr()
        shutil.copytree(os.path.join(merge_dir, date, 'referenceShelve'), os.path.join(date_dir, 'referenceShelve'))
        shutil.copytree(os.path.join(merge_dir, date, 'secondaryShelve'), os.path.join(date_dir, 'secondaryShelve'))

    return

################################################################################


def read_image(image_file, box=None, band=1):
    """ Reads images from isce. """

    ds = gdal.Open(image_file + '.vrt', gdal.GA_ReadOnly)
    if not box is None:
        imds = ds.GetRasterBand(band)
        image = imds.ReadAsArray()[box[1]:box[3], box[0]:box[2]]
    else:
        image = ds.GetRasterBand(band).ReadAsArray()

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
    N[:, :] = M[:, :]
    en = 1e-6
    t = 0
    while t < 500:
        if is_semi_pos_def_chol(N):
            return N
        else:
            N[:, :] = N[:, :] + en*np.identity(len(N))
            en = 2*en
            t = t+1

    return 0

###############################################################################


def gam_pta(ph_filt, vec_refined):
    """ Returns squeesar PTA coherence between the initial and estimated phase vectors.
    :param ph_filt: np.angle(coh) before inversion
    :param vec_refined: refined complex vector after inversion
    """
    nm = np.shape(ph_filt)[0]
    diagones = np.diag(np.diag(np.ones([nm, nm])+1j))
    phi_mat = np.exp(1j * ph_filt)
    ph_refined = np.angle(vec_refined)
    g1 = np.exp(-1j * ph_refined).reshape(nm, 1)
    g2 = np.exp(1j * ph_refined).reshape(1, nm)
    theta_mat = np.matmul(g1, g2)
    ifgram_diff = np.multiply(phi_mat, theta_mat)
    ifgram_diff = ifgram_diff - np.multiply(ifgram_diff, diagones)
    temp_coh = np.abs(np.sum(ifgram_diff) / (nm ** 2 - nm))

    #print(temp_coh)

    return temp_coh

###############################################################################


def optphase(x0, inverse_gam):
    """ Returns the PTA maximum likelihood function value. """

    n = len(x0)
    x = np.exp(1j * x0).reshape(n, 1)
    x = np.matrix(x)
    y = np.matmul(x.getH(), inverse_gam)
    y = np.matmul(y, x)
    f = float(np.abs(np.log(y)))
    return f

###############################################################################


def PTA_L_BFGS(coh0):
    """ Uses L-BFGS method to optimize PTA function and estimate phase values. """
    n_image = coh0.shape[0]
    x0 = np.angle(EMI_phase_estimation(coh0))
    x0 = x0 - x0[0]
    abs_coh = regularize_matrix(np.abs(coh0))
    if np.size(abs_coh) == np.size(coh0):
        inverse_gam = np.matrix(np.multiply(LA.pinv(abs_coh), coh0))
        res = minimize(optphase, x0, args=inverse_gam, method='L-BFGS-B',
                       tol=None, options={'gtol': 1e-6, 'disp': False})
        out = res.x.reshape(n_image, 1)
        vec = np.multiply(np.abs(x0), np.exp(1j * out)).reshape(n_image, 1)

        x0 = np.exp(1j * np.angle(vec[0]))
        vec = np.multiply(vec, np.conj(x0))

        return vec

    else:

        print('warning: coherence matrix not positive semidifinite, It is switched from PTA to EVD')
        return EVD_phase_estimation(coh0)

###############################################################################


def EVD_phase_estimation(coh0):
    """ Estimates the phase values based on eigen value decomosition """
    eigen_value, eigen_vector = LA.eigh(coh0)
    vec = eigen_vector[:, -1].reshape(len(eigen_value), 1)
    x0 = np.exp(1j * np.angle(vec[0]))
    vec = np.multiply(vec, np.conj(x0))
    return vec

###############################################################################


def EMI_phase_estimation(coh0):
    """ Estimates the phase values based on EMI decomosition (Homa Ansari, 2018 paper) """
    abscoh = regularize_matrix(np.abs(coh0))
    if np.size(abscoh) == np.size(coh0):
        M = np.multiply(LA.pinv(abscoh), coh0)
        eigen_value, eigen_vector = LA.eigh(M)
        vec = eigen_vector[:, 0].reshape(len(eigen_value), 1)
        x0 = np.exp(1j * np.angle(vec[0]))
        vec = np.multiply(vec, np.conj(x0))
        return vec
    else:
        print('warning: coherence matrix not positive semidifinite, It is switched from EMI to EVD')
        return EVD_phase_estimation(coh0)

###############################################################################


def test_PS(coh_mat):
    """ checks if the pixel is PS """

    Eigen_value, Eigen_vector = LA.eigh(coh_mat)
    norm_eigenvalues = Eigen_value*100/np.sum(Eigen_value)
    indx = np.where(norm_eigenvalues > 25)[0]

    if len(indx) >= 1:
        msk = (norm_eigenvalues <= 25)
        Eigen_value[msk] = 0.
        CM = np.dot(Eigen_vector, np.dot(np.diag(np.sqrt(Eigen_value)), np.matrix.getH(Eigen_vector)))
        vec = EMI_phase_estimation(CM)
    else:
        vec = EMI_phase_estimation(coh_mat)

    x0 = np.exp(1j * np.angle(vec[0]))
    vec = np.multiply(vec, np.conj(x0))

    return vec

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
# Simulation:


def simulate_volcano_def_phase(n_img=100, tmp_bl=6):
    """ Simulate Interferogram with complex deformation signal """
    t = np.ogrid[0:(tmp_bl * n_img):tmp_bl]
    nl = int(len(t) / 4)
    x = np.zeros(len(t))
    x[0:nl] = -2 * t[0:nl] / 365
    x[nl:2 * nl] = 2 * (np.log((t[nl:2 * nl] - t[nl - 1]) / 365)) - 3 * (np.log((t[nl] - t[nl - 1]) / 365))
    x[2 * nl:3 * nl] = 10 * t[2 * nl:3 * nl] / 365 - x[2 * nl - 1] / 2
    x[3 * nl::] = -2 * t[3 * nl::] / 365

    return t, x


def simulate_constant_vel_phase(n_img=100, tmp_bl=6):
    """ Simulate Interferogram with constant velocity deformation rate """
    t = np.ogrid[0:(tmp_bl * n_img):tmp_bl]
    x = t / 365

    return t, x

###############################################################################


def simulate_coherence_matrix_exponential(t, gamma0, gammaf, Tau0, ph, seasonal=False):
    """Simulate a Coherence matrix based on de-correlation rate, phase and dates"""
    # t: a vector of acquistion times
    # ph: a vector of simulated phase time-series for one pixel
    # returns the complex covariance matrix
    # corr_mat = (gamma0-gammaf)*np.exp(-np.abs(days_mat/decorr_days))+gammaf
    length = t.shape[0]
    C = np.ones((length, length), dtype=np.complex64)
    factor = gamma0 - gammaf
    if seasonal:
        f1 = lambda x, y: (x - y) ** 2 - gammaf
        f2 = lambda x, y: (x + y) ** 2 - gamma0
        res = double_solve(f1, f2, 0.5, 0.5)
        A = res[0]
        B = res[1]

    for ii in range(length):
        for jj in range(ii + 1, length):
            if seasonal:
                factor = (A + B * np.cos(2 * np.pi * t[ii] / 180)) * (A + B * np.cos(2 * np.pi * t[jj] / 180))
            #gamma = factor*((gamma0-gammaf)*np.exp(-np.abs((t[ii] - t[jj])/Tau0))+gammaf)
            gamma = factor * (np.exp((t[ii] - t[jj]) / Tau0)) + gammaf
            C[ii, jj] = gamma * np.exp(1j * (ph[ii] - ph[jj]))
            C[jj, ii] = np.conj(C[ii, jj])

    return C

################################################################################


def simulate_noise(corr_matrix):
    nsar = corr_matrix.shape[0]
    eigen_value, eigen_vector = np.linalg.eigh(corr_matrix)
    msk = (eigen_value < 1e-3)
    eigen_value[msk] = 0.
    # corr_matrix =  np.dot(eigen_vector, np.dot(np.diag(eigen_value), np.matrix.getH(eigen_vector)))

    # C = np.linalg.cholesky(corr_matrix)
    CM = np.dot(eigen_vector, np.dot(np.diag(np.sqrt(eigen_value)), np.matrix.getH(eigen_vector)))
    Zr = (np.random.randn(nsar) + 1j*np.random.randn(nsar)) / np.sqrt(2)
    noise = np.dot(CM, Zr)

    return noise


def simulate_neighborhood_stack(corr_matrix, neighborSamples=300):
    """Simulating the neighbouring pixels (SHPs) based on a given coherence matrix"""

    numberOfSlc = corr_matrix.shape[0]
    # A 2D matrix for a neighborhood over time. Each column is the neighborhood complex data for each acquisition date

    neighbor_stack = np.zeros((numberOfSlc, neighborSamples), dtype=np.complex64)
    for ii in range(neighborSamples):
        cpxSLC = simulate_noise(corr_matrix)
        neighbor_stack[:, ii] = cpxSLC
    return neighbor_stack

##############################################################################


def double_solve(f1,f2,x0,y0):
    """Solve for two equation with two unknowns using iterations"""

    from scipy.optimize import fsolve
    func = lambda x: [f1(x[0], x[1]), f2(x[0], x[1])]
    return fsolve(func, [x0, y0])

###############################################################################


def est_corr(CCGsam):
    """ Estimate Correlation matrix from an ensemble."""

    CCGS = np.matrix(CCGsam)

    cov_mat = np.matmul(CCGS, CCGS.getH()) / CCGS.shape[1]

    corr_matrix = cov2corr(cov_mat)

    #corr_matrix = np.multiply(cov2corr(np.abs(cov_mat)), np.exp(1j * np.angle(cov_mat)))

    return corr_matrix

###############################################################################


def custom_cmap(vmin=0, vmax=1):
    """ create a custom colormap based on visible portion of electromagnetive wave."""

    from spectrumRGB import rgb
    rgb = rgb()
    import matplotlib as mpl
    cmap = mpl.colors.ListedColormap(rgb)
    norm = mpl.colors.Normalize(vmin, vmax)

    return cmap, norm

###############################################################################


def EST_rms(x):
    """ Estimate Root mean square error."""

    out = np.sqrt(np.sum(x ** 2, axis=1) / (np.shape(x)[1] - 1))

    return out

###############################################################################


def phase_linking_process(ccg_sample, stepp, method, squeez=True):
    """Inversion of phase based on a selected method among PTA, EVD and EMI """

    coh_mat = est_corr(ccg_sample)

    if 'PTA' in method:
        res = PTA_L_BFGS(coh_mat)
    elif 'EMI' in method:
        res = EMI_phase_estimation(coh_mat)
    else:
        res = EVD_phase_estimation(coh_mat)

    res = res.reshape(len(res), 1)

    if squeez:

        vm = np.exp(1j * np.angle(np.matrix(res[stepp::, :])))
        vm = np.matrix(vm / LA.norm(vm))
        squeezed = np.matmul(vm.getH(), ccg_sample[stepp::, :])

        return res, squeezed
    else:
        return res


###############################################################################

def CRLB_cov(gama, L):
    """ Estimates the Cramer Rao Lowe Bound based on coherence=gam and ensemble size = L """

    B_theta = np.zeros([len(gama), len(gama) - 1])
    B_theta[1::, :] = np.identity(len(gama) - 1)
    X = 2 * L * (np.multiply(np.abs(gama), LA.pinv(np.abs(gama))) - np.identity(len(gama)))
    cov_out = LA.pinv(np.matmul(np.matmul(B_theta.T, (X + np.identity(len(X)))), B_theta))

    return cov_out


###############################################################################


def sequential_phase_linking(full_stack_complex_samples, method, num_stack=10):
    """ phase linking of each pixel sequentially and applying a datum shift at the end """

    n_image = full_stack_complex_samples.shape[0]
    mini_stack_size = 10
    num_mini_stacks = np.int(np.floor(n_image / mini_stack_size))
    vec_refined = np.zeros([np.shape(full_stack_complex_samples)[0], 1]) + 0j

    for sstep in range(0, num_mini_stacks):

        first_line = sstep * mini_stack_size
        if sstep == num_mini_stacks - 1:
            last_line = n_image
        else:
            last_line = first_line + mini_stack_size
        num_lines = last_line - first_line

        if sstep == 0:

            mini_stack_complex_samples = full_stack_complex_samples[first_line:last_line, :]
            res, squeezed_images = phase_linking_process(mini_stack_complex_samples, sstep, method)

            vec_refined[first_line:last_line, 0:1] = res[sstep::, 0:1]
        else:

            if num_stack == 1:
                mini_stack_complex_samples = np.zeros([1 + num_lines, full_stack_complex_samples.shape[1]]) + 0j
                mini_stack_complex_samples[0, :] = np.complex64(squeezed_images[-1, :])
                mini_stack_complex_samples[1::, :] = full_stack_complex_samples[first_line:last_line, :]
                res, new_squeezed_image = phase_linking_process(mini_stack_complex_samples, 1, method)
                vec_refined[first_line:last_line, :] = res[1::, :]
                squeezed_images = np.vstack([squeezed_images, new_squeezed_image])
            else:
                mini_stack_complex_samples = np.zeros([sstep + num_lines, full_stack_complex_samples.shape[1]]) + 0j
                mini_stack_complex_samples[0:sstep, :] = squeezed_images
                mini_stack_complex_samples[sstep::, :] = full_stack_complex_samples[first_line:last_line, :]
                res, new_squeezed_image = phase_linking_process(mini_stack_complex_samples, sstep, method)
                vec_refined[first_line:last_line, :] = res[sstep::, :]
                squeezed_images = np.vstack([squeezed_images, new_squeezed_image])
            ###

    datum_connection_samples = squeezed_images
    datum_shift = np.angle(phase_linking_process(datum_connection_samples, 0, 'PTA', squeez=False))

    for sstep in range(len(datum_shift)):
        first_line = sstep * mini_stack_size
        if sstep == num_mini_stacks - 1:
            last_line = n_image
        else:
            last_line = first_line + mini_stack_size

        vec_refined[first_line:last_line, 0:1] = np.multiply(vec_refined[first_line:last_line, 0:1],
                                                  np.exp(1j * datum_shift[sstep:sstep + 1, 0:1]))

    # return vec_refined_no_datum_shift, vec_refined
    return vec_refined

#############################################


def create_xml(fname, bands, line, sample, format):

    from isceobj.Util.ImageUtil import ImageLib as IML

    rslc = np.memmap(fname, dtype=np.complex64, mode='w+', shape=(bands, line, sample))
    IML.renderISCEXML(fname, bands, line, sample, format, 'BIL')

    return rslc

##############################################


def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)


def ks2smapletest(S1, S2, threshold=0.4):
    try:
        distance = ecdf_distance(S1, S2)
        if distance <= threshold:
             return 1
        else:
            return 0
    except:
        return 0


def ks2smapletestp(S1, S2, alpha=0.05):
    try:
        distance = ks_2samp(S1, S2)
        if distance[1] >= alpha:
             return 1
        else:
            return 0
    except:
        return 0


def ttest_indtest(S1, S2):
    try:
        test = ttest_ind(S1, S2, equal_var=False)
        if test[1] >= 0.05:
             return 1
        else:
            return 0
    except:
        return 0


def ADtest(S1, S2):
    try:
        test = anderson_ksamp([S1, S2])
        if test.significance_level >= 0.05:
             return 1
        else:
            return 0
    except:
        return 0


#####

def ks_lut(N1, N2, alpha=0.05):
    N = (N1 * N2) / float(N1 + N2)
    distances = np.arange(0.01, 1, 1/1000)
    lamda = distances*(np.sqrt(N) + 0.12 + 0.11/np.sqrt(N))
    alpha_c = np.zeros([len(distances)])
    for value in lamda:
        n = np.ogrid[1:101]
        pvalue = 2*np.sum(((-1)**(n-1))*np.exp(-2*(value**2)*(n**2)))
        pvalue = np.amin(np.amax(pvalue, initial=0), initial=1)
        alpha_c[lamda == value] = pvalue
    critical_distance = distances[alpha_c <= (alpha)]
    return np.min(critical_distance)


def ecdf_distance_old(S1, S2):
    data1 = np.sort(S1.flatten())
    data2 = np.sort(S2.flatten())
    n1 = len(data1)
    n2 = len(data2)
    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side='right') / (1.0 * n1)
    cdf2 = np.searchsorted(data2, data_all, side='right') / (1.0 * n2)
    d = np.max(np.absolute(cdf1 - cdf2))
    return d


def ecdf_distance(data):
    data_all = np.array(data).flatten()
    n1 = int(len(data_all)/2)
    data1 = data_all[0:n1]
    data2 = data_all[n1::]
    data_all = np.sort(data_all)
    cdf1 = np.searchsorted(data1, data_all, side='right') / (1.0 * n1)
    cdf2 = np.searchsorted(data2, data_all, side='right') / (1.0 * n1)
    di = np.max(np.absolute(cdf1 - cdf2))
    return di


def affine_transform(rows, cols, ovs_x, ovs_y):

    pts1 = np.float32([0, 0, rows, 0, rows, cols, 0, cols])
    pts2 = np.float32([[0, 0], [rows * ovs_x, 0], [rows * ovs_x, cols * ovs_y], [0, cols * ovs_y]])

    M = np.zeros([len(pts1), 4])
    for t in range(4):
        M[2*t, 0] = pts2[t, 0]
        M[2*t, 1] = -pts2[t, 1]
        M[2*t, 2] = 1
        M[2*t, 3] = 0
        M[2*t+1, 0] = pts2[t, 1]
        M[2*t+1, 1] = pts2[t, 0]
        M[2*t+1, 2] = 0
        M[2*t+1, 3] = 1

    X = np.matmul(np.linalg.pinv(M), np.transpose(pts1))
    X = np.matrix([[X[0], -X[1], X[2]], [X[1], X[0], X[3]]])
    return X


def find_coord(transform, coord, rows, cols):
    M = np.ones([3, 2])
    M[0, 0] = coord[0]
    M[1, 0] = coord[1]
    M[:, 1] = M[:, 0]
    M[2, 1] = 0
    out = np.floor(np.matmul(transform, M))
    out[out < 0] = 0
    if out[1, 1] > cols:
        out[1, 1] = cols
    if out[0, 0] > rows:
        out[0, 0] = rows

    return int(out[0, 0]), int(out[1, 1])


def apply_affine(rows, cols, ovs_x, ovs_y, transf):

    out_row = np.zeros([rows*ovs_x, cols*ovs_y])
    out_col = np.zeros([rows*ovs_x, cols*ovs_y])

    for row in range(out_row.shape[0]):
        for col in range(out_row.shape[1]):
            row0, col0 = find_coord(transf, [row, col], rows, cols)
            out_row[row, col] = row0
            out_col[row, col] = col0

    return out_row, out_col

#################################


def email_minopy(work_dir):
    """ email mintpy results """

    import subprocess
    import sys

    email_address = os.getenv('NOTIFICATIONEMAIL')

    textStr = 'email mintpy results'

    cwd = os.getcwd()

    pic_dir = os.path.join(work_dir, 'pic')
    flist = ['avgPhaseVelocity.png', 'avgSpatialCoh.png', 'geo_maskTempCoh.png', 'geo_temporalCoherence.png',
             'geo_velocity.png', 'maskConnComp.png', 'Network.pdf', 'BperpHistory.pdf', 'CoherenceMatrix.pdf',
             'rms_timeseriesResidual_ramp.pdf', 'geo_velocity.kmz']

    file_list = [os.path.join(pic_dir, i) for i in flist]
    print(file_list)

    attachmentStr = ''
    i = 0
    for fileList in file_list:
        i = i + 1
        attachmentStr = attachmentStr + ' -a ' + fileList

    mailCmd = 'echo \"' + textStr + '\" | mail -s ' + cwd + ' ' + attachmentStr + ' ' + email_address
    command = 'ssh pegasus.ccs.miami.edu \"cd ' + cwd + '; ' + mailCmd + '\"'
    print(command)
    status = subprocess.Popen(command, shell=True).wait()
    if status is not 0:
        sys.exit('Error in email_minopy')

    return

#################################


def invert_ifgrams_to_timeseries(template, inps_dict, work_dir, writefile):

    ## 1. input info
    inps = inps_dict
    inps.timeseriesFile = os.path.join(work_dir, 'timeseries.h5')
    inps.tempCohFile = os.path.join(work_dir, 'temporalCoherence.h5')
    inps.timeseriesFiles = [os.path.join(work_dir, 'timeseries.h5')]  # all ts files
    inps.numInvFile = os.path.join(work_dir, 'numInvIfgram.h5')

    ifgram_file = os.path.join(work_dir, 'inputs/ifgramStack.h5')

    stack_obj = ifgramStack(ifgram_file)
    stack_obj.open(print_msg=False)

    date12_list = stack_obj.get_date12_list(dropIfgram=True)
    date_list = stack_obj.get_date_list(dropIfgram=False)

    if template['MINOPY.interferograms.referenceDate']:
        reference_date = template['MINOPY.interferograms.referenceDate']
    else:
        reference_date = date_list[0]

    if template['MINOPY.interferograms.type'] == 'sequential':
        reference_ind = False
    else:
        reference_ind = date_list.index(reference_date)

    # 1.2 design matrix
    A = stack_obj.get_design_matrix4timeseries(date12_list=date12_list, refDate=reference_date)[0]
    num_ifgram, num_date = A.shape[0], A.shape[1] + 1
    inps.numIfgram = num_ifgram
    length, width = stack_obj.length, stack_obj.width

    ## 2. prepare output

    # 2.1 metadata
    metadata = dict(stack_obj.metadata)
    metadata['REF_DATE'] = date_list[0]
    metadata['FILE_TYPE'] = 'timeseries'
    metadata['UNIT'] = 'm'

    # 2.2 instantiate time-series
    dsNameDict = {
        "date": (np.dtype('S8'), (num_date,)),
        "bperp": (np.float32, (num_date,)),
        "timeseries": (np.float32, (num_date, length, width)),
    }

    ts_obj = timeseries(inps.timeseriesFile)
    ts_obj.layout_hdf5(dsNameDict, metadata)

    # write date time-series
    date_list_utf8 = [dt.encode('utf-8') for dt in date_list]
    writefile.write_hdf5_block(inps.timeseriesFile, date_list_utf8, datasetName='date')

    # write bperp time-series
    pbase = stack_obj.get_perp_baseline_timeseries(dropIfgram=True)
    writefile.write_hdf5_block(inps.timeseriesFile, pbase, datasetName='bperp')

    # 2.3 instantiate temporal coherence
    dsNameDict = {"temporalCoherence": (np.float32, (length, width))}
    metadata['FILE_TYPE'] = 'temporalCoherence'
    metadata['UNIT'] = '1'
    metadata.pop('REF_DATE')
    writefile.layout_hdf5(inps.tempCohFile, dsNameDict, metadata=metadata)

    # 2.4 instantiate number of inverted observations
    dsNameDict = {"mask": (np.float32, (length, width))}
    metadata['FILE_TYPE'] = 'mask'
    metadata['UNIT'] = '1'
    writefile.layout_hdf5(inps.numInvFile, dsNameDict, metadata=metadata)

    ## 3. run the inversion / estimation and write to disk
    # 3.1 split ifgram_file into blocks to save memory

    box_list, num_box = split2boxes(ifgram_file)

    # --dset option
    unwDatasetName = 'unwrapPhase'

    # determine suffix based on unwrapping error correction method
    obs_suffix_map = {'bridging': '_bridging',
                      'phase_closure': '_phaseClosure',
                      'bridging+phase_closure': '_bridging_phaseClosure'}
    key = 'mintpy.unwrapError.method'
    if key in template.keys() and template[key]:
        unw_err_method = template[key].lower().replace(' ', '')  # fix potential typo
        unwDatasetName += obs_suffix_map[unw_err_method]
        print('phase unwrapping error correction "{}" is turned ON'.format(unw_err_method))

    mask_dataset_name = template['mintpy.networkInversion.maskDataset']
    mask_threshold = float(template['mintpy.networkInversion.maskThreshold'])

    ref_phase = stack_obj.get_reference_phase(unwDatasetName=unwDatasetName,
                                              skip_reference=True,
                                              dropIfgram=False)

    phase2range = -1 * float(metadata['WAVELENGTH']) / (4. * np.pi)

    quality_name = template['quality_file']
    quality = np.memmap(quality_name, mode='r', dtype='float32', shape=(length, width))

    for i in range(num_box):
        box = box_list[i]
        num_row = box[3] - box[1]
        num_col = box[2] - box[0]
        num_pixel = num_row * num_col

        temp_coh = quality[box[1]:box[3], box[0]:box[2]]

        print('\n------- Processing Patch {} out of {} --------------'.format(i + 1, num_box))

        # Read/Mask unwrapPhase
        pha_data = read_unwrap_phase(stack_obj,
                                     box,
                                     ref_phase,
                                     obs_ds_name=unwDatasetName,
                                     dropIfgram=True)

        pha_data = mask_unwrap_phase(pha_data,
                                     stack_obj,
                                     box,
                                     dropIfgram=True,
                                     mask_ds_name=mask_dataset_name,
                                     mask_threshold=mask_threshold)

        # Mask for pixels to invert
        mask = np.ones(num_pixel, np.bool_)

        # Mask for Zero Phase in ALL ifgrams
        if 'phase' in unwDatasetName.lower():
            print('skip pixels with zero/nan value in all interferograms')
            with warnings.catch_warnings():
                # ignore warning message for all-NaN slices
                warnings.simplefilter("ignore", category=RuntimeWarning)
                phase_stack = np.nanmean(pha_data, axis=0)
            mask *= np.multiply(~np.isnan(phase_stack), phase_stack != 0.)
            del phase_stack

        num_pixel2inv = int(np.sum(mask))
        idx_pixel2inv = np.where(mask)[0]
        print('number of pixels to invert: {} out of {} ({:.1f}%)'.format(
            num_pixel2inv, num_pixel, num_pixel2inv / num_pixel * 100))

        # initiale the output matrices
        ts = np.zeros((num_date, num_pixel), np.float32)
        num_inv_ifg = np.zeros((num_row, num_col), np.int16) + num_ifgram

        if num_pixel2inv < 1:
            ts = ts.reshape(num_date, num_row, num_col)
        else:

            # Mask for Non-Zero Phase in ALL ifgrams (share one B in sbas inversion)
            mask_all_net = np.all(pha_data, axis=0)
            mask_all_net *= mask
            # mask_all_net *= mask_Coh
            idx_pixel2inv = np.where(mask_all_net)[0]

            if np.sum(mask_all_net) > 0:
                tsi = LA.lstsq(A, pha_data[:, mask_all_net], cond=1e-5)[0]

            ts[0:reference_ind, idx_pixel2inv] = tsi[0:reference_ind, :]
            ts[reference_ind + 1::, idx_pixel2inv] = tsi[reference_ind::, :]
            ts = ts.reshape(num_date, num_row, num_col)

        print('converting phase to range')
        ts *= phase2range

        block = [0, num_date, box[1], box[3], box[0], box[2]]
        writefile.write_hdf5_block(inps.timeseriesFile,
                                   data=ts,
                                   datasetName='timeseries',
                                   block=block)

        # temporal coherence - 2D
        block = [box[1], box[3], box[0], box[2]]
        writefile.write_hdf5_block(inps.tempCohFile,
                                   data=temp_coh,
                                   datasetName='temporalCoherence',
                                   block=block)

        # number of inverted obs - 2D
        writefile.write_hdf5_block(inps.numInvFile,
                                   data=num_inv_ifg,
                                   datasetName='mask',
                                   block=block)

    # 4 update output data on the reference pixel
    inps.skip_ref = True  # temporary
    if not inps.skip_ref:
        # grab ref_y/x
        ref_y = int(stack_obj.metadata['REF_Y'])
        ref_x = int(stack_obj.metadata['REF_X'])
        print('-' * 50)
        print('update values on the reference pixel: ({}, {})'.format(ref_y, ref_x))

        print('set temporal coherence on the reference pixel to 1.')
        with h5py.File(inps.tempCohFile, 'r+') as f:
            f['temporalCoherence'][ref_y, ref_x] = 1.
    return

################################################################


def get_latest_template(work_dir):
    from minopy.objects.read_template import Template

    """Get the latest version of default template file.
    If an obsolete file exists in the working directory, the existing option values are kept.
    """
    lfile = os.path.join(os.path.dirname(__file__), 'defaults/minopy_template.cfg')  # latest version
    cfile = os.path.join(work_dir, 'minopy_template.cfg')  # current version
    if not os.path.isfile(cfile):
        print('copy default template file {} to work directory'.format(lfile))
        shutil.copy2(lfile, work_dir)
    else:
        # read custom template from file
        cdict = Template(cfile).options
        ldict = Template(lfile).options

        if any([key not in cdict.keys() for key in ldict.keys()]):
            print('obsolete default template detected, update to the latest version.')
            shutil.copy2(lfile, work_dir)
            orig_dict = Template(cfile).options
            for key, value in orig_dict.items():
                if key in cdict.keys() and cdict[key] != value:
                    update = True
                else:
                    update = False
            if not update:
                print('No new option value found, skip updating ' + cfile)
                return cfile

            # Update template_file with new value from extra_dict
            tmp_file = cfile + '.tmp'
            f_tmp = open(tmp_file, 'w')
            for line in open(cfile, 'r'):
                c = [i.strip() for i in line.strip().split('=', 1)]
                if not line.startswith(('%', '#')) and len(c) > 1:
                    key = c[0]
                    value = str.replace(c[1], '\n', '').split("#")[0].strip()
                    if key in cdict.keys() and cdict[key] != value:
                        line = line.replace(value, cdict[key], 1)
                        print('    {}: {} --> {}'.format(key, value, cdict[key]))
                f_tmp.write(line)
            f_tmp.close()

            # Overwrite exsting original template file
            mvCmd = 'mv {} {}'.format(tmp_file, cfile)
            os.system(mvCmd)
    return cfile

################################################################


def get_phase_linking_coherence_mask(template, work_dir, functions):
    """
    Generate reliable pixel mask from temporal coherence
    functions = [generate_mask, readfile, run_or_skip, add_attribute]
    # from mintpy import generate_mask
    # from mintpy.utils import readfile
    # from mintpy.utils.utils import run_or_skip, add_attribute
    """

    generate_mask = functions[0]
    readfile = functions[1]
    run_or_skip = functions[2]
    add_attribute = functions[3]

    tcoh_file = os.path.join(work_dir, 'temporalCoherence.h5')
    water_mask_file = os.path.join(work_dir, 'waterMask.h5')
    mask_file = os.path.join(work_dir, 'maskTempCoh.h5')
    
    if os.path.exists(water_mask_file):
        f1 = h5py.File(tcoh_file, 'a')
        f2 = h5py.File(water_mask_file, 'r')
        water_mask = f2['waterMask']
        f1['temporalCoherence'][:, :] = np.multiply(f1['temporalCoherence'], water_mask)
        f1.close()
        f2.close()

    tcoh_min = float(template['mintpy.networkInversion.minTempCoh'])

    scp_args = '{} -m {} --nonzero -o {} --update'.format(tcoh_file, tcoh_min, mask_file)
    print('generate_mask.py', scp_args)

    # update mode: run only if:
    # 1) output file exists and newer than input file, AND
    # 2) all config keys are the same

    print('update mode: ON')
    flag = 'skip'
    if run_or_skip(out_file=mask_file, in_file=tcoh_file, print_msg=False) == 'run':
        flag = 'run'

    print('run or skip: {}'.format(flag))

    if flag == 'run':
        generate_mask.main(scp_args.split())
        # update configKeys
        atr = {}
        atr['mintpy.networkInversion.minTempCoh'] = tcoh_min
        add_attribute(mask_file, atr)
        add_attribute(mask_file, atr)

    # check number of pixels selected in mask file for following analysis
    num_pixel = np.sum(readfile.read(mask_file)[0] != 0.)
    print('number of reliable pixels: {}'.format(num_pixel))

    min_num_pixel = float(template['mintpy.networkInversion.minNumPixel'])
    if num_pixel < min_num_pixel:
        msg = "Not enough reliable pixels (minimum of {}). ".format(int(min_num_pixel))
        msg += "Try the following:\n"
        msg += "1) Check the reference pixel and make sure it's not in areas with unwrapping errors\n"
        msg += "2) Check the network and make sure it's fully connected without subsets"
        raise RuntimeError(msg)
    return

################################################################


def update_or_skip_inversion(inverted_date_list, slc_dates):

    with open(inverted_date_list, 'r') as f:
        inverted_dates = f.readlines()

    inverted_dates = [date.split('\n')[0] for date in inverted_dates]
    new_slc_dates = list(set(slc_dates) - set(inverted_dates))
    all_date_list = new_slc_dates + inverted_dates

    updated_index = None
    if inverted_dates == slc_dates:
        print(('All date exists in file {} with same size as required,'
               ' no need to update inversion.'.format(os.path.basename(inverted_date_list))))
    elif len(slc_dates) < 10 + len(inverted_dates):
        print('Number of new images is less than 10 --> wait until at least 10 images are acquired')

    else:
        updated_index = len(inverted_dates)

    return updated_index, all_date_list

#########################################################


def multilook(infile, outfile, rlks, alks, multilook_tool='gdal'):

    if multilook_tool == "gdal":

        print(infile)
        ds = gdal.Open(infile + ".vrt", gdal.GA_ReadOnly)

        xSize = ds.RasterXSize
        ySize = ds.RasterYSize

        outXSize = xSize / int(rlks)
        outYSize = ySize / int(alks)

        gdalTranslateOpts = gdal.TranslateOptions(format="ENVI", width=outXSize, height=outYSize)

        gdal.Translate(outfile, ds, options=gdalTranslateOpts)
        ds = None

        ds = gdal.Open(outfile, gdal.GA_ReadOnly)
        gdal.Translate(outfile + ".vrt", ds, options=gdal.TranslateOptions(format="VRT"))
        ds = None

    else:

        print('Multilooking {0} ...'.format(infile))

        inimg = isceobj.createImage()
        inimg.load(infile + '.xml')

        lkObj = Looks()
        lkObj.setDownLooks(alks)
        lkObj.setAcrossLooks(rlks)
        lkObj.setInputImage(inimg)
        lkObj.setOutputFilename(outfile)
        lkObj.looks()

    return outfile