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
from numpy import linalg as LA
#from scipy import linalg as LA
from scipy.optimize import minimize, Bounds
from scipy.stats import ks_2samp, anderson_ksamp, ttest_ind
import gdal
import isce
import isceobj
import matplotlib.pyplot as plt

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


def convert_geo2image_coord(geo_master_dir, lat_south, lat_north, lon_west, lon_east):
    """ Finds the corresponding line and sample based on geographical coordinates. """

    ds = gdal.Open(geo_master_dir + '/lat.rdr.full.vrt', gdal.GA_ReadOnly)
    lat_lut = ds.GetRasterBand(1).ReadAsArray()
    del ds

    ds = gdal.Open(geo_master_dir + "/lon.rdr.full.vrt", gdal.GA_ReadOnly)
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
    patch_row = [[patch_row_1], [patch_row_2]]
    patch_cols = [[patch_col_1], [patch_col_2]]
    patchlist = []

    for row in range(len(patch_row_1)):
        for col in range(len(patch_col_1)):
            patchlist.append(str(row) + '_' + str(col))

    return patch_row, patch_cols, patchlist

##############################################################################


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


def read_image(image_file, box=None):
    """ Reads images from isce. """

    ds = gdal.Open(image_file + '.vrt', gdal.GA_ReadOnly)
    if not box is None:
        band = ds.GetRasterBand(1)
        image = band.ReadAsArray()[box[1]:box[3], box[0]:box[2]]
    else:
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
    diagones = np.diag(np.diag(np.ones([nm, nm])))
    phi_mat = np.exp(1j * ph_filt)
    ph_refined = np.angle(vec_refined)
    g1 = np.exp(-1j * ph_refined).reshape(nm, 1)
    g2 = np.exp(1j * ph_refined).reshape(1, nm)
    theta_mat = np.matmul(g1, g2)
    ifgram_diff = np.multiply(phi_mat, theta_mat)
    ifgram_diff = ifgram_diff - np.multiply(ifgram_diff, diagones)
    temp_coh = np.abs(np.sum(ifgram_diff) / (nm ** 2 - nm))

    print(temp_coh)

    return temp_coh

###############################################################################


def optphase(x0, inverse_gam):
    """ Returns the PTA maximum likelihood function value. """

    n = len(x0)
    x = np.exp(1j * x0).reshape(n, 1)
    x = np.matrix(x)
    y = np.matmul(x.getH(), inverse_gam)
    y = np.matmul(y, x)
    f = np.abs(np.log(y))

    return f

###############################################################################


def PTA_L_BFGS(coh0):
    """ Uses L-BFGS method to optimize PTA function and estimate phase values. """

    n = coh0.shape[0]
    x0 = np.angle(EMI_phase_estimation(coh0))
    x0 = x0 - x0[0]
    abs_coh = regularize_matrix(np.abs(coh0))
    if np.size(abs_coh) == np.size(coh0):
        inverse_gam = np.matrix(np.multiply(LA.pinv(abs_coh), coh0))
        res = minimize(optphase, x0, args=inverse_gam, method='L-BFGS-B',
                       tol=None, options={'gtol': 1e-6, 'disp': False})
        out = res.x.reshape(n, 1)
        vec = np.multiply(np.abs(x0), np.exp(1j * out)).reshape(n, 1)

        x0 = np.exp(-1j * np.angle(vec[0]))
        vec = np.multiply(vec, x0)

        return vec

    else:

        print('warning: coherence matrix not positive semidifinite, It is switched from PTA to EVD')
        return EVD_phase_estimation(coh)

###############################################################################


def EVD_phase_estimation(coh0):
    """ Estimates the phase values based on eigen value decomosition """
    eigen_value, eigen_vector = LA.eigh(coh0, UPLO='U')
    vec = eigen_vector[:, -1].reshape(len(eigen_value), 1)
    x0 = np.exp(-1j * np.angle(vec[0]))
    vec = np.multiply(vec, x0)
    return vec

###############################################################################


def EMI_phase_estimation(coh0):
    """ Estimates the phase values based on EMI decomosition (Homa Ansari, 2018 paper) """
    abscoh = regularize_matrix(np.abs(coh0))
    if np.size(abscoh) == np.size(coh0):
        M = np.multiply(LA.pinv(abscoh), coh0)
        eigen_value, eigen_vector = LA.eigh(M, UPLO='U')
        vec = eigen_vector[:, 0].reshape(len(eigen_value), 1)
        x0 = np.exp(-1j * np.angle(vec[0]))
        vec = np.multiply(vec, x0)
        return vec
    else:
        print('warning: coherence matrix not positive semidifinite, It is switched from EMI to EVD')
        return EVD_phase_estimation(coh0)

###############################################################################


def test_PS(coh_mat):
    """ checks if the pixel is PS """

    Eigen_value, Eigen_vector = LA.eigh(coh_mat)
    med_w = np.median(Eigen_value)
    MAD = np.median(np.absolute(Eigen_value - med_w))

    thold1 = np.abs(med_w + 3.5*MAD)
    thold2 = np.abs(med_w - 3.5*MAD)
    treshhold = np.max([thold1, thold2])
    status = len(np.where(np.abs(Eigen_value) > treshhold))

    if status > 0:
        return True
    else:
        return False

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
            gamma = factor * (np.exp((t[ii] - t[jj]) / Tau0) + gammaf)
            C[ii, jj] = gamma * np.exp(1j * (ph[ii] - ph[jj]))
            C[jj, ii] = np.conj(C[ii, jj])

    return C

################################################################################


def simulate_noise(corr_matrix):
    N = corr_matrix.shape[0]

    nsar = corr_matrix.shape[0]
    w, v = np.linalg.eigh(corr_matrix)
    msk = (w < 1e-3)
    w[msk] = 0.
    # corr_matrix =  np.dot(v, np.dot(np.diag(w), np.matrix.getH(v)))

    # C = np.linalg.cholesky(corr_matrix)
    C = np.dot(v, np.dot(np.diag(np.sqrt(w)), np.matrix.getH(v)))
    Z = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
    noise = np.dot(C, Z)

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

    corr_mat = np.matmul(CCGS, CCGS.getH()) / CCGS.shape[1]

    coh = np.multiply(cov2corr(np.abs(corr_mat)), np.exp(1j * np.angle(corr_mat)))

    return coh

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


def ttest_indtest(S1, S2):
    try:
        test = ttest_ind(S1, S2, equal_var=False)
        if test[1] <= 0.05:
             return 1
        else:
            return 0
    except:
        return 0


def ADtest(S1, S2):
    try:
        test = anderson_ksamp([S1, S2])
        if test.significance_level <= 0.05:
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


def ecdf_distance(S1, S2):
    data1 = np.sort(S1.flatten())
    data2 = np.sort(S2.flatten())
    n1 = len(data1)
    n2 = len(data2)
    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side='right') / (1.0 * n1)
    cdf2 = np.searchsorted(data2, data_all, side='right') / (1.0 * n2)
    d = np.max(np.absolute(cdf1 - cdf2))
    return d


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
