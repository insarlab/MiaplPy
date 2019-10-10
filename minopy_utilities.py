#! /usr/bin/env python3
###############################################################################
# Project: Utilitiels for minopy
# Author: Sara Mirzaee
# Created: 10/2018
###############################################################################
from mintpy.prep_isce import extract_tops_metadata, extract_geometry_metadata
import sys
import os
import numpy as np
import cmath
import argparse
import glob
from numpy import linalg as LA
from scipy.optimize import minimize, Bounds
from scipy import stats
from scipy.stats import ks_2samp, anderson_ksamp, ttest_ind
import gdal
import isce
import isceobj
from mintpy.utils import readfile
from minsar.objects.auto_defaults import PathFind
import minsar.utils.process_utilities as putils


pathObj = PathFind()
################################################################################


def cmd_line_parse(iargs=None, script=None):
    """Command line parser."""

    parser = argparse.ArgumentParser(description='MiNoPy scripts parser')
    parser = add_common_parser(parser)

    if script == 'patch_inversion':
        parser = add_patch_inversion(parser)
    if script == 'timeseries_corrections':
        parser = add_mintpy_corrections(parser)
    if script == 'oversample_minopy':
        parser = add_oversample(parser)

    inps = parser.parse_args(args=iargs)
    inps = putils.create_or_update_template(inps)

    return inps


def add_common_parser(parser):

    commonp = parser.add_argument_group('General options:')
    commonp.add_argument('custom_template_file', nargs='?', help='custom template with option settings.\n')
    commonp.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    commonp.add_argument('--submit', dest='submit_flag', action='store_true', help='submits job')
    commonp.add_argument('--walltime', dest='wall_time', default='None',
                        help='walltime for submitting the script as a job')
    commonp.add_argument('--wait', dest='wait_time', default='00:00', metavar="Wait time (hh:mm)",
                         help="wait time to submit a job")
    return parser


def add_oversample(parser):
    crp = parser.add_argument_group('Crop options:')
    crp.add_argument('--ox', dest='over_sample_x', default=3, help='Oversampling in azimuth direction')
    crp.add_argument('--oy', dest='over_sample_y', default=1, help='Oversampling in range direction')
    return parser


def add_minopy_wrapper(parser):

    STEP_LIST, STEP_HELP = pathObj.minopy_help()

    minp = parser.add_argument_group('MiNoPy Routine InSAR Time Series Analysis. steps processing '
                                    '(start/end/step)', STEP_HELP)
    minp.add_argument('--remove_minopy_dir', dest='remove_minopy_dir', action='store_true',
                     help='remove directory before download starts')
    minp.add_argument('--start', dest='startStep', metavar='STEP', default=STEP_LIST[0],
                      help='start processing at the named step, default: {}'.format(STEP_LIST[0]))
    minp.add_argument('--stop', dest='endStep', metavar='STEP', default=STEP_LIST[-1],
                      help='end processing at the named step, default: {}'.format(STEP_LIST[-1]))
    minp.add_argument('--step', dest='step', metavar='STEP',
                      help='run processing at the named step only')
    minp.add_argument('--email', action='store_true', dest='email', default=False,
                    help='opt to email results')

    return parser


def add_patch_inversion(parser):

    pi = parser.add_argument_group('Patch inversion option')
    pi.add_argument('-p', '--patch', type=str, dest='patch', help='patch directory')

    return parser


def add_mintpy_corrections(parser):

    corrections = parser.add_argument_group('Mintpy options')

    corrections.add_argument('--dir', dest='work_dir',
                        help='MintPy working directory, default is:\n' +
                             'a) current directory, or\n' +
                             'b) $SCRATCHDIR/projectName/mintpy, if meets the following 3 requirements:\n' +
                             '    1) autoPath = True in mintpy/defaults/auto_path.py\n' +
                             '    2) environmental variable $SCRATCHDIR exists\n' +
                             '    3) input custom template with basename same as projectName\n')
    corrections.add_argument('-g', dest='generate_template', action='store_true',
                        help='Generate default template (and merge with custom template), then exit.')
    corrections.add_argument('-H', dest='print_auto_template', action='store_true',
                        help='Print/Show the example template file for routine processing.')

    return parser

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


def convert_geo2image_coord_old(geo_master_dir, master_dir, lat_south, lat_north, lon_west, lon_east):
    """ Finds the corresponding line and sample based on geographical coordinates. """

    import isceobj.Planet.AstronomicalHandbook as AstronomicalHandbook
    from isceobj.Planet.Ellipsoid import Ellipsoid

    master_xml = glob.glob(master_dir + '/IW*.xml')[0]
    metadata = extract_tops_metadata(master_xml)[0]
    gmeta = extract_geometry_metadata(geo_master_dir + '/lat.rdr.full.xml', metadata)
    rmeta = readfile.read_isce_xml(geo_master_dir + '/lat.rdr.full.xml')

    # Read Attributes
    range_n = float(gmeta['startingRange'])
    dR = float(gmeta['rangePixelSize'])
    width = int(rmeta['WIDTH'])
    Re = float(gmeta['earthRadius'])
    Height = float(gmeta['altitude'])
    range_f = range_n + dR * width
    inc_angle_n = (np.pi - np.arccos((Re ** 2 + range_n ** 2 - (Re + Height) ** 2) / (2 * Re * range_n))) * 180.0 / np.pi
    inc_angle_f = (np.pi - np.arccos((Re ** 2 + range_f ** 2 - (Re + Height) ** 2) / (2 * Re * range_f))) * 180.0 / np.pi

    inc_angle = (inc_angle_n + inc_angle_f) / 2.0
    rg_step = float(dR) / np.sin(inc_angle / 180.0 * np.pi)
    az_step = float(gmeta['azimuthPixelSize']) * Re / (Re + Height)

    lat = [lat_south, lat_north]
    lon = [lon_west, lon_east]

    lat_c = (np.nanmax(lat) + np.nanmin(lat)) / 2.
    az_step_deg = 180. / np.pi * az_step / Re
    rg_step_deg = 180. / np.pi * rg_step / (Re * np.cos(lat_c * np.pi / 180.))

    y_factor = 10 * az_step_deg
    x_factor = 10 * rg_step_deg

    ds = gdal.Open(geo_master_dir + '/lat.rdr.full.vrt', gdal.GA_ReadOnly)
    lut_y = ds.GetRasterBand(1).ReadAsArray()

    ds = gdal.Open(geo_master_dir + "/lon.rdr.full.vrt", gdal.GA_ReadOnly)
    lut_x = ds.GetRasterBand(1).ReadAsArray()

    rows = []
    cols = []

    for lat0 in lat:
        for lon0 in lon:
            ymin = lat0 - y_factor;   ymax = lat0 + y_factor
            xmin = lon0 - x_factor;   xmax = lon0 + x_factor

            mask_y = np.multiply(lut_y >= ymin, lut_y <= ymax)
            mask_x = np.multiply(lut_x >= xmin, lut_x <= xmax)
            mask_yx = np.multiply(mask_y, mask_x)
            row, col = np.nanmean(np.where(mask_yx), axis=1)
            rows.append(row)
            cols.append(col)

    first_row = np.rint(np.min(rows)).astype(int)
    last_row = np.rint(np.max(rows)).astype(int)
    first_col = np.rint(np.min(cols)).astype(int)
    last_col = np.rint(np.max(cols)).astype(int)

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


def gam_pta(ph_filt, ph_refined):
    """ Returns squeesar PTA coherence between the initial and estimated phase vectors. """

    n = np.shape(ph_filt)[0]
    indx = np.triu_indices(n, 1)
    phi_mat = ph_filt[indx]
    g1 = np.exp(1j * ph_refined).reshape(n, 1)
    g2 = np.exp(-1j * ph_refined).reshape(1, n)
    theta_mat = np.angle(np.matmul(g1, g2))
    theta_mat = theta_mat[indx]
    ifgram_diff = phi_mat - theta_mat
    temp_coh = np.real(np.sum(np.exp(1j * ifgram_diff))) * 2 / (n ** 2 - n)

    return temp_coh

###############################################################################


def optphase(x0, inverse_gam):
    """ Returns the PTA maximum likelihood function value. """

    n = len(x0)
    x = np.ones([n+1, 1])+0j
    x[1::, 0] = np.exp(1j*x0[:])
    x = np.matrix(x)
    y = -np.matmul(x.getH(), inverse_gam)
    y = np.matmul(y, x)
    f = np.abs(np.log(y))

    return f

###############################################################################


def PTA_L_BFGS(xm):
    """ Uses L-BFGS method to optimize PTA function and estimate phase values. """

    n = len(xm)
    x0 = np.zeros([n-1, 1])
    x0[:, 0] = np.real(xm[1::, 0])
    coh = 1j*np.zeros([n, n])
    coh[:, :] = xm[:, 1::]
    abs_coh = regularize_matrix(np.abs(coh))
    if np.size(abs_coh) == np.size(coh):
        inverse_gam = np.matrix(np.multiply(LA.pinv(abs_coh), coh))
        res = minimize(optphase, x0, args=inverse_gam, method='L-BFGS-B',
                       bounds=Bounds(-100, 100, keep_feasible=False),
                       tol=None, options={'gtol': 1e-6, 'disp': False})

        out = np.zeros([n, 1])
        out[1::, 0] = res.x
        #out = np.unwrap(out, np.pi, axis=0)

        return out

    else:

        print('warning: coherence matrix not positive semidifinite, It is switched from PTA to EVD')
        return EVD_phase_estimation(coh)

###############################################################################


def EVD_phase_estimation(coh0):
    """ Estimates the phase values based on eigen value decomosition """

    Eigen_value, Eigen_vector = LA.eigh(coh0)
    f = np.where(np.abs(Eigen_value) == np.sort(np.abs(Eigen_value))[len(coh0)-1])
    vec = Eigen_vector[:, f].reshape(len(Eigen_value), 1)
    x0 = np.angle(vec)
    x0 = x0 - x0[0, 0]
    #x0 = np.unwrap(x0, np.pi, axis=0)

    return x0

###############################################################################


def EMI_phase_estimation(coh0):
    """ Estimates the phase values based on EMI decomosition (Homa Ansari, 2018 paper) """

    abscoh = regularize_matrix(np.abs(coh0))
    if np.size(abscoh) == np.size(coh0):
        M = np.multiply(LA.pinv(abscoh), coh0)
        Eigen_value, Eigen_vector = LA.eigh(M)
        f = np.where(np.abs(Eigen_value) == np.sort(np.abs(Eigen_value))[0])
        vec = Eigen_vector[:, f[0][0]].reshape(Eigen_vector.shape[0], 1)
        x0 = np.angle(vec).reshape(len(Eigen_value), 1)
        x0 = x0 - x0[0, 0]
        #x0 = np.unwrap(x0, np.pi, axis=0)
        return x0
    else:
        print('warning: coherence matrix not positive semidifinite, It is switched from EMI to EVD')
        return EVD_phase_estimation(coh0)

###############################################################################


def test_PS(ccg):
    """ checks if the pixel is PS """

    coh_mat = est_corr(ccg)
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
                factor = (A + B * np.cos(2 * np.pi * t[ii] / 90)) * (A + B * np.cos(2 * np.pi * t[jj] / 90))
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
        neighbor_stack[:,ii] = cpxSLC
    return neighbor_stack

##############################################################################


def double_solve(f1,f2,x0,y0):
    """Solve for two equation with two unknowns using iterations"""

    from scipy.optimize import fsolve
    func = lambda x: [f1(x[0], x[1]), f2(x[0], x[1])]
    return fsolve(func,[x0,y0])

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
        ph_EMI = EMI_phase_estimation(coh_mat)
        xm = np.zeros([len(ph_EMI), len(ph_EMI) + 1]) + 0j
        xm[:, 0:1] = np.reshape(ph_EMI, [len(ph_EMI), 1])
        xm[:, 1::] = coh_mat[:, :]
        res = PTA_L_BFGS(xm)

    elif 'EMI' in method:
        res = EMI_phase_estimation(coh_mat)
    else:
        res = EVD_phase_estimation(coh_mat)

    res = res.reshape(len(res), 1)

    if squeez:
        vm = np.matrix(np.exp(1j * res[stepp::, 0:1]) / LA.norm(np.exp(1j * res[stepp::, 0:1])))
        squeezed = np.matmul(np.conjugate(vm.T), ccg_sample[stepp::, :])
        # squeezed = squeez_im(res[stepp::, 0], ccg_sample[stepp::, :])
        return res, squeezed
    else:
        return res,


def squeez_im(ph, ccg):
    """Squeeze a stack of images in to one (PCA)"""

    vm = np.matrix(np.exp(1j * ph) / LA.norm(np.exp(1j * ph)))
    squeezed = np.complex64(np.matmul(np.conjugate(vm), ccg))
    return squeezed


###############################################################################

def CRLB_cov(gama, L):
    """ Estimates the Cramer Rao Lowe Bound based on coherence=gam and ensemble size = L """

    B_theta = np.zeros([len(gama), len(gama) - 1])
    B_theta[1::, :] = np.identity(len(gama) - 1)
    X = 2 * L * (np.multiply(np.abs(gama), LA.pinv(np.abs(gama))) - np.identity(len(gama)))
    cov_out = LA.pinv(np.matmul(np.matmul(B_theta.T, (X + np.identity(len(X)))), B_theta))

    return cov_out


###############################################################################


def sequential_phase_linking(full_stack_complex_samples, method, num_stack=1):
    """ phase linking of each pixel sequentially and applying a datum shift at the end """

    n_image = full_stack_complex_samples.shape[0]
    mini_stack_size = 10
    num_mini_stacks = np.int(np.floor(n_image / mini_stack_size))
    phas_refined = np.zeros([np.shape(full_stack_complex_samples)[0], 1])

    for step in range(0, num_mini_stacks):

        first_line = step * mini_stack_size
        if step == num_mini_stacks - 1:
            last_line = n_image
        else:
            last_line = first_line + mini_stack_size
        num_lines = last_line - first_line

        if step == 0:
            mini_stack_complex_samples = full_stack_complex_samples[first_line:last_line, :]
            res, squeezed_images = phase_linking_process(mini_stack_complex_samples, step, method)

            phas_refined[first_line:last_line, 0:1] = res[step::, 0:1]
        else:

            if num_stack == 1:
                mini_stack_complex_samples = np.zeros([1 + num_lines, full_stack_complex_samples.shape[1]]) + 1j
                mini_stack_complex_samples[0, :] = np.complex64(squeezed_images[-1, :])
                mini_stack_complex_samples[1::, :] = full_stack_complex_samples[first_line:last_line, :]
                res, new_squeezed_image = phase_linking_process(mini_stack_complex_samples, 1, method)
                phas_refined[first_line:last_line, 0:1] = res[1::, 0:1]
                squeezed_images = np.vstack([squeezed_images, new_squeezed_image])
            else:

                mini_stack_complex_samples = np.zeros([step + num_lines, full_stack_complex_samples.shape[1]]) + 1j
                mini_stack_complex_samples[0:step, :] = np.complex64(squeezed_images)
                mini_stack_complex_samples[step::, :] = full_stack_complex_samples[first_line:last_line, :]
                res, new_squeezed_image = phase_linking_process(mini_stack_complex_samples, step, method)
                phas_refined[first_line:last_line, 0:1] = res[step::, 0:1]
                squeezed_images = np.vstack([squeezed_images, new_squeezed_image])
            ###

    datum_connection_samples = squeezed_images
    datum_shift, squeezed_datum = phase_linking_process(datum_connection_samples, 0, 'EMI')

    # phas_refined_no_datum_shift = np.zeros(np.shape(phas_refined))
    # phas_refined_no_datum_shift[:, :] = phas_refined[:, :]

    for step in range(len(datum_shift)):
        first_line = step * mini_stack_size
        if step == num_mini_stacks - 1:
            last_line = n_image
        else:
            last_line = first_line + mini_stack_size

        phas_refined[first_line:last_line, 0:1] = phas_refined[first_line:last_line, 0:1] - \
                                                  datum_shift[step:step + 1, 0:1]

    # return phas_refined_no_datum_shift, phas_refined
    return phas_refined

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
