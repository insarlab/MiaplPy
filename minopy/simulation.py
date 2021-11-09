#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os

import time
import numpy as np
from minopy.lib.utils import sequential_phase_linking_py, phase_linking_process_py, datum_connect_py
import argparse
from scipy import linalg as LA
from scipy.linalg import lapack as lap
import matplotlib.pyplot as plt


def create_parser():
    """ Creates command line argument parser object. """

    parser = argparse.ArgumentParser(description='phase linking simulation and assessment')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('-l', '--lambda', dest='lamda', type=float, default=56.0, help='Sensor wavelength (mm)')
    parser.add_argument('-ns', '--nshp', dest='n_shp', type=int, default=300, help='Number of neighbouring samples')
    parser.add_argument('-ni', '--nimg', dest='n_img', type=int, default=100, help='Number of images')
    parser.add_argument('-dd', '--decorr_days', dest='decorr_days', type=int, default=50, help='Decorrelatopn days')
    parser.add_argument('-df', '--decorr_days_fading', dest='decorr_days_fading', type=int, default=11,
                        help='Decorrelatopn days_fading')
    parser.add_argument('-tb', '--tmp_bl', dest='tmp_bl', type=int, default=6, help='Temporal baseline')
    parser.add_argument('-dr', '--def_rate', dest='deformation_rate', type=float, default=4,
                        help='Linear deformation rate. -- Default : 4 mm/y')
    parser.add_argument('-fr', '--fading_rate', dest='fading_rate', type=float, default=50,
                        help='Fading signal rate. -- Default : 50 mm/y')
    parser.add_argument('-g0', '--gamma0', dest='gamma0', type=float, default=0.6,
                        help='Short temporal coherence. -- Default : 0.6')
    parser.add_argument('-gl', '--gammal', dest='gammal', type=float, default=0.2,
                        help='Long temporal coherence. -- Default : 0.2')
    parser.add_argument('-gf', '--gamma_fading', dest='gamma_fading', type=float, default=0.18,
                        help='Fading signal coherence. -- Default : 0.18')
    #parser.add_argument('-st', '--signal_type', dest='signal_type', type=str, default='linear',
    #                    help='(linear or nonlinear) deformation signal')
    parser.add_argument('-nr', '--n_sim', dest='n_sim', type=int, default=1000, help='Number of simulation')
    parser.add_argument('--o', '--out_dir', dest='out_dir', type=str, default='./simulation', help='output directory')
    parser.add_argument('--se', action='store_true', dest='seasonality', default=False,
                        help='add seasonality')
    #parser.add_argument('--m', action='store_true', dest='multistack', default=False,
    #                    help='do multistack for sequential')

    return parser


def command_line_parse(iargs=None):
    """ Parses command line agurments into inps variable. """

    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    return inps


#######################################################################

def CRLB_cov(gama, L):
    """ Estimates the Cramer Rao Lowe Bound based on coherence=gam and ensemble size = L """

    B_theta = np.zeros([len(gama), len(gama) - 1])
    B_theta[1::, :] = np.identity(len(gama) - 1)

    invabscoh, uu = LA.lapack.spotrf(np.abs(gama), False, False)
    invabscoh = LA.lapack.spotri(invabscoh)[0]
    invabscoh = np.triu(invabscoh) + np.triu(invabscoh, k=1).T

    X = 2 * L * (np.multiply(np.abs(gama), invabscoh) - np.identity(len(gama)))
    cov_out = np.matmul(np.matmul(B_theta.T, (X + np.identity(len(X)))), B_theta)
    invcov_out, uu = LA.lapack.cpotrf(cov_out, False, False)
    invcov_out = LA.lapack.cpotri(invcov_out)[0]
    invcov_out = np.triu(invcov_out) + np.triu(invcov_out, k=1).T
    return invcov_out


def rgbf():
    wl = np.arange(380., 781.)
    gamma = 0.80
    factor_wl = np.select([wl > 700., wl < 420., True],
                          [.3 + .7 * (780. - wl) / (780. - 700.), 3 + .7 * (wl - 380.) / (420. - 380.), 1.0])
    raw_r_wl = np.select([wl >= 580., wl >= 510., wl >= 440., wl >= 380., True],
                         [1.0, (wl - 510.) / (580. - 510.), 0.0, (wl - 440.) / (380. - 440.), 0.0])
    raw_g_wl = np.select([wl >= 645., wl >= 580., wl >= 490., wl >= 440., True], [0.0, (wl - 645.) / (580. - 645.),
                                                                                  1.0, (wl - 440.) / (490. - 440.),
                                                                                  0.0])
    raw_b_wl = np.select([wl >= 510., wl >= 490., wl >= 380., True], [0.0, (wl - 510.) / (490. - 510.), 1.0, 0.0])

    correct_r = np.power(factor_wl * raw_r_wl, gamma)
    correct_g = np.power(factor_wl * raw_g_wl, gamma)
    correct_b = np.power(factor_wl * raw_b_wl, gamma)

    return np.transpose([correct_r, correct_g, correct_b])


def simulate_neighborhood_stack(corr_matrix, neighborSamples=300):
    """Simulating the neighbouring pixels (SHPs) based on a given coherence matrix"""

    numberOfSlc = corr_matrix.shape[0]
    # A 2D matrix for a neighborhood over time. Each column is the neighborhood complex data for each acquisition date

    neighbor_stack = np.zeros((numberOfSlc, neighborSamples), dtype=np.complex64)
    for ii in range(neighborSamples):
        cpxSLC = simulate_noise(corr_matrix)
        neighbor_stack[:, ii] = cpxSLC

    return neighbor_stack


def simulate_noise(corr_matrix):
    nsar = corr_matrix.shape[0]
    eigen_value, eigen_vector = lap.cheevx(corr_matrix)[0:2]
    msk = (eigen_value < 1e-3)
    eigen_value[msk] = 0.
    # corr_matrix =  np.dot(eigen_vector, np.dot(np.diag(eigen_value), np.matrix.getH(eigen_vector)))

    # C = np.linalg.cholesky(corr_matrix)
    CM = np.matmul(eigen_vector, np.matmul(np.diag(np.sqrt(eigen_value)), np.conj(eigen_vector.T)))
    Zr = (np.random.randn(nsar) + 1j * np.random.randn(nsar))/np.sqrt(2)
    noise = np.matmul(CM, Zr)
    return noise


def EST_rms(x):
    """ Estimate Root mean square error."""

    out = np.sqrt(np.sum(x ** 2, axis=1) / (np.shape(x)[1] - 1))

    return out


def simulate_constant_vel_phase(n_img=100, tmp_bl=6):
    """ Simulate Interferogram with constant velocity deformation rate """
    t = np.ogrid[0:(tmp_bl * n_img):tmp_bl]
    x = t / 365
    return t, x


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


def double_solve(f1, f2, x0, y0):
    """Solve for two equation with two unknowns using iterations"""

    from scipy.optimize import fsolve
    func = lambda x: [f1(x[0], x[1]), f2(x[0], x[1])]
    return fsolve(func, [x0, y0])


def simulate_coherence_matrix_exponential(t, gamma0, gammaf, gamma_fading, vel_phase, decorr_days,
                                                        vel_fading, decorr_days_fading, seasonal=False):
    """Simulate a Coherence matrix based on de-correlation rate, phase and dates"""
    # t: a vector of acquistion times
    # ph: a vector of simulated phase time-series for one pixel
    # returns the complex covariance matrix
    # corr_mat = (gamma0-gammaf)*np.exp(-np.abs(days_mat/decorr_days))+gammaf

    length = t.shape[0]
    C = np.ones((length, length), dtype=np.complex64)

    if seasonal:
        f1 = lambda x, y: (x - y) ** 2 - gammaf
        f2 = lambda x, y: (x + y) ** 2 - gamma0
        res = double_solve(f1, f2, 0.5, 0.5)
        A = res[0]
        B = res[1]

    for ii in range(length):
        for jj in range(ii + 1, length):

            factor1 = (gamma0 - gammaf) * np.exp(-np.abs(t[ii] - t[jj]) / decorr_days) + gammaf
            factor2 = gamma_fading * np.exp(-np.abs(t[ii] - t[jj]) / decorr_days_fading)

            if seasonal:
                factor1 = (A + B * np.cos(2 * np.pi * t[ii] / 180)) * (A + B * np.cos(2 * np.pi * t[jj] / 180))

            ph0 = vel_phase * (t[jj] - t[ii])
            fading = vel_fading * (t[jj] - t[ii])
            C[ii, jj] = factor1 * np.exp(1j * ph0) + factor2 * np.exp(1j * fading)
            C[jj, ii] = np.conj(C[ii, jj])

    return C


def repeat_simulation(numr, n_img, n_shp, phas, coh_sim_S, coh_sim_L, outname): #, stacknumber=1):

    Timesmat = np.zeros([14, numr])

    EVD_est_resS = np.zeros([n_img, numr])
    EVD_est_resL = np.zeros([n_img, numr])
    EMI_est_resS = np.zeros([n_img, numr])
    EMI_est_resL = np.zeros([n_img, numr])
    PTA_est_resS = np.zeros([n_img, numr])
    PTA_est_resL = np.zeros([n_img, numr])
    stbas_est_resS = np.zeros([n_img, numr])
    stbas_est_resL = np.zeros([n_img, numr])

    EVD_seq_est_resS = np.zeros([n_img, numr])
    EVD_seq_est_resL = np.zeros([n_img, numr])
    EMI_seq_est_resS = np.zeros([n_img, numr])
    EMI_seq_est_resL = np.zeros([n_img, numr])
    PTA_seq_est_resS = np.zeros([n_img, numr])
    PTA_seq_est_resL = np.zeros([n_img, numr])


    for t in range(numr):
        if np.mod(t, 10) == 0:
            print('Iteration: ', str(t))

        CCGsam_Sterm = simulate_neighborhood_stack(coh_sim_S, neighborSamples=n_shp)
        CCGsam_Lterm = simulate_neighborhood_stack(coh_sim_L, neighborSamples=n_shp)

        time0 = time.time()
        ####
        ph_stbas, noval, temp_quality = phase_linking_process_py(CCGsam_Sterm, 0, b'StBAS', False, 4)
        stbas_est_resS[:, t:t + 1] = np.angle(np.array(ph_stbas).reshape(-1, 1) * np.exp(-1j * phas))
        time01 = time.time()
        ph_stbas, noval, temp_quality = phase_linking_process_py(CCGsam_Lterm, 0, b'StBAS', False, 4)
        stbas_est_resL[:, t:t + 1] = np.angle(np.array(ph_stbas).reshape(-1, 1) * np.exp(-1j * phas))
        time02 = time.time()

        ####
        ph_EVD, noval, temp_quality = phase_linking_process_py(CCGsam_Sterm, 0, b'EVD', False, 0)
        EVD_est_resS[:, t:t + 1] = np.angle(np.array(ph_EVD).reshape(-1, 1) * np.exp(-1j * phas))
        time1 = time.time()
        ph_EVD, noval, temp_quality = phase_linking_process_py(CCGsam_Lterm, 0, b'EVD', False, 0)
        EVD_est_resL[:, t:t + 1] = np.angle(np.array(ph_EVD).reshape(-1, 1) * np.exp(-1j * phas))
        time2 = time.time()

        ####
        ph_EMI, noval, temp_quality = phase_linking_process_py(CCGsam_Sterm, 0, b'EMI', False, 0)
        EMI_est_resS[:, t:t + 1] = np.angle(np.array(ph_EMI).reshape(-1, 1) * np.exp(-1j * phas))
        time3 = time.time()

        ph_EMI, noval, temp_quality = phase_linking_process_py(CCGsam_Lterm, 0, b'EMI', False, 0)
        EMI_est_resL[:, t:t + 1] = np.angle(np.array(ph_EMI).reshape(-1, 1) * np.exp(-1j * phas))
        time4 = time.time()

        ####
        ph_PTA, noval, temp_quality = phase_linking_process_py(CCGsam_Sterm, 0, b'PTA', False, 0)
        PTA_est_resS[:, t:t + 1] = np.angle(np.array(ph_PTA).reshape(-1, 1) * np.exp(-1j * phas))
        time5 = time.time()

        ph_PTA, noval, temp_quality = phase_linking_process_py(CCGsam_Lterm, 0, b'PTA', False, 0)
        PTA_est_resL[:, t:t + 1] = np.angle(np.array(ph_PTA).reshape(-1, 1) * np.exp(-1j * phas))
        time6 = time.time()

        ####

        num_seq = np.int(n_img // 10)
        ph_vec, sqeezed, temp_quality = sequential_phase_linking_py(CCGsam_Sterm, b'EVD', 10, num_seq)
        ph_vec = datum_connect_py(sqeezed, ph_vec, 10)
        EVD_seq_est_resS[:, t:t + 1] = np.angle(np.array(ph_vec).reshape(-1, 1) * np.exp(-1j * phas))
        time7 = time.time()

        ph_vec, sqeezed, temp_quality = sequential_phase_linking_py(CCGsam_Lterm, b'EVD', 10, num_seq)
        ph_vec = datum_connect_py(sqeezed, ph_vec, 10)
        EVD_seq_est_resL[:, t:t + 1] = np.angle(np.array(ph_vec).reshape(-1, 1) * np.exp(-1j * phas))
        time8 = time.time()

        ####
        ph_vec, sqeezed, temp_quality = sequential_phase_linking_py(CCGsam_Sterm, b'EMI', 10, num_seq)
        ph_vec = datum_connect_py(sqeezed, ph_vec, 10)
        EMI_seq_est_resS[:, t:t + 1] = np.angle(np.array(ph_vec).reshape(-1, 1) * np.exp(-1j * phas))
        time9 = time.time()

        ph_vec, sqeezed, temp_quality = sequential_phase_linking_py(CCGsam_Lterm, b'EMI', 10, num_seq)
        ph_vec = datum_connect_py(sqeezed, ph_vec, 10)
        EMI_seq_est_resL[:, t:t + 1] = np.angle(np.array(ph_vec).reshape(-1, 1) * np.exp(-1j * phas))
        time10 = time.time()


        ####
        ph_vec, sqeezed, temp_quality = sequential_phase_linking_py(CCGsam_Sterm, b'PTA', 10, num_seq)
        ph_vec = datum_connect_py(sqeezed, ph_vec, 10)
        PTA_seq_est_resS[:, t:t + 1] = np.angle(np.array(ph_vec).reshape(-1, 1) * np.exp(-1j * phas))
        time11 = time.time()

        ph_vec, sqeezed, temp_quality = sequential_phase_linking_py(CCGsam_Lterm, b'PTA', 10, num_seq)
        ph_vec = datum_connect_py(sqeezed, ph_vec, 10)
        PTA_seq_est_resL[:, t:t + 1] = np.angle(np.array(ph_vec).reshape(-1, 1) * np.exp(-1j * phas))
        time12 = time.time()

        Timesmat[0, t] = time1 - time01
        Timesmat[1, t] = time2 - time1
        Timesmat[2, t] = time3 - time2
        Timesmat[3, t] = time4 - time3
        Timesmat[4, t] = time5 - time4
        Timesmat[5, t] = time6 - time5
        Timesmat[6, t] = time7 - time6
        Timesmat[7, t] = time8 - time7
        Timesmat[8, t] = time9 - time8
        Timesmat[9, t] = time10 - time9
        Timesmat[10, t] = time11 - time10
        Timesmat[11, t] = time12 - time11
        Timesmat[12, t] = time01 - time0
        Timesmat[13, t] = time02 - time01

    rmsemat_est = np.zeros([n_img, 14])

    rmsemat_est[:, 0] = EST_rms(EVD_est_resS)
    rmsemat_est[:, 1] = EST_rms(EVD_est_resL)
    rmsemat_est[:, 2] = EST_rms(EMI_est_resS)
    rmsemat_est[:, 3] = EST_rms(EMI_est_resL)
    rmsemat_est[:, 4] = EST_rms(PTA_est_resS)
    rmsemat_est[:, 5] = EST_rms(PTA_est_resL)

    rmsemat_est[:, 6] = EST_rms(EVD_seq_est_resS)
    rmsemat_est[:, 7] = EST_rms(EVD_seq_est_resL)
    rmsemat_est[:, 8] = EST_rms(EMI_seq_est_resS)
    rmsemat_est[:, 9] = EST_rms(EMI_seq_est_resL)
    rmsemat_est[:, 10] = EST_rms(PTA_seq_est_resS)
    rmsemat_est[:, 11] = EST_rms(PTA_seq_est_resL)

    rmsemat_est[:, 12] = EST_rms(stbas_est_resS)
    rmsemat_est[:, 13] = EST_rms(stbas_est_resL)

    out_time = np.mean(Timesmat, axis=1)
    out_time_name = outname.split('.npy')[0] + '_time.npy'

    np.save(outname, rmsemat_est)
    np.save(out_time_name, out_time)

    return None

    ####################################


def simulate_and_calculate_different_method_rms(iargs=None):
    inps = command_line_parse(iargs)
    simul_dir = inps.out_dir
    if not os.path.isdir(simul_dir):
        os.mkdir(simul_dir)

    outname = 'rmsemat_modifiedSignalEq_linear'
    if inps.seasonality:
        outname = outname + '_seasonal'

    outname = outname + '.npy'

    inps.outname = os.path.join(simul_dir, outname)

    vel_phase = inps.deformation_rate / 365 * 4 * np.pi / inps.lamda
    vel_fading = inps.fading_rate / 365 * 4 * np.pi / inps.lamda  # 0.031 # rad/day

    temp_baseline = np.ogrid[0:(inps.tmp_bl * inps.n_img):inps.tmp_bl]

    #if inps.signal_type == 'linear':
    #    temp_baseline, displacement = simulate_constant_vel_phase(inps.n_img, inps.tmp_bl)
    #else:
    #    temp_baseline, displacement = simulate_volcano_def_phase(inps.n_img, inps.tmp_bl)

    ph0 = -vel_phase * (temp_baseline)
    gamma_l = 0
    inps.coh_sim_S = simulate_coherence_matrix_exponential(temp_baseline, inps.gamma0, gamma_l, inps.gamma_fading,
                                                           vel_phase, inps.decorr_days,
                                                           vel_fading, inps.decorr_days_fading, seasonal=inps.seasonality)


    gamma_l = inps.gammal
    inps.coh_sim_L = simulate_coherence_matrix_exponential(temp_baseline, inps.gamma0, gamma_l, inps.gamma_fading,
                                                           vel_phase, inps.decorr_days,
                                                           vel_fading, inps.decorr_days_fading, seasonal=inps.seasonality)

    repeat_simulation(numr=inps.n_sim, n_img=inps.n_img, n_shp=inps.n_shp,
                      phas=ph0.reshape(-1, 1), coh_sim_S=inps.coh_sim_S, coh_sim_L=inps.coh_sim_L, outname=inps.outname)
                      #stacknumber=stacknum)

    return


if __name__ == '__main__':
    '''
    Simulates the phase linking process

    '''
    simulate_and_calculate_different_method_rms()
