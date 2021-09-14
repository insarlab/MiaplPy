#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as LA
from minopy.lib.utils import est_corr_py, sequential_phase_linking_py, phase_linking_process_py
import time
import argparse

# displacement = 1mm/y = (1 mm *4pi / lambda(mm)) rad/y  --> 6 day = 4pi*6/lambda*365
# displacement = lambda*phi/ 4*pi.

def create_parser():
    """ Creates command line argument parser object. """

    parser = argparse.ArgumentParser(description='Squeesar simulation')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('-l', '--lambda', dest='lamda', type=float, default=56.0, help='Sensor wavelength (mm)')
    parser.add_argument('-ns', '--nshp', dest='n_shp', type=int, default=300,help='Number of neighbouring samples')
    parser.add_argument('-ni', '--nimg', dest='n_img', type=int, default=100, help='Number of images')
    parser.add_argument('-dd', '--decorr_days', dest='decorr_days', type=int, default=50, help='Decorrelatopn days')
    parser.add_argument('-tb', '--tmp_bl', dest='tmp_bl', type=int, default=6, help='Temporal baseline')
    parser.add_argument('-dr', '--def_rate', dest='deformation_rate', type=float, default=1 , help='Linear deformation rate. -- Default : 1 mm/y')
    parser.add_argument('-st', '--signal_type', dest='signal_type', type=str, default='linear',
                    help = '(linear or nonlinear) deformation signal')
    parser.add_argument('-nr', '--n_sim', dest='n_sim', type=int, default=1000, help='Number of simulation')
    parser.add_argument('--plot', action='store_true', dest='plot_mat', default=False,
                        help='plot and save coherence matrix')
    parser.add_argument('--se', action='store_true', dest='seasonality', default=False,
                        help='plot and save coherence matrix')


    return parser




def command_line_parse(iargs=None):
    """ Parses command line agurments into inps variable. """

    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    return inps


def main(iargs=None):

    inps = command_line_parse(iargs)
    simul_dir = os.path.join(os.getenv('SCRATCHDIR'),'simulation')
    if not os.path.isdir(simul_dir):
        os.mkdir(simul_dir)

    if inps.plot_mat:
        plot_simulated(inps)

    outname = 'rmsemat_'+inps.signal_type
    if inps.seasonality:
        outname = outname+'_seasonal.npy'

    inps.outname = os.path.join(simul_dir,outname)


    if inps.signal_type=='linear':
        temp_baseline, displacement = simulate_constant_vel_phase(inps.n_img, inps.tmp_bl)
    else:
        temp_baseline, displacement = simulate_volcano_def_phase(inps.n_img, inps.tmp_bl)

    inps.ph0 = np.matrix((displacement * 4 * np.pi * inps.deformation_rate / (inps.lamda)).reshape(len(displacement), 1))


    inps.coh_sim_S = simulate_coherence_matrix_exponential(temp_baseline, 0.8, 0, inps.decorr_days,
                                                          inps.ph0, seasonal=inps.seasonality)
    inps.coh_sim_L = simulate_coherence_matrix_exponential(temp_baseline, 0.8, 0.2, inps.decorr_days,
                                                          inps.ph0, seasonal=inps.seasonality)

    repeat_simulation(numr=inps.n_sim, n_img=inps.n_img, n_shp=inps.n_shp,
                      phas=inps.ph0, coh_sim_S=inps.coh_sim_S, coh_sim_L=inps.coh_sim_L, outname=inps.outname)

    return
##################3

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
    eigen_value, eigen_vector = LA.eigh(corr_matrix)
    msk = (eigen_value < 1e-3)
    eigen_value[msk] = 0.
    # corr_matrix =  np.dot(eigen_vector, np.dot(np.diag(eigen_value), np.matrix.getH(eigen_vector)))

    # C = np.linalg.cholesky(corr_matrix)
    CM = np.dot(eigen_vector, np.dot(np.diag(np.sqrt(eigen_value)), np.matrix.getH(eigen_vector)))
    Zr = (np.random.randn(nsar) + 1j*np.random.randn(nsar)) / np.sqrt(2)
    noise = np.dot(CM, Zr)

    return noise

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
            ph0 = ph[ii] - ph[jj]
            C[ii, jj] = gamma * np.exp(1j * ph0)
            C[jj, ii] = np.conj(C[ii, jj])

    return C



def EST_rms(x):
    """ Estimate Root mean square error."""

    out = np.sqrt(np.sum(x ** 2, axis=1) / (np.shape(x)[1] - 1))

    return out


def custom_cmap(vmin=0, vmax=1):
    """ create a custom colormap based on visible portion of electromagnetive wave."""

    from minopy.spectrumRGB import rgb
    rgb = rgb()
    import matplotlib as mpl
    cmap = mpl.colors.ListedColormap(rgb)
    norm = mpl.colors.Normalize(vmin, vmax)

    return cmap, norm

def sequential_phase_linking(CCG, method, numsq=1):

    n_image = CCG.shape[0]
    num_seq = np.int(np.floor(n_image / 10))
    ph_ref = np.zeros([np.shape(CCG)[0],1])

    datumshift = np.zeros([num_seq, 1])
    Laq = -1

    for stepp in range(0, num_seq):

        first_line = stepp  * 10
        if stepp == num_seq-1:
            last_line = n_image
        else:
            last_line = first_line + 10
        num_lines = last_line - first_line

        if stepp == 0:

            ccg_sample = CCG[first_line:last_line, :]
            res, La, squeezed_pixels = phase_linking_process_py(ccg_sample, 0, method, squeez=True)
            ph_ref[first_line:last_line, 0:1] = res[stepp::].reshape(num_lines, 1)

        else:

            if numsq==1:
                ccg_sample = np.zeros([1 + num_lines, CCG.shape[1]]) + 1j
                ccg_sample[0:1, :] = np.complex64(squeezed_pixels[-1, :])
                ccg_sample[1::, :] = CCG[first_line:last_line, :]
                res, La, squeezed_p = phase_linking_process_py(ccg_sample, 1, method, squeez=True)
                ph_ref[first_line:last_line, 0:1] = res[1::].reshape(num_lines, 1)
                squeezed_pixels = np.complex64(np.vstack([squeezed_pixels, squeezed_p]))
            else:
                ccg_sample = np.zeros([stepp + num_lines, CCG.shape[1]]) + 1j
                ccg_sample[0:stepp, :] = np.complex64(squeezed_pixels[0:stepp, :])
                ccg_sample[stepp::, :] = CCG[first_line:last_line, :]
                res, La, squeezed_p = phase_linking_process_py(ccg_sample, stepp, method, squeez=True)
                ph_ref[first_line:last_line, 0:1] = res[stepp::].reshape(num_lines, 1)
                squeezed_pixels = np.complex64(np.vstack([squeezed_pixels, squeezed_p]))
        Laq = np.max([La, Laq])
    res_d, Lad = phase_linking_process_py(squeezed_pixels, 0, 'EMI', squeez=False)

    for stepp in range(0, len(res_d)):
        first_line = stepp * 10
        if stepp == num_seq - 1:
            last_line = n_image
        else:
            last_line = first_line + 10
        num_lines = last_line - first_line

        ph_ref[first_line:last_line, 0:1] = (ph_ref[first_line:last_line, 0:1] +
                                             np.matrix(res_d[int(stepp)]) - datumshift[
                                                 int(stepp)]).reshape(num_lines, 1)

    return  ph_ref

#######################################################################
############## Make a plot of coherence matrix and save: ##############

def plot_simulated(inps):

    plt.switch_backend('Agg')

    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=[12, 10], sharey=True)


    temp_baseline_v, displacement_v = simulate_volcano_def_phase(inps.n_img,inps.tmp_bl)   #(displacement in cm)
    #temp_baseline, displacement = simulate_constant_vel_phase(n_img,tmp_bl)
    ph_v = np.matrix((displacement_v*4*np.pi*inps.deformation_rate/(inps.lamda)).reshape(len(displacement_v),1))

    seasonality = inps.seasonality

    coh_sim_S_v = simulate_coherence_matrix_exponential(temp_baseline_v, 0.8,0, inps.decorr_days, ph_v, seasonal=inps.seasonality)
    coh_sim_L_v = simulate_coherence_matrix_exponential(temp_baseline_v, 0.8,0.2, inps.decorr_days, ph_v, seasonal=inps.seasonality)

    Ip_v = np.angle(coh_sim_L_v)

    CCGsam_Sterm_v = simulate_neighborhood_stack(coh_sim_S_v, neighborSamples=inps.n_shp)
    CCGsam_Lterm_v = simulate_neighborhood_stack(coh_sim_L_v, neighborSamples=inps.n_shp)

    coh_est_S_v = np.array(est_corr_py(CCGsam_Sterm_v))
    coh_est_L_v = np.array(est_corr_py(CCGsam_Lterm_v))

    Ip_S_v = np.angle(coh_est_S_v)
    Ip_L_v = np.angle(coh_est_L_v)



    cmap, norm = custom_cmap()
    im1=axs[0,0].imshow(np.abs(coh_sim_S_v), cmap='jet', norm=norm)
    im2=axs[0,1].imshow(np.abs(coh_sim_L_v), cmap='jet', norm=norm)
    im3=axs[0,2].imshow(np.abs(coh_est_S_v), cmap='jet', norm=norm)
    im4=axs[0,3].imshow(np.abs(coh_est_L_v), cmap='jet', norm=norm)

    cmap, norm = custom_cmap(-np.pi,np.pi)
    im5=axs[1,0].imshow(Ip_v, cmap='jet', norm=norm)
    im6=axs[1,1].imshow(Ip_v, cmap='jet', norm=norm)
    im7=axs[1,2].imshow(Ip_S_v, cmap='jet', norm=norm)
    im8=axs[1,3].imshow(Ip_L_v, cmap='jet', norm=norm)


    cax = fig.add_axes([0.336, 0.68, 0.08, 0.02])
    cbar = plt.colorbar(im2, cax=cax, ticks=[0,1], orientation='horizontal')
    cbar.set_label('Coherence', fontsize=12,color = "black")
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_tick_params(color='black')
    cbar.outline.set_edgecolor('black')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='black')

    cax = fig.add_axes([0.336, 0.425, 0.1, 0.02])
    cbar = plt.colorbar(im6, cax=cax, ticks=[-np.pi,0, np.pi], orientation='horizontal')
    cbar.set_label('Phase [rad]', fontsize=12,color = "black")
    cbar.ax.set_xticklabels([r'-$\pi$','0', r'$\pi$'], fontsize=12)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_tick_params(color='black')
    cbar.outline.set_edgecolor('black')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='black')

    cax = fig.add_axes([0.736, 0.68, 0.1, 0.02])
    cbar = plt.colorbar(im4, cax=cax, ticks=[0,1], orientation='horizontal')
    cbar.set_label('Coherence', fontsize=12,color = "black")
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_tick_params(color='black')
    cbar.outline.set_edgecolor('black')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='black')


    cax = fig.add_axes([0.736, 0.425, 0.1, 0.02])
    cbar = plt.colorbar(im8, cax=cax, ticks=[-np.pi,0, np.pi], orientation='horizontal')
    cbar.set_label('Phase [rad]', fontsize=12,color = "black")
    cbar.ax.set_xticklabels([r'-$\pi$','0', r'$\pi$'], fontsize=12)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_tick_params(color='black')
    cbar.outline.set_edgecolor('black')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='black')


    temp_baseline, displacement = simulate_constant_vel_phase(inps.n_img,inps.tmp_bl)
    ph0 = np.matrix((displacement*4*np.pi*inps.deformation_rate/(inps.lamda)).reshape(len(displacement),1))

    coh_sim_S = simulate_coherence_matrix_exponential(temp_baseline, 0.8,0, inps.decorr_days, ph0, seasonal=True)
    coh_sim_L = simulate_coherence_matrix_exponential(temp_baseline, 0.8,0.2, inps.decorr_days, ph0, seasonal=True)

    Ip = np.angle(coh_sim_L)

    CCGsam_Sterm = simulate_neighborhood_stack(coh_sim_S, neighborSamples=inps.n_shp)
    CCGsam_Lterm = simulate_neighborhood_stack(coh_sim_L, neighborSamples=inps.n_shp)

    coh_est_S = np.array(est_corr_py(CCGsam_Sterm))
    coh_est_L = np.array(est_corr_py(CCGsam_Lterm))

    Ip_S = np.angle(coh_est_S)
    Ip_L = np.angle(coh_est_L)


    cmap, norm = custom_cmap(np.min(Ip),np.max(Ip))
    im9=axs[2,0].imshow(Ip, cmap='jet', norm=norm)
    im10=axs[2,1].imshow(Ip, cmap='jet', norm=norm)
    cmap, norm = custom_cmap(np.min(Ip_L),np.max(Ip_L))
    im11=axs[2,2].imshow(Ip_S, cmap='jet', norm=norm)
    im12=axs[2,3].imshow(Ip_L, cmap='jet', norm=norm)


    fig.subplots_adjust(hspace=0.12, wspace=0.13)
    # axis format
    axs[0,0].set_ylabel('Image Number', fontsize=10)
    axs[1,0].set_ylabel('Image Number', fontsize=10)
    axs[2,0].set_ylabel('Image Number', fontsize=10)

    # colorbars

    cax = fig.add_axes([0.348, 0.165, 0.1, 0.02])
    cbar = plt.colorbar(im10, cax=cax, ticks=[np.min(Ip),0, np.max(Ip)], orientation='horizontal')
    cbar.set_label('Phase [rad]', fontsize=12,color = "black")
    cbar.ax.set_xticklabels([str(np.round(np.min(Ip),2)),'0', str(np.round(np.max(Ip),2))], fontsize=12)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_tick_params(color='black')
    cbar.outline.set_edgecolor('black')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='black')



    cax = fig.add_axes([0.748, 0.165, 0.1, 0.02])
    cbar = plt.colorbar(im12, cax=cax, ticks=[np.min(Ip_L),0, np.max(Ip_L)], orientation='horizontal')
    cbar.set_label('Phase [rad]', fontsize=12,color = "black")
    cbar.ax.set_xticklabels([str(np.round(np.min(Ip_L),2)),'0', str(np.round(np.max(Ip_L),2))], fontsize=12)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_tick_params(color='black')
    cbar.outline.set_edgecolor('black')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='black')

    outfile = os.path.join(os.getenv('SCRATCHDIR'),'simulation')

    if seasonality:
        plt.savefig(os.path.join(outfile,'Coherence_mat_seasonal.png'), bbox_inches='tight', transparent=True)
    else:
        plt.savefig(os.path.join(outfile,'Coherence_mat.png'), bbox_inches='tight', transparent=True)

    return None


def double_solve(f1,f2,x0,y0):
    """Solve for two equation with two unknowns using iterations"""

    from scipy.optimize import fsolve
    func = lambda x: [f1(x[0], x[1]), f2(x[0], x[1])]
    return fsolve(func, [x0, y0])


###################################################

def repeat_simulation(numr, n_img, n_shp, phas, coh_sim_S, coh_sim_L, outname):

    Timesmat = np.zeros([12, numr])

    EVD_est_resS = np.zeros([n_img, numr])
    EVD_est_resL = np.zeros([n_img, numr])
    EMI_est_resS = np.zeros([n_img, numr])
    EMI_est_resL = np.zeros([n_img, numr])
    PTA_est_resS = np.zeros([n_img, numr])
    PTA_est_resL = np.zeros([n_img, numr])

    EVD_seq_est_resS = np.zeros([n_img, numr])
    EVD_seq_est_resL = np.zeros([n_img, numr])
    EMI_seq_est_resS = np.zeros([n_img, numr])
    EMI_seq_est_resL = np.zeros([n_img, numr])
    PTA_seq_est_resS = np.zeros([n_img, numr])
    PTA_seq_est_resL = np.zeros([n_img, numr])

    for t in range(numr):
        CCGsam_Sterm = simulate_neighborhood_stack(coh_sim_S, neighborSamples=n_shp)
        CCGsam_Lterm = simulate_neighborhood_stack(coh_sim_L, neighborSamples=n_shp)

        time0 = time.time()
        ####
        #ph_EVD,la = phase_linking_process(CCGsam_Sterm, 0, 'EVD', squeez=False)
        ph_EVD, noval, temp_quality = phase_linking_process_py(CCGsam_Sterm, 0, b'EVD', False)
        EVD_est_resS[:, t:t + 1] = np.angle(np.array(ph_EVD)).reshape(-1, 1) - phas
        time1 = time.time()
        #ph_EVD,la = phase_linking_process(CCGsam_Lterm, 0, 'EVD', squeez=False)
        ph_EVD, noval, temp_quality = phase_linking_process_py(CCGsam_Lterm, 0, b'EVD', False)
        EVD_est_resL[:, t:t + 1] = np.angle(np.array(ph_EVD)).reshape(-1, 1) - phas
        time2 = time.time()

        ####
        #ph_EMI,la = phase_linking_process(CCGsam_Sterm, 0, 'EMI', squeez=False)
        ph_EMI, noval, temp_quality = phase_linking_process_py(CCGsam_Sterm, 0, b'EMI', False)
        EMI_est_resS[:, t:t + 1] = np.angle(np.array(ph_EMI)).reshape(-1, 1) - phas
        time3 = time.time()
        #ph_EMI,la = phase_linking_process(CCGsam_Lterm, 0, 'EMI', squeez=False)
        ph_EMI, noval, temp_quality = phase_linking_process_py(CCGsam_Lterm, 0, b'EMI', False)
        EMI_est_resL[:, t:t + 1] = np.angle(np.array(ph_EMI)).reshape(-1, 1) - phas
        time4 = time.time()

        ####
        #ph_PTA, la = phase_linking_process(CCGsam_Sterm, 0, 'PTA', squeez=False)
        ph_PTA, noval, temp_quality = phase_linking_process_py(CCGsam_Sterm, 0, b'PTA', False)
        PTA_est_resS[:, t:t + 1] = np.angle(np.array(ph_PTA)).reshape(-1, 1) - phas
        time5 = time.time()
        #ph_PTA, la = phase_linking_process(CCGsam_Lterm, 0, 'PTA', squeez=False)
        ph_PTA, noval, temp_quality = phase_linking_process_py(CCGsam_Lterm, 0, b'PTA', False)
        PTA_est_resL[:, t:t + 1] = np.angle(np.array(ph_PTA)).reshape(-1, 1) - phas
        time6 = time.time()


        ####
        num_seq = np.int(n_img // 10)
        ph_vec, sqeezed, temp_quality = sequential_phase_linking_py(CCGsam_Sterm, b'EVD', 10, num_seq)
        EVD_seq_est_resS[:, t:t + 1] = np.angle(np.array(ph_vec)).reshape(-1, 1) - phas
        time7 = time.time()
        ph_vec, sqeezed, temp_quality = sequential_phase_linking_py(CCGsam_Lterm, b'EVD', 10, num_seq)
        EVD_seq_est_resL[:, t:t + 1] = np.angle(np.array(ph_vec)).reshape(-1, 1) - phas
        time8 = time.time()
        ph_vec, sqeezed, temp_quality = sequential_phase_linking_py(CCGsam_Sterm, b'EMI', 10, num_seq)
        EMI_seq_est_resS[:, t:t + 1] = np.angle(np.array(ph_vec)).reshape(-1, 1) - phas
        time9 = time.time()
        ph_vec, sqeezed, temp_quality = sequential_phase_linking_py(CCGsam_Lterm, b'EMI', 10, num_seq)
        EMI_seq_est_resL[:, t:t + 1] = np.angle(np.array(ph_vec)).reshape(-1, 1) - phas
        time10 = time.time()
        ph_vec, sqeezed, temp_quality = sequential_phase_linking_py(CCGsam_Sterm, b'PTA', 10, num_seq)
        PTA_seq_est_resS[:, t:t + 1] = np.angle(np.array(ph_vec)).reshape(-1, 1) - phas
        time11 = time.time()
        ph_vec, sqeezed, temp_quality = sequential_phase_linking_py(CCGsam_Lterm, b'PTA', 10, num_seq)
        PTA_seq_est_resL[:, t:t + 1] = np.angle(np.array(ph_vec)).reshape(-1, 1) - phas
        time12 = time.time()

    rmsemat_est = np.zeros([n_img, 12])

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

    Timesmat[0:1, t:t + 1] = time1 - time0  # EVD_S
    Timesmat[1:2, t:t + 1] = time2 - time1  # EVD_L
    Timesmat[2:3, t:t + 1] = time3 - time2  # EMI_S
    Timesmat[3:4, t:t + 1] = time4 - time3  # EMI_L
    Timesmat[4:5, t:t + 1] = time5 - time4  # PTA_S
    Timesmat[5:6, t:t + 1] = time6 - time5  # PTA_L
    Timesmat[6:7, t:t + 1] = time7 - time6  # EVD_S_seq
    Timesmat[7:8, t:t + 1] = time8 - time7  # EVD_L_seq
    Timesmat[8:9, t:t + 1] = time9 - time8  # EMI_S_seq
    Timesmat[9:10, t:t + 1] = time10 - time9  # EMI_L_seq
    Timesmat[10:11, t:t + 1] = time11 - time10  # PTA_S_seq
    Timesmat[11:12, t:t + 1] = time12 - time11  # PTA_L_seq

    rmsemat_est_time = np.mean(Timesmat, 1)

    np.save(outname, rmsemat_est)
    np.save(outname.split('.npy')[0]+'_time.npy', rmsemat_est_time)

    return

###################################################3
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



def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)

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
####################################


if __name__ == '__main__':
    '''
    Simulates the phase linking process

    '''
    main()
