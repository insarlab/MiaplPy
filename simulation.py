#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os,sys
import numpy as np
from matplotlib import pyplot as plt, ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy import linalg as LA
import _pysqsar_utilities as psq
import cmath
import pandas as pd

from pysar.utils import ptime, utils as ut, network as pnet, plot as pp

#displacement = 1mm/y = (1 mm *4pi / lambda(mm)) rad/y  --> 6 day = 4pi*6/lambda*365
#displacement = lambda*phi/ 4*pi.

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



#######################################################################
############## Make a plot of coherence matrix and save: ##############

def plot_simulated(inps):

    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=[12, 10], sharey=True)


    temp_baseline_v, displacement_v = psq.simulate_volcano_def_phase(inps.n_img,inps.tmp_bl)   #(displacement in cm)
    #temp_baseline, displacement = psq.simulate_constant_vel_phase(n_img,tmp_bl)
    ph_v = np.matrix((displacement_v*4*np.pi*inps.deformation_rate/(inps.lamda)).reshape(len(displacement_v),1))

    seasonality = inps.seasonality

    coh_sim_S_v = psq.simulate_coherence_matrix_exponential(temp_baseline_v, 0.8,0, inps.decorr_days, ph_v, seasonal=seasonality)
    coh_sim_L_v = psq.simulate_coherence_matrix_exponential(temp_baseline_v, 0.8,0.2, inps.decorr_days, ph_v, seasonal=seasonality)

    Ip_v = np.angle(coh_sim_L_v)

    CCGsam_Sterm_v = psq.simulate_neighborhood_stack(coh_sim_S_v, neighborSamples=inps.n_shp)
    CCGsam_Lterm_v = psq.simulate_neighborhood_stack(coh_sim_L_v, neighborSamples=inps.n_shp)

    coh_est_S_v = psq.est_corr(CCGsam_Sterm_v)
    coh_est_L_v = psq.est_corr(CCGsam_Lterm_v)

    Ip_S_v = np.angle(coh_est_S_v)
    Ip_L_v = np.angle(coh_est_L_v)



    cmap, norm = psq.custom_cmap()
    im1=axs[0,0].imshow(np.abs(coh_sim_S_v), cmap='jet', norm=norm)
    im2=axs[0,1].imshow(np.abs(coh_sim_L_v), cmap='jet', norm=norm)
    im3=axs[0,2].imshow(np.abs(coh_est_S_v), cmap='jet', norm=norm)
    im4=axs[0,3].imshow(np.abs(coh_est_L_v), cmap='jet', norm=norm)

    cmap, norm = psq.custom_cmap(-np.pi,np.pi)
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


    temp_baseline, displacement = psq.simulate_constant_vel_phase(n_img,tmp_bl)
    ph0 = np.matrix((displacement*4*np.pi*deformation_rate/(lamda)).reshape(len(displacement),1))

    coh_sim_S = psq.simulate_coherence_matrix_exponential(temp_baseline, 0.8,0, decorr_days, ph0, seasonal=True)
    coh_sim_L = psq.simulate_coherence_matrix_exponential(temp_baseline, 0.8,0.2, decorr_days, ph0, seasonal=True)

    Ip = np.angle(coh_sim_L)

    CCGsam_Sterm = psq.simulate_neighborhood_stack(coh_sim_S, neighborSamples=n_shp)
    CCGsam_Lterm = psq.simulate_neighborhood_stack(coh_sim_L, neighborSamples=n_shp)

    coh_est_S = psq.est_corr(CCGsam_Sterm)
    coh_est_L = psq.est_corr(CCGsam_Lterm)

    Ip_S = np.angle(coh_est_S)
    Ip_L = np.angle(coh_est_L)


    cmap, norm = psq.custom_cmap(np.min(Ip),np.max(Ip))
    im9=axs[2,0].imshow(Ip, cmap='jet', norm=norm)
    im10=axs[2,1].imshow(Ip, cmap='jet', norm=norm)
    cmap, norm = psq.custom_cmap(np.min(Ip_L),np.max(Ip_L))
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

    if seasonality:
        plt.savefig('Coherence_mat_seasonal.png', bbox_inches='tight', transparent=True)
    else:
        plt.savefig('Coherence_mat.png', bbox_inches='tight', transparent=True)

    return None

###################################################

def repeat_simulation(numr, n_img, n_shp, phas, coh_sim_S, coh_sim_L):


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
        CCGsam_Sterm = psq.simulate_neighborhood_stack(coh_sim_S, neighborSamples=n_shp)
        CCGsam_Lterm = psq.simulate_neighborhood_stack(coh_sim_L, neighborSamples=n_shp)

        coh_est_S = psq.est_corr(CCGsam_Sterm)
        coh_est_L = psq.est_corr(CCGsam_Lterm)

        ####
        ph_EVD = psq.EVD_phase_estimation(coh_est_S)
        EVD_est_resS[:, t:t + 1] = ph_EVD - phas
        ph_EVD = psq.EVD_phase_estimation(coh_est_L)
        EVD_est_resL[:, t:t + 1] = ph_EVD - phas

        ####
        ph_EMI = psq.EMI_phase_estimation(coh_est_S)
        EMI_est_resS[:, t:t + 1] = ph_EMI - phas
        xm = np.zeros([len(ph0), len(ph0) + 1]) + 0j
        xm[:, 0:1] = np.reshape(ph_EMI, [len(ph0), 1])
        xm[:, 1::] = coh_est_S[:, :]
        res_PTA = psq.PTA_L_BFGS(xm)
        PTA_est_resS[:, t:t + 1] = res_PTA - phas

        ####
        ph_EMI = psq.EMI_phase_estimation(coh_est_L)
        EMI_est_resL[:, t:t + 1] = ph_EMI - phas
        xm = np.zeros([len(ph0), len(ph0) + 1]) + 0j
        xm[:, 0:1] = np.reshape(ph_EMI, [len(ph0), 1])
        # xm[:,0:1] = np.reshape(np.angle(CCGsam_Lterm[:,0]),[len(ph0),1])
        xm[:, 1::] = coh_est_L[:, :]
        res_PTA = psq.PTA_L_BFGS(xm)
        PTA_est_resL[:, t:t + 1] = res_PTA - phas

        ####
        EVD_seq_est_resS[:, t:t + 1] = psq.sequential_phase_linking(CCGsam_Sterm, 'EVD') - phas
        EVD_seq_est_resL[:, t:t + 1] = psq.sequential_phase_linking(CCGsam_Lterm, 'EVD') - phas
        EMI_seq_est_resS[:, t:t + 1] = psq.sequential_phase_linking(CCGsam_Sterm, 'EMI') - phas
        EMI_seq_est_resL[:, t:t + 1] = psq.sequential_phase_linking(CCGsam_Lterm, 'EMI') - phas
        PTA_seq_est_resS[:, t:t + 1] = psq.sequential_phase_linking(CCGsam_Sterm, 'PTA') - phas
        PTA_seq_est_resL[:, t:t + 1] = psq.sequential_phase_linking(CCGsam_Lterm, 'PTA') - phas

    rmsemat_est = np.zeros([n_img, 12])

    rmsemat_est[:, 0] = psq.EST_rms(EVD_est_resS)
    rmsemat_est[:, 1] = psq.EST_rms(EVD_est_resL)
    rmsemat_est[:, 2] = psq.EST_rms(EMI_est_resS)
    rmsemat_est[:, 3] = psq.EST_rms(EMI_est_resL)
    rmsemat_est[:, 4] = psq.EST_rms(PTA_est_resS)
    rmsemat_est[:, 5] = psq.EST_rms(PTA_est_resL)

    rmsemat_est[:, 6] = psq.EST_rms(EVD_seq_est_resS)
    rmsemat_est[:, 7] = psq.EST_rms(EVD_seq_est_resL)
    rmsemat_est[:, 8] = psq.EST_rms(EMI_seq_est_resS)
    rmsemat_est[:, 9] = psq.EST_rms(EMI_seq_est_resL)
    rmsemat_est[:, 10] = psq.EST_rms(PTA_seq_est_resS)
    rmsemat_est[:, 11] = psq.EST_rms(PTA_seq_est_resL)

    np.save('rmsemat_linear_seasonal.npy', rmsemat_est)


    ####################################

    def main():

        inps = command_line_parse(iargs)


        inps = ()
        inps.lamda = 56.0  # wavelength (mm)
        inps.n_img = 100
        inps.n_shp = 300
        inps.decorr_days = 50
        inps.deformation_rate = 1  # mm/y
        inps.tmp_bl = 6  # days