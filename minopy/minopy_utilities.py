#! /usr/bin/env python3
###############################################################################
# Project: Utilitiels for minopy
# Author: Sara Mirzaee
# Created: 10/2018
###############################################################################

import os
import numpy as np
import datetime
import time
from scipy import linalg as LA
from osgeo import gdal
import isce
import isceobj
import shutil
import h5py
import minopy
from minopy.objects.arg_parser import MinoPyParser

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
    eigen_value, eigen_vector = LA.eigh(corr_matrix)
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


def custom_cmap(vmin=0, vmax=1):
    """ create a custom colormap based on visible portion of electromagnetive wave."""

    from minopy.spectrumRGB import rgb
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

    print(gam_pta_c(np.angle(coh_mat), res))

    if squeez:

        vm = np.exp(1j * np.angle(np.matrix(res[stepp::, :])))
        vm = np.matrix(vm / LA.norm(vm))
        squeezed = np.matmul(vm.getH(), ccg_sample[stepp::, :])

        return res, squeezed
    else:
        return res


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

################################################################


def get_latest_template_minopy(work_dir):
    from minopy.objects.read_template import Template

    """Get the latest version of default template file.
    If an obsolete file exists in the working directory, the existing option values are kept.
    """
    lfile = os.path.join(os.path.dirname(__file__), 'defaults/minopyApp.cfg')  # latest version
    cfile = os.path.join(work_dir, 'minopyApp.cfg')  # current version
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
    from mroipac.looks.Looks import Looks

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

    return outfilie


def read_initial_info(work_dir, templateFile):
    from minopy.objects.slcStack import slcStack

    slc_file = os.path.join(work_dir, 'inputs/slcStack.h5')

    if os.path.exists(slc_file):
        slcObj = slcStack(slc_file)
        slcObj.open(print_msg=False)
        date_list = slcObj.get_date_list()
        metadata = slcObj.get_metadata()
        num_pixels = int(metadata['LENGTH']) * int(metadata['WIDTH'])
    else:
        scp_args = '--template {}'.format(templateFile)
        scp_args += ' --project_dir {}'.format(os.path.dirname(work_dir))

        Parser_LoadSlc = MinoPyParser(scp_args.split(), script='load_slc')
        inps_loadSlc = Parser_LoadSlc.parse()

        iDict = minopy.load_slc.read_inps2dict(inps_loadSlc)
        minopy.load_slc.prepare_metadata(iDict)
        metadata = minopy.load_slc.read_subset_box(iDict)
        box = metadata['box']
        num_pixels = (box[2] - box[0]) * (box[3] - box[1])
        stackObj = minopy.load_slc.read_inps_dict2slc_stack_dict_object(iDict)
        date_list = stackObj.get_date_list()

    return date_list, num_pixels, metadata

##########################################

