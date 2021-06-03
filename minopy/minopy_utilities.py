#! /usr/bin/env python3
###############################################################################
# Project: Utilitiels for minopy
# Author: Sara Mirzaee
# Created: 10/2018
###############################################################################

import os
import numpy as np
import cmath
import datetime, time
from scipy import linalg as LA
from scipy.optimize import minimize
from scipy.stats import ks_2samp, anderson_ksamp, ttest_ind
from osgeo import gdal
import isce
import isceobj
from mroipac.looks.Looks import Looks
import glob
import shutil
import warnings
import h5py
from mintpy.objects import timeseries, ifgramStack, cluster
from mintpy.ifgram_inversion import split2boxes, read_unwrap_phase, mask_unwrap_phase, ifgram_inversion_patch
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

#################################
def invert_ifgrams_to_timeseries(template, inps_dict, work_dir, writefile, num_workers=1):
    
    inps = inps_dict
    start_time = time.time()
     
    ## 0. set MiNoPy defaults:
    
    key_prefix = 'mintpy.networkInversion.'
    configKeys = ['obsDatasetName',
              'numIfgram',
              'weightFunc',
              'maskDataset',
              'maskThreshold',
              'minRedundancy',
              'minNormVelocity']
 
    inps.waterMaskFile = os.path.join(work_dir, 'waterMask.h5')
    if not os.path.exists(inps.waterMaskFile):
       inps.waterMaskFile = None 
    
    inps.ifgramStackFile = os.path.join(work_dir, 'inputs/ifgramStack.h5')
    inps.skip_ref = True
    inps.minNormVelocity = False
    inps.minRedundancy = 1
    inps.maskDataset = 'coherence'
    inps.maskThreshold = 0.0
    inps.weightFunc = 'no'
    inps.outfile = ['timeseries.h5', 'temporalCoherence.h5', 'numInvIfgram.h5']
    inps.tsFile, inps.invQualityFile, inps.numInvFile = inps.outfile
    inps.obsDatasetName = 'unwrapPhase'
    # determine suffix based on unwrapping error correction method
    obs_suffix_map = {'bridging'               : '_bridging',
                      'phase_closure'          : '_phaseClosure',
                      'bridging+phase_closure' : '_bridging_phaseClosure'}
    key = 'mintpy.unwrapError.method'
    if key in template.keys() and template[key]:
        unw_err_method = template[key].lower().replace(' ','')   # fix potential typo
        inps.obsDatasetName += obs_suffix_map[unw_err_method]
        print('phase unwrapping error correction "{}" is turned ON'.format(unw_err_method))
    print('use dataset "{}" by default'.format(inps.obsDatasetName))
    
    # check if input observation dataset exists.
    stack_obj = ifgramStack(inps.ifgramStackFile)
    stack_obj.open(print_msg=False)
    if inps.obsDatasetName not in stack_obj.datasetNames:
        msg = 'input dataset name "{}" not found in file: {}'.format(inps.obsDatasetName, inps.ifgramStackFile)
        raise ValueError(msg)    
    
   
    ## 1. input info

    stack_obj = ifgramStack(inps.ifgramStackFile)
    stack_obj.open(print_msg=False)
    date12_list = stack_obj.get_date12_list(dropIfgram=True)
    date_list = stack_obj.get_date_list(dropIfgram=True)
    length, width = stack_obj.length, stack_obj.width
    
    # 1.0 read minopy quality
    quality_name = template['quality_file']
    quality = np.memmap(quality_name, mode='r', dtype='float32', shape=(length, width))

    with h5py.File(os.path.join(work_dir, 'avgSpatialCoh.h5'), 'r') as dsa:
        avgSpCoh = dsa['coherence'][:, :]


    # 1.1 read values on the reference pixel
    inps.refPhase = stack_obj.get_reference_phase(unwDatasetName=inps.obsDatasetName,
                                                  skip_reference=inps.skip_ref,
                                                  dropIfgram=True)

    # 1.2 design matrix
    A = stack_obj.get_design_matrix4timeseries(date12_list)[0]
    num_ifgram, num_date = A.shape[0], A.shape[1]+1
    inps.numIfgram = num_ifgram

    # 1.3 print key setup info
    msg = '-------------------------------------------------------------------------------\n'
    if inps.minNormVelocity:
        suffix = 'deformation velocity'
    else:
        suffix = 'deformation phase'
    msg += 'least-squares solution with L2 min-norm on: {}\n'.format(suffix)
    msg += 'minimum redundancy: {}\n'.format(inps.minRedundancy)
    msg += 'weight function: {}\n'.format(inps.weightFunc)

    if inps.maskDataset:
        if inps.maskDataset in ['coherence', 'offsetSNR']:
            suffix = '{} < {}'.format(inps.maskDataset, inps.maskThreshold)
        else:
            suffix = '{} == 0'.format(inps.maskDataset)
        msg += 'mask out pixels with: {}\n'.format(suffix)
    else:
        msg += 'mask: no\n'

    if np.linalg.matrix_rank(A) < A.shape[1]:
        msg += '***WARNING: the network is NOT fully connected.\n'
        msg += '\tInversion result can be biased!\n'
        msg += '\tContinue to use SVD to resolve the offset between different subsets.\n'
    msg += '-------------------------------------------------------------------------------'
    print(msg)

    print('number of interferograms: {}'.format(num_ifgram))
    print('number of acquisitions  : {}'.format(num_date))
    print('number of lines   : {}'.format(length))
    print('number of columns : {}'.format(width))

    ## 2. prepare output

    # 2.1 metadata
    meta = dict(stack_obj.metadata)
    for key in configKeys:
        meta[key_prefix+key] = str(vars(inps)[key])

    meta['FILE_TYPE'] = 'timeseries'
    meta['UNIT'] = 'm'
    meta['REF_DATE'] = date_list[0]

    # 2.2 instantiate time-series
    dates = np.array(date_list, dtype=np.string_)
    pbase = stack_obj.get_perp_baseline_timeseries(dropIfgram=True)
    ds_name_dict = {
        "date"       : [dates.dtype, (num_date,), dates],
        "bperp"      : [np.float32,  (num_date,), pbase],
        "timeseries" : [np.float32,  (num_date, length, width), None],
    }
    writefile.layout_hdf5(inps.tsFile, ds_name_dict, meta)

    # 2.3 instantiate invQualifyFile: temporalCoherence / residualInv
    if 'residual' in os.path.basename(inps.invQualityFile).lower():
        inv_quality_name = 'residual'
        meta['UNIT'] = 'pixel'
    else:
        inv_quality_name = 'temporalCoherence'
        meta['UNIT'] = '1'
    meta['FILE_TYPE'] = inv_quality_name
    meta.pop('REF_DATE')
    ds_name_dict = {meta['FILE_TYPE'] : [np.float32, (length, width)]}
    writefile.layout_hdf5(inps.invQualityFile, ds_name_dict, metadata=meta)

    # 2.4 instantiate number of inverted observations
    meta['FILE_TYPE'] = 'mask'
    meta['UNIT'] = '1'
    ds_name_dict = {"mask" : [np.float32, (length, width)]}
    writefile.layout_hdf5(inps.numInvFile, ds_name_dict, metadata=meta)

    ## 3. run the inversion / estimation and write to disk

    # 3.1 split ifgram_file into blocks to save memory
    box_list, num_box = split2boxes(inps.ifgramStackFile)

    # 3.2 prepare the input arguments for *_patch()
    data_kwargs = {
        "ifgram_file"       : inps.ifgramStackFile,
        "ref_phase"         : inps.refPhase,
        "obs_ds_name"       : inps.obsDatasetName,
        "weight_func"       : inps.weightFunc,
        "min_norm_velocity" : inps.minNormVelocity,
        "water_mask_file"   : inps.waterMaskFile,
        "mask_ds_name"      : inps.maskDataset,
        "mask_threshold"    : inps.maskThreshold,
        "min_redundancy"    : inps.minRedundancy
    }

    # 3.3 invert / write block-by-block
    for i, box in enumerate(box_list):
        box_wid = box[2] - box[0]
        box_len = box[3] - box[1]
        if num_box > 1:
            print('\n------- processing patch {} out of {} --------------'.format(i+1, num_box))
            print('box width:  {}'.format(box_wid))
            print('box length: {}'.format(box_len))

        # update box argument in the input data
        data_kwargs['box'] = box

        if num_workers == 1:
            # non-parallel
            ts, inv_quality, num_inv_ifg = ifgram_inversion_patch(**data_kwargs)[:-1]

        else:
            # parallel
            print('\n\n------- start parallel processing using Dask -------')

            # initiate the output data
            ts = np.zeros((num_date, box_len, box_wid), np.float32)
            inv_quality = np.zeros((box_len, box_wid), np.float32)
            num_inv_ifg  = np.zeros((box_len, box_wid), np.float32)

            # initiate dask cluster and client
            cluster_obj = cluster.DaskCluster('local', num_workers)
            cluster_obj.open()

            # run dask
            ts, inv_quality, num_inv_ifg = cluster_obj.run(func=ifgram_inversion_patch,
                                                           func_data=data_kwargs,
                                                           results=[ts, inv_quality, num_inv_ifg])

            # close dask cluster and client
            cluster_obj.close()

            print('------- finished parallel processing -------\n\n')

        # write the block to disk
        # with 3D block in [z0, z1, y0, y1, x0, x1]
        # and  2D block in         [y0, y1, x0, x1]
        # time-series - 3D
        block = [0, num_date, box[1], box[3], box[0], box[2]] 
        writefile.write_hdf5_block(inps.tsFile,
                                   data=ts,
                                   datasetName='timeseries',
                                   block=block)

        # temporal coherence - 2D
        block = [box[1], box[3], box[0], box[2]]
        #temp_coh = quality[box[1]:box[3], box[0]:box[2]]
        inv_quality[:, :] = quality[box[1]:box[3], box[0]:box[2]]
        inv_quality[inv_quality<=0] = np.nan
        writefile.write_hdf5_block(inps.invQualityFile,
                                   data=inv_quality,
                                   datasetName=inv_quality_name,
                                   block=block)

        # number of inverted obs - 2D
        num_inv_ifg * 0 + (len(date_list) - 1)
        writefile.write_hdf5_block(inps.numInvFile,
                                   data=num_inv_ifg,
                                   datasetName='mask',
                                   block=block)

        if num_box > 1:
            m, s = divmod(time.time() - start_time, 60)
            print('time used: {:02.0f} mins {:02.1f} secs.\n'.format(m, s))

    # 3.4 update output data on the reference pixel (for phase)
    if not inps.skip_ref:
        # grab ref_y/x
        ref_y = int(stack_obj.metadata['REF_Y'])
        ref_x = int(stack_obj.metadata['REF_X'])
        print('-'*50)
        print('update values on the reference pixel: ({}, {})'.format(ref_y, ref_x))

        print('set {} on the reference pixel to 1.'.format(inv_quality_name))
        with h5py.File(inps.invQualityFile, 'r+') as f:
            f['temporalCoherence'][ref_y, ref_x] = 1.

        print('set  # of observations on the reference pixel as {}'.format(num_ifgram))
        with h5py.File(inps.numInvFile, 'r+') as f:
            f['mask'][ref_y, ref_x] = num_ifgram

    m, s = divmod(time.time() - start_time, 60)
    print('time used: {:02.0f} mins {:02.1f} secs.\n'.format(m, s))
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

##########################################

