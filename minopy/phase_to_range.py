#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################
import os
import sys
import time
import datetime
import numpy as np
from minopy.objects.arg_parser import MinoPyParser
from mintpy.objects import ifgramStack, cluster
from mintpy.utils import writefile, readfile, utils as ut
from mintpy.ifgram_inversion import split2boxes, ifgram_inversion_patch
import h5py
from mintpy import generate_mask

############################################################


def get_phase_linking_coherence_mask(template, work_dir):
    """
    Generate reliable pixel mask from temporal coherence
    functions = [generate_mask, readfile, run_or_skip, add_attribute]
    # from mintpy import generate_mask
    # from mintpy.utils import readfile
    # from mintpy.utils.utils import run_or_skip, add_attribute
    """

    tcoh_file = os.path.join(work_dir, 'temporalCoherence.h5')

    mask_file = os.path.join(work_dir, 'maskTempCoh.h5')

    if 'minopy.timeseries.waterMask' in template:
        water_mask_file = template['minopy.timeseries.waterMask']
        if os.path.exists(water_mask_file):
            f1 = h5py.File(tcoh_file, 'a')
            f2 = h5py.File(water_mask_file, 'r')
            if 'waterMask' in f2:
                water_mask = f2['waterMask']
            else:
                water_mask = f2['mask']
            f1['temporalCoherence'][:, :] = np.multiply(f1['temporalCoherence'], water_mask)
            f1.close()
            f2.close()

    tcoh_min = float(template['minopy.timeseries.minTempCoh'])

    scp_args = '{} -m {} --nonzero -o {} --update'.format(tcoh_file, tcoh_min, mask_file)
    print('generate_mask.py', scp_args)

    # update mode: run only if:
    # 1) output file exists and newer than input file, AND
    # 2) all config keys are the same

    print('update mode: ON')
    flag = 'skip'
    if ut.run_or_skip(out_file=mask_file, in_file=tcoh_file, print_msg=False) == 'run':
        flag = 'run'

    print('run or skip: {}'.format(flag))

    if flag == 'run':
        generate_mask.main(scp_args.split())
        # update configKeys
        atr = {}
        atr['minopy.timeseries.minTempCoh'] = tcoh_min
        ut.add_attribute(mask_file, atr)
        ut.add_attribute(mask_file, atr)

    # check number of pixels selected in mask file for following analysis
    #num_pixel = np.sum(readfile.read(mask_file)[0] != 0.)
    #print('number of reliable pixels: {}'.format(num_pixel))

    #min_num_pixel = float(template['mintpy.networkInversion.minNumPixel'])   # 100
    #if num_pixel < min_num_pixel:
    #    msg = "Not enough reliable pixels (minimum of {}). ".format(int(min_num_pixel))
    #    msg += "Try the following:\n"
    #    msg += "1) Check the reference pixel and make sure it's not in areas with unwrapping errors\n"
    #    msg += "2) Check the network and make sure it's fully connected without subsets"
    #    raise RuntimeError(msg)

    return


def main(iargs=None):
    """
        Overwrite filtered SLC images in Isce merged/SLC directory.
    """

    Parser = MinoPyParser(iargs, script='phase_to_range')
    inps = Parser.parse()

    dateStr = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d:%H%M%S')

    if not iargs is None:
        msg = os.path.basename(__file__) + ' ' + ' '.join(iargs[:])
        string = dateStr + " * " + msg
        print(string)
    else:
        msg = os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1::])
        string = dateStr + " * " + msg
        print(string)

    start_time = time.time()
    os.chdir(inps.work_dir)

    ## 0. set MiNoPy defaults:

    key_prefix = 'mintpy.networkInversion.'
    configKeys = ['obsDatasetName',
                  'numIfgram',
                  'weightFunc',
                  'maskDataset',
                  'maskThreshold',
                  'minRedundancy',
                  'minNormVelocity']

    inps.ifgramStackFile = os.path.join(inps.work_dir, 'inputs/ifgramStack.h5')
    inps.skip_ref = False
    inps.minNormVelocity = False
    inps.minRedundancy = 1
    inps.maskDataset = 'coherence'
    inps.maskThreshold = 0.0
    inps.weightFunc = 'no'
    inps.outfile = ['timeseries.h5', 'temporalCoherence.h5', 'numInvIfgram.h5']
    inps.tsFile, inps.invQualityFile, inps.numInvFile = inps.outfile
    inps.obsDatasetName = 'unwrapPhase'
    # determine suffix based on unwrapping error correction method
    obs_suffix_map = {'bridging': '_bridging',
                      'phase_closure': '_phaseClosure',
                      'bridging+phase_closure': '_bridging_phaseClosure'}
    key = 'mintpy.unwrapError.method'

    # check if input observation dataset exists.
    stack_obj = ifgramStack(inps.ifgramStackFile)
    stack_obj.open(print_msg=False)
    metadata = stack_obj.get_metadata()

    if key in metadata.keys() and metadata[key]:
        unw_err_method = metadata[key].lower().replace(' ', '')  # fix potential typo
        inps.obsDatasetName += obs_suffix_map[unw_err_method]
        print('phase unwrapping error correction "{}" is turned ON'.format(unw_err_method))
    print('use dataset "{}" by default'.format(inps.obsDatasetName))

    if inps.obsDatasetName not in stack_obj.datasetNames:
        msg = 'input dataset name "{}" not found in file: {}'.format(inps.obsDatasetName, inps.ifgramStackFile)
        raise ValueError(msg)

    ## 1. input info

    date12_list = stack_obj.get_date12_list(dropIfgram=True)
    date_list = stack_obj.get_date_list(dropIfgram=True)
    length, width = stack_obj.length, stack_obj.width

    # 1.0 read minopy quality
    quality_name = os.path.join(inps.work_dir, 'inverted/quality')
    quality = np.memmap(quality_name, mode='r', dtype='float32', shape=(length, width))

    with h5py.File(os.path.join(inps.work_dir, 'avgSpatialCoh.h5'), 'r') as dsa:
        avgSpCoh = dsa['coherence'][:, :]

    # inps.waterMaskFile = os.path.join(inps.work_dir, 'waterMask.h5')
    inps.waterMaskFile = None
    water_mask = quality * 0 + 1
    if 'minopy.timeseries.waterMask' in metadata:
        inps.waterMaskFile = metadata['minopy.timeseries.waterMask']
        if os.path.exists(inps.waterMaskFile):
            with h5py.File(inps.waterMaskFile, 'r') as f2:
                if 'waterMask' in f2:
                    water_mask = f2['waterMask'][:, :]
                else:
                    water_mask = f2['mask'][:, :]

    if os.path.exists(os.path.join(inps.work_dir, 'shadow_mask.h5')):
        with h5py.File(os.path.join(inps.work_dir, 'shadow_mask.h5'), 'r') as f2:
            shadow_mask = f2['mask'][:, :]
            water_mask = water_mask * shadow_mask

    # 1.1 read values on the reference pixel
    inps.refPhase = stack_obj.get_reference_phase(unwDatasetName=inps.obsDatasetName,
                                                  skip_reference=inps.skip_ref,
                                                  dropIfgram=True)

    # 1.2 design matrix
    A = stack_obj.get_design_matrix4timeseries(date12_list)[0]
    num_ifgram, num_date = A.shape[0], A.shape[1] + 1
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
    for key in configKeys:
        metadata[key_prefix + key] = str(vars(inps)[key])

    metadata['FILE_TYPE'] = 'timeseries'
    metadata['UNIT'] = 'm'
    metadata['REF_DATE'] = date_list[0]

    # 2.2 instantiate time-series
    dates = np.array(date_list, dtype=np.string_)
    pbase = stack_obj.get_perp_baseline_timeseries(dropIfgram=True)
    ds_name_dict = {
        "date": [dates.dtype, (num_date,), dates],
        "bperp": [np.float32, (num_date,), pbase],
        "timeseries": [np.float32, (num_date, length, width), None],
    }
    writefile.layout_hdf5(inps.tsFile, ds_name_dict, metadata)

    # 2.3 instantiate invQualifyFile: temporalCoherence / residualInv
    if 'residual' in os.path.basename(inps.invQualityFile).lower():
        inv_quality_name = 'residual'
        metadata['UNIT'] = 'pixel'
    else:
        inv_quality_name = 'temporalCoherence'
        metadata['UNIT'] = '1'
    metadata['FILE_TYPE'] = inv_quality_name
    metadata.pop('REF_DATE')
    ds_name_dict = {metadata['FILE_TYPE']: [np.float32, (length, width)]}
    writefile.layout_hdf5(inps.invQualityFile, ds_name_dict, metadata=metadata)

    # 2.4 instantiate number of inverted observations
    metadata['FILE_TYPE'] = 'mask'
    metadata['UNIT'] = '1'
    ds_name_dict = {"mask": [np.float32, (length, width)]}
    writefile.layout_hdf5(inps.numInvFile, ds_name_dict, metadata=metadata)

    ## 3. run the inversion / estimation and write to disk

    # 3.1 split ifgram_file into blocks to save memory
    box_list, num_box = split2boxes(inps.ifgramStackFile)

    # 3.2 prepare the input arguments for *_patch()
    data_kwargs = {
        "ifgram_file": inps.ifgramStackFile,
        "ref_phase": inps.refPhase,
        "obs_ds_name": inps.obsDatasetName,
        "weight_func": inps.weightFunc,
        "min_norm_velocity": inps.minNormVelocity,
        "water_mask_file": inps.waterMaskFile,
        "mask_ds_name": inps.maskDataset,
        "mask_threshold": inps.maskThreshold,
        "min_redundancy": inps.minRedundancy
    }

    # 3.3 invert / write block-by-block
    for i, box in enumerate(box_list):
        box_wid = box[2] - box[0]
        box_len = box[3] - box[1]
        if num_box > 1:
            print('\n------- processing patch {} out of {} --------------'.format(i + 1, num_box))
            print('box width:  {}'.format(box_wid))
            print('box length: {}'.format(box_len))

        # update box argument in the input data
        data_kwargs['box'] = box
        num_workers = int(metadata['minopy.compute.numWorker'])
        if num_workers == 1:
            # non-parallel
            ts, inv_quality, num_inv_ifg = ifgram_inversion_patch(**data_kwargs)[:-1]

        else:
            # parallel
            print('\n\n------- start parallel processing using Dask -------')

            # initiate the output data
            ts = np.zeros((num_date, box_len, box_wid), np.float32)
            inv_quality = np.zeros((box_len, box_wid), np.float32)
            num_inv_ifg = np.zeros((box_len, box_wid), np.float32)

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
        # temp_coh = quality[box[1]:box[3], box[0]:box[2]]
        inv_quality[:, :] = quality[box[1]:box[3], box[0]:box[2]]
        inv_quality[inv_quality <= 0] = np.nan
        water_mask_box = water_mask[box[1]:box[3], box[0]:box[2]]
        inv_quality[water_mask_box < 0.5] = np.nan
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
        print('-' * 50)
        print('update values on the reference pixel: ({}, {})'.format(ref_y, ref_x))

        print('set {} on the reference pixel to 1.'.format(inv_quality_name))
        with h5py.File(inps.invQualityFile, 'r+') as f:
            f['temporalCoherence'][ref_y, ref_x] = 1.

        print('set  # of observations on the reference pixel as {}'.format(num_ifgram))
        with h5py.File(inps.numInvFile, 'r+') as f:
            f['mask'][ref_y, ref_x] = num_ifgram

    get_phase_linking_coherence_mask(metadata, inps.work_dir)

    m, s = divmod(time.time() - start_time, 60)
    print('time used: {:02.0f} mins {:02.1f} secs.\n'.format(m, s))

    return


if __name__ == '__main__':
    main()
