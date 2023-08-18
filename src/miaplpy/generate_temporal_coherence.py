#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################
import os
import sys
import time
import datetime
import numpy as np
from miaplpy.objects.arg_parser import MiaplPyParser
from mintpy.objects import ifgramStack, cluster
from mintpy.utils import writefile, readfile, utils as ut
import h5py
from mintpy.cli import generate_mask

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

    if 'miaplpy.timeseries.minTempCoh' in template:
        tcoh_min = float(template['miaplpy.timeseries.minTempCoh'])
    else:
        tcoh_min = 0.5

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
        atr['miaplpy.timeseries.minTempCoh'] = tcoh_min
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

    Parser = MiaplPyParser(iargs, script='generate_temporal_coherence')
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

    miaplpy_dir = os.path.dirname(inps.work_dir)
    miaplpy_template_file = os.path.join(miaplpy_dir, 'miaplpyApp.cfg')
    inps.ifgramStackFile = os.path.join(inps.work_dir, 'inputs/ifgramStack.h5')

    template = readfile.read_template(miaplpy_template_file)
    if template['miaplpy.timeseries.tempCohType'] == 'auto':
        template['miaplpy.timeseries.tempCohType'] = 'full'

    atr = {}
    atr['miaplpy.timeseries.tempCohType'] = template['miaplpy.timeseries.tempCohType']
    ut.add_attribute(inps.ifgramStackFile, atr)

    # check if input observation dataset exists.
    stack_obj = ifgramStack(inps.ifgramStackFile)
    stack_obj.open(print_msg=False)
    metadata = stack_obj.get_metadata()
    length, width = stack_obj.length, stack_obj.width

    inps.invQualityFile = 'temporalCoherence.h5'
    #mintpy_mask_file = os.path.join(inps.work_dir, 'maskTempCoh.h5')
    os.system('cp {} {}'.format(os.path.join(inps.work_dir, 'temporalCoherence.h5'),
                                os.path.join(inps.work_dir, 'temporalCoherence_mintpy.h5')))

    quality_name = os.path.join(miaplpy_dir,
                                'inverted/tempCoh_{}'.format(template['miaplpy.timeseries.tempCohType']))
    quality = np.memmap(quality_name, mode='r', dtype='float32', shape=(length, width))

    # inps.waterMaskFile = os.path.join(miaplpy_dir, 'waterMask.h5')
    inps.waterMaskFile = None
    water_mask = np.ones(quality.shape, dtype=np.int8)

    if template['miaplpy.timeseries.waterMask'] != 'auto':
        inps.waterMaskFile = template['miaplpy.timeseries.waterMask']
        if os.path.exists(inps.waterMaskFile):
            with h5py.File(inps.waterMaskFile, 'r') as f2:
                if 'waterMask' in f2:
                    water_mask = f2['waterMask'][:, :]
                else:
                    water_mask = f2['mask'][:, :]

    if inps.shadow_mask:
        if os.path.exists(os.path.join(miaplpy_dir, 'shadow_mask.h5')):
            with h5py.File(os.path.join(miaplpy_dir, 'shadow_mask.h5'), 'r') as f2:
                shadow_mask = f2['mask'][:, :]
                water_mask = water_mask * shadow_mask
    
    inv_quality = np.zeros((quality.shape[0], quality.shape[1]))
    inv_quality_name = 'temporalCoherence'
    inv_quality[:, :] = quality[:, :]
    inv_quality[inv_quality <= 0] = np.nan
    inv_quality[water_mask < 0.5] = np.nan
    #if os.path.exists(mintpy_mask_file):
        #mintpy_mask = readfile.read(mintpy_mask_file, datasetName='mask')[0]
        #inv_quality[mintpy_mask == 0] = np.nan

    if not os.path.exists(inps.invQualityFile):
        metadata['UNIT'] = '1'
        metadata['FILE_TYPE'] = inv_quality_name
        if 'REF_DATE' in metadata:
            metadata.pop('REF_DATE')
        ds_name_dict = {metadata['FILE_TYPE']: [np.float32, (length, width)]}
        writefile.layout_hdf5(inps.invQualityFile, ds_name_dict, metadata=metadata)

    # write the block to disk
    # with 3D block in [z0, z1, y0, y1, x0, x1]
    # and  2D block in         [y0, y1, x0, x1]
    block = [0, length, 0, width]
    writefile.write_hdf5_block(inps.invQualityFile,
                               data=inv_quality,
                               datasetName=inv_quality_name,
                               block=block)

    get_phase_linking_coherence_mask(metadata, inps.work_dir)

    m, s = divmod(time.time() - start_time, 60)
    print('time used: {:02.0f} mins {:02.1f} secs.\n'.format(m, s))

    return


if __name__ == '__main__':
    main()
