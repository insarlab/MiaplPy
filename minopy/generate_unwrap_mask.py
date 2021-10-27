#! /usr/bin/env python3
############################################################
# Copyright(c) 2021 Sara Mirzaee                          #
############################################################

import os
import sys
import datetime
from minopy.objects.arg_parser import MinoPyParser
import h5py
import mintpy
from mintpy.utils import readfile, writefile
import mintpy.workflow
import numpy as np

def main(iargs=None):
    """
        Generate_mask for unwrap with snaphu.
    """

    Parser = MinoPyParser(iargs, script='generate_mask')
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

    minopy_dir = os.path.dirname(os.path.dirname(inps.geometry_stack))

    shadow_mask = os.path.join(minopy_dir, 'shadow_mask.h5')
    args_shm = '{} shadowMask -m 0.5 --revert -o {}'.format(inps.geometry_stack, shadow_mask)

    mintpy.generate_mask.main(args_shm.split())

    if not inps.custom_mask in ['None', None]:
        h5_mask = inps.custom_mask
    else:
        h5_mask = shadow_mask

    atr = readfile.read_attribute(h5_mask)

    atr['FILE_TYPE'] = '.msk'

    atr_in = {}
    for key, value in atr.items():
        atr_in[key] = str(value)

    # drop the following keys
    key_list = ['width', 'Width', 'samples', 'length', 'lines',
                'SUBSET_XMIN', 'SUBSET_XMAX', 'SUBSET_YMIN', 'SUBSET_YMAX',
                ]
    for key in key_list:
        if key in atr_in.keys():
            atr_in.pop(key)

    # drop all keys that are not all UPPER_CASE
    key_list = list(atr_in.keys())
    for key in key_list:
        if not key.isupper():
            atr_in.pop(key)

    atr_in['FILE_LENGTH'] = atr_in['LENGTH']

    if not inps.custom_mask in ['None', None]:
        with h5py.File(h5_mask, 'r') as f:
            mask = f['mask'][:, :]
    else:
        mask = np.ones((int(atr_in['LENGTH']), int(atr_in['WIDTH'])), dtype=np.int)

    if not inps.output_mask is None:
        unwrap_mask = inps.output_mask
    else:
        unwrap_mask = os.path.join(minopy_dir, 'inverted/mask_unwrap')

    writefile.write(mask, out_file=unwrap_mask, metadata=atr)

    h5_mask = shadow_mask
    if not inps.custom_mask in ['None', None]:
        if os.path.exists(inps.custom_mask):
            args_shm = '{} -m {} -o {} --fill 0'.format(shadow_mask,
                                                        inps.custom_mask, h5_mask)
            mintpy.mask.main(args_shm.split())

    corr_file = os.path.join(minopy_dir, 'inverted/quality_average')
    mask_arg = ' {} -m {} --fill 0 -o {}'.format(corr_file,
                                                 h5_mask,
                                                 corr_file + '_msk')

    mintpy.mask.main(mask_arg.split())

    corr_file = os.path.join(minopy_dir, 'inverted/quality_full')
    mask_arg = ' {} -m {} --fill 0 -o {}'.format(corr_file,
                                                 h5_mask,
                                                 corr_file + '_msk')

    mintpy.mask.main(mask_arg.split())
    plot_masks(minopy_dir)

    return


def plot_masks(minopy_dir):
    files = [os.path.join(minopy_dir, 'waterMask.h5'),
             os.path.join(minopy_dir, 'shadow_mask.h5'),
             os.path.join(minopy_dir, 'shp'),
             os.path.join(minopy_dir, 'maskPS.h5')]
    for item in files:
        if os.path.exists(item):
            plt_args = '{} --nodisplay --noverbose --save -o {}'.format(item, item.split('.')[0] + '.png')
            mintpy.view.main(plt_args.split())
    return


if __name__ == '__main__':
    main()



