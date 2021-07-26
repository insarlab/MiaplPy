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
import mintpy.workflow

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

    if inps.custom_mask in ['None', None]:
        h5_mask = shadow_mask
    else:
        h5_mask = os.path.join(minopy_dir, 'mask_unwrap.h5')
        args_shm = '{} -m {} -o {} --fill 0'.format(inps.custom_mask,
                                                    shadow_mask, h5_mask)
        mintpy.mask.main(args_shm.split())

    corr_file = os.path.join(minopy_dir, 'inverted/quality')
    mask_arg = ' {} -m {} --fill 0 -o {}'.format(corr_file,
                                                 h5_mask,
                                                 corr_file + '_msk')
    mintpy.mask.main(mask_arg.split())

    if not inps.output_mask is None:
        unwrap_mask = inps.output_mask
    else:
        unwrap_mask = os.path.join(minopy_dir, 'inverted/mask_unwrap')

    save_cmd = '{} save_roipac.py {} -o {}'.format(inps.text_cmd, h5_mask, unwrap_mask)
    save_cmd = save_cmd.lstrip()
    os.system(save_cmd)

    return


if __name__ == '__main__':
    main()

