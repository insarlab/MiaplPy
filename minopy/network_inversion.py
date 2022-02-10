#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################
import os
import sys
import time
import datetime
from minopy.objects.arg_parser import MinoPyParser
from mintpy.utils import readfile, utils as ut
from mintpy import ifgram_inversion
import generate_temporal_coherence

############################################################

def main(iargs=None):
    """
        Overwrite filtered SLC images in Isce merged/SLC directory.
    """

    Parser = MinoPyParser(iargs, script='invert_network')
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
    minopy_dir = os.path.dirname(inps.work_dir)

    if inps.template_file is None:
        inps.template_file = os.path.join(minopy_dir, 'smallbaselineApp.cfg')

    minopy_template_file = os.path.join(minopy_dir, 'minopyApp.cfg')
    inps.ifgramStackFile = os.path.join(inps.work_dir, 'inputs/ifgramStack.h5')

    template = readfile.read_template(minopy_template_file)

    if template['minopy.timeseries.tempCohType'] == 'auto':
        template['minopy.timeseries.tempCohType'] = 'full'

    atr = {}
    atr['minopy.timeseries.tempCohType'] = template['minopy.timeseries.tempCohType']
    ut.add_attribute(inps.ifgramStackFile, atr)

    # 1) invert ifgramStack for time-series
    stack_file = ut.check_loaded_dataset(inps.work_dir, print_msg=False)[1]
    iargs = [stack_file, '-t', inps.template_file, '--update'] #, '--calc-cov']
    print('\nifgram_inversion.py', ' '.join(iargs))
    ifgram_inversion.main(iargs)

    # 1) Replace temporal coherence with the one obtained from full stack inversion
    iargs = ['-d', inps.work_dir]
    if inps.shadow_mask:
        iargs = ['-d', inps.work_dir, '--shadow_mask']
    print('\ngenerate_temporal_coherence.py', ' '.join(iargs))
    generate_temporal_coherence.main(iargs)

    #m, s = divmod(time.time() - start_time, 60)
    #print('time used: {:02.0f} mins {:02.1f} secs.\n'.format(m, s))

    return


if __name__ == '__main__':
    main()
