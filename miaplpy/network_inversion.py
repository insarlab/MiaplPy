#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################
import os
import sys
import datetime
from mintpy.utils import readfile, utils as ut
from miaplpy.objects.arg_parser import MiaplPyParser
from miaplpy.dev import ifgram_inversion_L1L2
import generate_temporal_coherence

############################################################

def main(iargs=None):
    """
        Overwrite filtered SLC images in Isce merged/SLC directory.
    """

    Parser = MiaplPyParser(iargs, script='invert_network')
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

    os.chdir(inps.work_dir)
    miaplpy_dir = os.path.dirname(inps.work_dir)

    if inps.template_file is None:
        inps.template_file = os.path.join(miaplpy_dir, 'smallbaselineApp.cfg')

    miaplpy_template_file = os.path.join(miaplpy_dir, 'miaplpyApp.cfg')
    template = readfile.read_template(miaplpy_template_file)

    if template['miaplpy.timeseries.tempCohType'] == 'auto':
        template['miaplpy.timeseries.tempCohType'] = 'full'

    atr = {}
    atr['miaplpy.timeseries.tempCohType'] = template['miaplpy.timeseries.tempCohType']
    ut.add_attribute(inps.ifgramStackFile, atr)

    # 1) invert ifgramStack for time-series
    #wrapped_phase_series = os.path.join(miaplpy_dir, 'inverted/phase_series.h5')
    iargs = [inps.ifgramStackFile, '-t', inps.template_file, '--update', '--norm', inps.residualNorm,
             '--tcoh', inps.temp_coh, '--mask-threshold', str(inps.maskThreshold),
             '--smooth_factor', inps.L1_alpha]   #, '--calc-cov']

    if not inps.minNormVelocity:
        iargs += ['--min-norm-phase']

    print('\nifgram_inversion_L1L2.py', ' '.join(iargs))
    ifgram_inversion_L1L2.main(iargs)

    # 1) Replace temporal coherence with the one obtained from full stack inversion
    iargs = ['-d', inps.work_dir]
    if inps.shadow_mask:
        iargs = ['-d', inps.work_dir, '--shadow_mask']
    print('\ngenerate_temporal_coherence.py', ' '.join(iargs))
    generate_temporal_coherence.main(iargs)


    return


if __name__ == '__main__':
    main()
