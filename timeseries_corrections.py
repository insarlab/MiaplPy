#!/usr/bin/env python3
# Author: Sara Mirzaee


import os
import time
import datetime
import shutil
import argparse
import numpy as np
import gdal
import mintpy
import mintpy.workflow  #dynamic import for modules used by smallbaselineApp workflow
from mintpy.utils import readfile, utils as ut
from mintpy.defaults.auto_path import autoPath
from mintpy.smallbaselineApp import TimeSeriesAnalysis
from mintpy.objects import ifgramStack
from mintpy.ifgram_inversion import write2hdf5_file, read_unwrap_phase, mask_unwrap_phase

##########################################################################
STEP_LIST = [
    'crop',
    'create_patch',
    'inversion',
    'load_data',
    'ifgrams',
    'unwrap'
    'load_data'
    'reference_point',
    'correct_unwrap_error',
    'stack_interferograms',
    'correct_troposphere',
    'deramp',
    'correct_topography',
    'residual_RMS',
    'reference_date',
    'velocity',
    'geocode',
    'google_earth',
    'hdfeos5',
]

STEP_HELP = """Command line options for steps processing with names are chosen from the following list:
                {}

                In order to use either --start or --step, it is necessary that a
                previous run was done using one of the steps options to process at least
                through the step immediately preceding the starting step of the current run.
                """.format(STEP_LIST[0:7])

EXAMPLE = """example: 
      minopy_wrapper.py  <custom_template_file>              # run with default and custom templates
      minopy_wrapper.py  <custom_template_file>  --submit    # submit as job
      minopy_wrapper.py  -h / --help                       # help 
      minopy_wrapper.py  -H                                # print    default template options
      # Run with --start/stop/step options
      minopy_wrapper.py GalapagosSenDT128.template --step  crop         # run the step 'download' only
      minopy_wrapper.py GalapagosSenDT128.template --start crop         # start from the step 'download' 
      minopy_wrapper.py GalapagosSenDT128.template --stop  unwrap       # end after step 'interferogram'
"""


def create_parser():
    parser = argparse.ArgumentParser(description='Routine Time Series Analysis for Small Baseline InSAR Stack',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)

    parser.add_argument('customTemplateFile', nargs='?',
                        help='custom template with option settings.\n' +
                             "ignored if the default smallbaselineApp.cfg is input.")
    parser.add_argument('--dir', dest='workDir',
                        help='specify custom working directory. The default is:\n' +
                             'a) current directory, OR\n' +
                             'b) $SCRATCHDIR/$projectName/mintpy, if:\n' +
                             '    1) autoPath == True in $MINTPY_HOME/mintpy/defaults/auto_path.py AND\n' +
                             '    2) environment variable $SCRATCHDIR exists AND\n' +
                             '    3) customTemplateFile is specified (projectName.*)\n')

    parser.add_argument('-g', dest='generate_template', action='store_true',
                        help='generate default template (if it does not exist) and exit.')
    parser.add_argument('-H', dest='print_template', action='store_true',
                        help='print the default template file and exit.')
    parser.add_argument('-v','--version', action='store_true', help='print software version and exit')

    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='do not plot results at the end of the processing.')
    parser.add_argument('--oy', dest='over_sample_y', default=3, help='Oversampling in azimuth direction')
    parser.add_argument('--ox', dest='over_sample_x', default=1, help='Oversampling in range direction')

    step = parser.add_argument_group('steps processing (start/end/dostep)', STEP_HELP)
    step.add_argument('--start', dest='startStep', metavar='STEP', default=STEP_LIST[0],
                      help='start processing at the named step, default: {}'.format(STEP_LIST[0]))
    step.add_argument('--end', '--stop', dest='endStep', metavar='STEP',  default=STEP_LIST[-1],
                      help='end processing at the named step, default: {}'.format(STEP_LIST[-1]))
    step.add_argument('--dostep', dest='doStep', metavar='STEP',
                      help='run processing at the named step only')
    return parser


def cmd_line_parse(iargs=None):
    """Command line parser."""
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    template_file = os.path.join(os.getenv('MINTPY_HOME'), 'mintpy/defaults/smallbaselineApp.cfg')

    # print default template
    if inps.print_template:
        raise SystemExit(open(template_file, 'r').read())

    # print software version
    if inps.version:
        raise SystemExit(mintpy.version.description)

    if (not inps.customTemplateFile
            and not os.path.isfile(os.path.basename(template_file))
            and not inps.generate_template):
        parser.print_usage()
        print(EXAMPLE)
        msg = "ERROR: no template file found! It requires:"
        msg += "\n  1) input a custom template file, OR"
        msg += "\n  2) there is a default template 'smallbaselineApp.cfg' in current directory."
        print(msg)
        raise SystemExit()

    # invalid input of custom template
    if inps.customTemplateFile:
        inps.customTemplateFile = os.path.abspath(inps.customTemplateFile)
        if not os.path.isfile(inps.customTemplateFile):
            raise FileNotFoundError(inps.customTemplateFile)
        elif os.path.basename(inps.customTemplateFile) == os.path.basename(template_file):
            # ignore if smallbaselineApp.cfg is input as custom template
            inps.customTemplateFile = None

    # check input --start/end/dostep
    for key in ['startStep', 'endStep', 'doStep']:
        value = vars(inps)[key]
        if value and value not in STEP_LIST:
            msg = 'Input step not found: {}'.format(value)
            msg += '\nAvailable steps: {}'.format(STEP_LIST)
            raise ValueError(msg)

    # ignore --start/end input if --dostep is specified
    if inps.doStep:
        inps.startStep = inps.doStep
        inps.endStep = inps.doStep

    # get list of steps to run
    idx0 = STEP_LIST.index(inps.startStep)
    idx1 = STEP_LIST.index(inps.endStep)
    if idx0 > idx1:
        msg = 'input start step "{}" is AFTER input end step "{}"'.format(inps.startStep, inps.endStep)
        raise ValueError(msg)
    inps.runSteps = STEP_LIST[idx0:idx1+1]

    # empty the step list for -g option
    if inps.generate_template:
        inps.runSteps = []

    # message - software version
    if len(inps.runSteps) <= 1:
        print(mintpy.version.description)
    else:
        print(mintpy.version.logo)

    # mssage - processing steps
    if len(inps.runSteps) > 0:
        print('--RUN-at-{}--'.format(datetime.datetime.now()))
        print('Run routine processing with {} on steps: {}'.format(os.path.basename(__file__), inps.runSteps))
        if inps.doStep:
            print('Remaining steps: {}'.format(STEP_LIST[idx0+1:]))
            print('--dostep option enabled, disable the plotting at the end of the processing.')
            inps.plot = False

    print('-'*50)
    return inps


##########################################################################
class minopyTimeSeriesAnalysis(TimeSeriesAnalysis):
    """ Routine processing workflow for time series analysis of small baseline InSAR stacks
    """
    def __init__(self, customTemplateFile=None, workDir=None, inps=None):
        super().__init__(customTemplateFile, workDir)
        self.inps = inps

    def startup(self):
        super().startup()


        # 4. Copy the plot shell file
        sh_file = os.path.join(os.getenv('MINTPY_HOME'), 'sh/plot_smallbaselineApp.sh')

        # 1) copy to work directory (if not existed yet).
        if not os.path.isfile(os.path.basename(sh_file)):
            print('copy {} to work directory: {}'.format(sh_file, self.workDir))
            shutil.copy2(sh_file, self.workDir)
        else:
            os.system('mv {f} {f}_obsolete'.format(f=os.path.basename(sh_file)))
            print('obsolete shell file detected, renamed it to: {}_obsolete'.format(os.path.basename(sh_file)))
            print('copy {} to work directory: {}'.format(sh_file, self.workDir))
            shutil.copy2(sh_file, self.workDir)
        os.system('chmod +x {}'.format(os.path.join(self.workDir, sh_file.split('/')[-1])))

        self.plot_sh_cmd = './'+os.path.basename(sh_file)

        return


    def get_phase_linking_coherence_mask(self):
        """Generate reliable pixel mask from temporal coherence"""

        tcoh_file = os.path.join(self.workDir, 'temporalCoherence.h5')
        mask_file = os.path.join(self.workDir, 'maskTempCoh.h5')

        tcoh_min = 0.3

        scp_args = '{} --nonzero -o {} --update'.format(tcoh_file, mask_file)
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
            mintpy.generate_mask.main(scp_args.split())
            # update configKeys
            atr = {}
            atr['mintpy.networkInversion.minTempCoh'] = tcoh_min
            ut.add_attribute(mask_file, atr)
            ut.add_attribute(mask_file, atr)

        # check number of pixels selected in mask file for following analysis
        num_pixel = np.sum(readfile.read(mask_file)[0] != 0.)
        print('number of reliable pixels: {}'.format(num_pixel))

        min_num_pixel = float(self.template['mintpy.networkInversion.minNumPixel'])
        if num_pixel < min_num_pixel:
            msg = "Not enough reliable pixels (minimum of {}). ".format(int(min_num_pixel))
            msg += "Try the following:\n"
            msg += "1) Check the reference pixel and make sure it's not in areas with unwrapping errors\n"
            msg += "2) Check the network and make sure it's fully connected without subsets"
            raise RuntimeError(msg)
        return

    def write_to_timeseries(self):

        inps = self.inps

        inps.timeseriesFile = os.path.join(self.workDir, 'timeseries.h5')
        inps.tempCohFile = os.path.join(self.workDir, 'temporalCoherence.h5')
        inps.timeseriesFiles = [os.path.join(self.workDir, 'timeseries.h5')]  # all ts files
        inps.outfile = [os.path.join(self.workDir, 'timeseries.h5'),
                        os.path.join(self.workDir, 'temporalCoherence.h5')]

        ifgram_file = os.path.join(self.workDir, 'inputs/ifgramStack.h5')

        stack_obj = ifgramStack(ifgram_file)
        stack_obj.open(print_msg=False)
        date_list = stack_obj.get_date_list(dropIfgram=True)
        num_date = len(date_list)

        metadata = dict(stack_obj.metadata)
        metadata['REF_DATE'] = date_list[0]
        metadata['FILE_TYPE'] = 'timeseries'
        metadata['UNIT'] = 'm'

        num_row = stack_obj.length
        num_col = stack_obj.width

        phase2range = -1 * float(stack_obj.metadata['WAVELENGTH']) / (4. * np.pi)

        box = None
        ref_phase = stack_obj.get_reference_phase(dropIfgram=False)
        unwDatasetName = 'unwrapPhase'
        mask_dataset_name = None
        mask_threshold = 0.3

        pha_data = read_unwrap_phase(stack_obj,
                                     box,
                                     ref_phase,
                                     unwDatasetName=unwDatasetName,
                                     dropIfgram=True)

        pha_data = mask_unwrap_phase(pha_data,
                                     stack_obj,
                                     box,
                                     dropIfgram=True,
                                     mask_ds_name=mask_dataset_name,
                                     mask_threshold=mask_threshold)

        ts = pha_data * phase2range
        ts0 = ts.reshape(num_date - 1, num_row, num_col)
        ts = np.zeros((num_date, num_row, num_col), np.float32)
        ts[1::, :, :] = ts0
        num_inv_ifg = np.zeros((num_row, num_col), np.int16) + num_date - 1

        gfilename = os.path.join(self.workDir, '../merged/geom_master/Quality.rdr')
        ds = gdal.Open(gfilename, gdal.GA_ReadOnly)
        quality_map = ds.GetRasterBand(1).ReadAsArray()

        if 'SUBSET_XMIN' in metadata:
            first_row = int(metadata['SUBSET_YMIN'])
            last_row = int(metadata['SUBSET_YMAX'])
            first_col = int(metadata['SUBSET_XMIN'])
            last_col = int(metadata['SUBSET_XMAX'])

            quality_map = quality_map[first_row:last_row, first_col:last_col]

        os.chdir(self.workDir)

        write2hdf5_file(ifgram_file, metadata, ts, quality_map, num_inv_ifg, suffix='', inps=inps)

        self.get_phase_linking_coherence_mask()


    def run(self, steps=STEP_LIST, plot=True):
        for sname in steps:
            print('\n\n******************** step - {} ********************'.format(sname))

            if sname == 'load_data':
                super().run_load_data(sname)

            elif sname == 'modify_network':
                super().run_network_modification(sname)

            elif sname == 'reference_point':
                super().run_reference_point(sname)

            elif sname == 'write_to_timeseries':
                self.write_to_timeseries()

            elif sname == 'correct_unwrap_error':
                if self.template['mintpy.unwrapError.method']:
                    self.template['mintpy.unwrapError.method'] = 'bridging'
                super().run_unwrap_error_correction(sname)

            elif sname == 'stack_interferograms':
                super().run_ifgram_stacking(sname)

            elif sname == 'invert_network':
                super().run_network_inversion(sname)

            elif sname == 'correct_LOD':
                super().run_local_oscillator_drift_correction(sname)

            elif sname == 'correct_troposphere':
                super().run_tropospheric_delay_correction(sname)

            elif sname == 'deramp':
                super().run_phase_deramping(sname)

            elif sname == 'correct_topography':
                super().run_topographic_residual_correction(sname)

            elif sname == 'residual_RMS':
                super().run_residual_phase_rms(sname)

            elif sname == 'reference_date':
                super().run_reference_date(sname)

            elif sname == 'velocity':
                super().run_timeseries2velocity(sname)

            elif sname == 'geocode':
                super().run_geocode(sname)

            elif sname == 'google_earth':
                super().run_save2google_earth(sname)

            elif sname == 'hdfeos5':
                super().run_save2hdfeos5(sname)

        # plot result (show aux visualization message more multiple steps processing)
        print_aux = len(steps) > 1
        super().plot_result(print_aux=print_aux, plot=plot)

        # go back to original directory
        print('Go back to directory:', self.cwd)
        os.chdir(self.cwd)

        # message
        msg = '\n################################################'
        msg += '\n   Normal end of smallbaselineApp processing!'
        msg += '\n################################################'
        print(msg)
        return


##########################################################################
def main(iargs=None):
    start_time = time.time()
    inps = cmd_line_parse(iargs)

    app = minopyTimeSeriesAnalysis(inps.customTemplateFile, inps.workDir, inps)
    app.startup()
    if len(inps.runSteps) > 0:
        app.run(steps=inps.runSteps, plot=inps.plot)

    # Timing
    m, s = divmod(time.time()-start_time, 60)
    print('Time used: {:02.0f} mins {:02.1f} secs\n'.format(m, s))
    return


###########################################################################################
if __name__ == '__main__':
    main()


