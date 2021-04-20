#!/usr/bin/env python3
# Author: Sara Mirzaee


import os
import time
import datetime
import shutil
import argparse
import mintpy
import mintpy.workflow  #dynamic import for modules used by smallbaselineApp workflow
from mintpy.objects import sensor, RAMP_LIST
from mintpy.utils import readfile, writefile, utils as ut
from mintpy.objects import ifgramStack, timeseries
from mintpy.smallbaselineApp import TimeSeriesAnalysis
import h5py
from osgeo import gdal
import numpy as np

##########################################################################

STEP_LIST = [
    'load_data',
    'modify_network',
    'reference_point',
    'correct_unwrap_error',
    'stack_interferograms',
    'invert_network',
    'invert_iono',
    'correct_LOD',
    'correct_troposphere',
    'deramp',
    'correct_topography',
    'correct_iono',
    'residual_RMS',
    'reference_date',
    'velocity',
    'geocode',
    'google_earth',
    'hdfeos5',
]
'''

STEP_LIST = [
    'load_data',
    'modify_network',
    'reference_point',
    'correct_unwrap_error',
    'correct_iono_ifg',
    'stack_interferograms',
    'invert_network',
    'correct_LOD',
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
'''



STEP_HELP = """Command line options for steps processing with names are chosen from the following list:

{}
{}
{}

In order to use either --start or --dostep, it is necessary that a
previous run was done using one of the steps options to process at least
through the step immediately preceding the starting step of the current run.
""".format(STEP_LIST[0:5], STEP_LIST[5:10], STEP_LIST[10:])

EXAMPLE = """example:
  iono_mintpy.py                         #run with default template 'smallbaselineApp.cfg'
  iono_mintpy.py <custom_template>       #run with default and custom templates
  iono_mintpy.py -h / --help             #help
  iono_mintpy.py -H                      #print    default template options
  iono_mintpy.py -g                      #generate default template if it does not exist
  iono_mintpy.py -g <custom_template>    #generate/update default template based on custom template

  # Run with --start/stop/dostep options
  iono_mintpy.py GalapagosSenDT128.template --dostep velocity  #run at step 'velocity' only
  iono_mintpy.py GalapagosSenDT128.template --end load_data    #end after step 'load_data'
"""

REFERENCE = """reference:
  Yunjun, Z., H. Fattahi, F. Amelung (2019), Small baseline InSAR time series analysis: unwrapping error
  correction and noise reduction (under review), preprint doi:10.31223/osf.io/9sz6m.
"""


def create_parser():
    parser = argparse.ArgumentParser(description='Routine Time Series Analysis for Small Baseline InSAR Stack',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=REFERENCE+'\n'+EXAMPLE)

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

    template_file = os.path.join(os.path.dirname(__file__), 'defaults/smallbaselineApp.cfg')

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
class ionoTimeSeriesAnalysis(TimeSeriesAnalysis):
    """ Routine processing workflow for time series analysis of small baseline InSAR stacks
    """
    def __init__(self, customTemplateFile=None, workDir=None):
        super().__init__(customTemplateFile, workDir)

    def startup(self):
        super().startup
        # 4. Copy the plot shell file
        sh_file = os.path.join(os.path.dirname(__file__), './sh/plot_smallbaselineApp.sh')

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

    def run_iono_inversion(self, step_name):
        """Invert network of Ionosphere phases for raw phase time-series.
        1) ionosphere network inversion --> timeseriesiono.h5, temporalCoherenceIono.h5, numInvIfgramIono.h5
        """
        # check the existence of ifgramStack.h5
        stack_file = ut.check_loaded_dataset(self.workDir, print_msg=False)[1]

        # 1) invert ionosphere for time-series
        scp_args = '{} -d iono -o iono.h5, temporalIono.h5' \
                   ' -t {} --update '.format(stack_file, self.templateFile)
        print('ifgram_inversion.py', scp_args)
        mintpy.ifgram_inversion.main(scp_args.split())

        return

    @staticmethod
    def get_timeseries_filename(template):
        """Get input/output time-series filename for each step
        Parameters: template : dict, content of smallbaselineApp.cfg
        Returns:    steps    : dict of dicts, input/output filenames for each step
        """
        steps = dict()
        fname0 = 'timeseries.h5'
        fname1 = 'timeseries.h5'
        atr = readfile.read_attribute(fname0)

        # loop for all steps
        phase_correction_steps = ['correct_LOD', 'correct_troposphere', 'deramp', 'correct_topography', 'correct_iono']
        for sname in phase_correction_steps:
            # fname0 == fname1 if no valid correction method is set.
            fname0 = fname1
            if sname == 'correct_LOD':
                if atr['PLATFORM'].lower().startswith('env'):
                    fname1 = '{}_LODcor.h5'.format(os.path.splitext(fname0)[0])

            elif sname == 'correct_troposphere':
                method = template['mintpy.troposphericDelay.method']
                model = template['mintpy.troposphericDelay.weatherModel']
                if method:
                    if method == 'height_correlation':
                        fname1 = '{}_tropHgt.h5'.format(os.path.splitext(fname0)[0])

                    elif method == 'pyaps':
                        fname1 = '{}_{}.h5'.format(os.path.splitext(fname0)[0], model)

                    else:
                        msg = 'un-recognized tropospheric correction method: {}'.format(method)
                        raise ValueError(msg)

            elif sname == 'deramp':
                method = template['mintpy.deramp']
                if method:
                    if method in RAMP_LIST:
                        fname1 = '{}_ramp.h5'.format(os.path.splitext(fname0)[0])
                    else:
                        msg = 'un-recognized phase ramp type: {}'.format(method)
                        msg += '\navailable ramp types:\n{}'.format(RAMP_LIST)
                        raise ValueError(msg)

            elif sname == 'correct_topography':
                method = template['mintpy.topographicResidual']
                if method:
                    fname1 = '{}_demErr.h5'.format(os.path.splitext(fname0)[0])

            elif sname == 'correct_iono':
                    fname1 = '{}_iono.h5'.format(os.path.splitext(fname0)[0])

            step = dict()
            step['input'] = fname0
            step['output'] = fname1
            steps[sname] = step

        # step - reference_date
        fnames = [steps[sname]['output'] for sname in phase_correction_steps]
        fnames += [steps[sname]['input'] for sname in phase_correction_steps]
        fnames = sorted(list(set(fnames)))
        step = dict()
        step['input'] = fnames
        steps['reference_date'] = step

        # step - velocity / geocode
        step = dict()
        step['input'] = steps['reference_date']['input'][-1]
        steps['velocity'] = step
        steps['geocode'] = step

        # step - hdfeos5
        if 'Y_FIRST' not in atr.keys():
            step = dict()
            step['input'] = './geo/geo_{}'.format(steps['reference_date']['input'][-1])
        steps['hdfeos5'] = step
        return steps

    def run_ionospheric_correction(self, step_name):
        fnames = self.get_timeseries_filename(self.template)[step_name]
        in_file = fnames['input']
        out_file = fnames['output']
        iono_file = './iono.h5'

        def get_dataset_size(fname):
            atr = readfile.read_attribute(fname)
            return (atr['LENGTH'], atr['WIDTH'])

        if ut.run_or_skip(out_file=out_file, in_file=[in_file, iono_file]) == 'run':
            if os.path.isfile(iono_file) and get_dataset_size(iono_file) == get_dataset_size(in_file):
                scp_args = '{f} {t} -o {o} --force'.format(f=in_file, t=iono_file, o=out_file)
                print('--------------------------------------------')
                print('Use existed tropospheric delay file: {}'.format(iono_file))
                print('diff.py', scp_args)
                mintpy.diff.main(scp_args.split())

        return

    def run_ionospheric_correction_ifgrams(self, step_name):

        ifgram_file = os.path.join(self.workDir, 'inputs/ifgramStack.h5')

        stack_obj = ifgramStack(ifgram_file)
        stack_obj.open(print_msg=False)
        unwDatasetName = [i for i in ['unwrapPhase_bridging_phaseClosure',
                                      'unwrapPhase_bridging',
                                      'unwrapPhase_phaseClosure',
                                      'unwrapPhase'] if i in stack_obj.datasetNames][0]
        ionoDatasetName = 'iono'

        dsNameOut = unwDatasetName + '_' + ionoDatasetName
        atr = readfile.read_attribute(ifgram_file)
        length, width = int(atr['LENGTH']), int(atr['WIDTH'])
        date12_list = ifgramStack(ifgram_file).get_date12_list(dropIfgram=False)
        num_ifgram = len(date12_list)
        shape_out = (num_ifgram, length, width)

        pha_data = stack_obj.read(datasetName=unwDatasetName,
                                  box=None,
                                  dropIfgram=False,
                                  print_msg=False)

        iono_data = stack_obj.read(datasetName=ionoDatasetName,
                                  box=None,
                                  dropIfgram=False,
                                  print_msg=False)

        corrected_iono = pha_data - iono_data

        f = h5py.File(ifgram_file, 'r+')
        if dsNameOut in f.keys():
            ds = f[dsNameOut]
            print('access /{d} of np.float32 in size of {s}'.format(d=dsNameOut, s=shape_out))
        else:
            ds = f.create_dataset(dsNameOut,
                                  shape_out,
                                  maxshape=(None, None, None),
                                  chunks=True,
                                  compression=None)
            print('create /{d} of np.float32 in size of {s}'.format(d=dsNameOut, s=shape_out))

        ds[:, :, :] = corrected_iono
        ds.attrs['MODIFICATION_TIME'] = str(time.time())
        f.close()
        print('close {} file.'.format(ifgram_file))

        return

    def run_network_inversion(self, step_name):
        """Invert network of interferograms for raw phase time-series.
        1) network inversion --> timeseries.h5, temporalCoherence.h5, numInvIfgram.h5
        2) temporalCoherence.h5 --> maskTempCoh.h5
        """
        ifgram_file = os.path.join(self.workDir, 'inputs/ifgramStack.h5')

        stack_obj = ifgramStack(ifgram_file)
        stack_obj.open(print_msg=False)


        unwDatasetName = [i for i in ['unwrapPhase_bridging_phaseClosure',
                                      'unwrapPhase_bridging',
                                      'unwrapPhase_phaseClosure',
                                      'unwrapPhase']
                          if i in stack_obj.datasetNames][0]
        '''

        unwDatasetName = [i for i in ['unwrapPhase_bridging_phaseClosure_iono',
                                      'unwrapPhase_bridging_phaseClosure',
                                      'unwrapPhase_bridging_iono',
                                      'unwrapPhase_bridging',
                                      'unwrapPhase_phaseClosure_iono',
                                      'unwrapPhase_phaseClosure',
                                      'unwrapPhase_iono',
                                      'unwrapPhase']
                          if i in stack_obj.datasetNames][0]

        '''
        # check the existence of ifgramStack.h5
        stack_file = ut.check_loaded_dataset(self.workDir, print_msg=False)[1]

        # 1) invert ifgramStack for time-series
        scp_args = '{} --dset {} -t {} --update '.format(stack_file, unwDatasetName, self.templateFile)
        print('ifgram_inversion.py', scp_args)
        mintpy.ifgram_inversion.main(scp_args.split())

        # 2) get reliable pixel mask: maskTempCoh.h5
        self.generate_temporal_coherence_mask()

        return

    def run(self, steps=STEP_LIST, plot=True):
        for sname in steps:
            print('\n\n******************** step - {} ********************'.format(sname))

            if sname == 'load_data':
                super().run_load_data(sname)

            elif sname == 'modify_network':
                super().run_network_modification(sname)

            elif sname == 'reference_point':
                super().run_reference_point(sname)

            elif sname == 'correct_unwrap_error':
                super().run_unwrap_error_correction(sname)

            elif sname == 'correct_iono_ifg':
                self.run_ionospheric_correction_ifgrams(sname)

            elif sname == 'stack_interferograms':
                super().run_ifgram_stacking(sname)

            elif sname == 'invert_network':
                self.run_network_inversion(sname)

            elif sname == 'invert_iono':
                self.run_iono_inversion(sname)

            elif sname == 'correct_LOD':
                super().run_local_oscillator_drift_correction(sname)

            elif sname == 'correct_troposphere':
                super().run_tropospheric_delay_correction(sname)

            elif sname == 'deramp':
                super().run_phase_deramping(sname)

            elif sname == 'correct_topography':
                super().run_topographic_residual_correction(sname)

            elif sname == 'correct_iono':
                self.run_ionospheric_correction(sname)

            elif sname == 'residual_RMS':
                super().run_residual_phase_rms(sname)

            elif sname == 'reference_date':
                super().run_reference_date(sname)

            elif sname == 'velocity':
                super().run_timeseries2velocity(sname)

            elif sname == 'geocode':

                incfile = os.path.abspath(self.workDir + '/../geom_master/incLocal.rdr.vrt')
                ds = gdal.Open(incfile, gdal.GA_ReadOnly)
                inclocal = ds.GetRasterBand(2).ReadAsArray()

                geo_file = os.path.join(self.workDir, 'inputs/geometryRadar.h5')
                f = h5py.File(geo_file, 'a')
                if 'incLocal' in f.keys():
                    del f['incLocal']
                ds = f.create_dataset('incLocal',
                                      data=inclocal,
                                      dtype=np.float32,
                                      chunks=True,
                                      compression='lzf')

                ds.attrs['MODIFICATION_TIME'] = str(time.time())
                f.close()

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
    app = ionoTimeSeriesAnalysis(inps.customTemplateFile, inps.workDir)
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


