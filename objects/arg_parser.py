#! /usr/bin/env python3
###############################################################################
# Project: Argument parser for minopy
# Author: Sara Mirzaee
###############################################################################
import sys
import argparse
from minopy.defaults import auto_path
from minopy.defaults.auto_path import autoPath, PathFind
import minopy
import os
import datetime
pathObj = PathFind()

CLUSTER_LIST = ['lsf', 'pbs', 'slurm', 'local']



class MinoPyParser:

    def __init__(self, iargs=None, script=None):
        self.iargs = iargs
        self.script = script
        self.parser = argparse.ArgumentParser(description='MiNoPy scripts parser')
        commonp = self.parser.add_argument_group('General options:')
        commonp.add_argument('-v', '--version', action='store_true', help='print software version and exit')
        commonp.add_argument('--submit', dest='submit_flag', action='store_true', help='submits job')
        commonp.add_argument('--walltime', dest='wall_time', default='None',
                             help='walltime for submitting the script as a job')
        commonp.add_argument('--queue', dest='queue_name', default=None, help='Queue name')

    def parse(self):

        if self.script == 'crop_images':
            self.parser = self.crop_image_parser()
        elif self.script == 'create_patch':
            self.parser = self.create_patch_parser()
        elif self.script == 'patch_inversion':
            self.parser = self.patch_inversion_parser()
        elif self.script == 'phase_inversion':
            self.parser = self.phase_inversion_parser()
        elif self.script == 'generate_ifgram':
            self.parser = self.generate_ifgrams_parser()
        elif self.script == 'generate_interferograms':
            self.parser = self.generate_interferograms_parser()
        elif self.script == 'minopy_wrapper':
            self.parser = self.minopy_wrapper_parser()

        inps = self.parser.parse_args(args=self.iargs)

        if self.script == 'crop_images':
            inps = self.out_crop_image(inps)

        if self.script == 'minopy_wrapper':
            inps = self.out_minopy_wrapper(inps)

        return inps

    def out_crop_image(self, sinps):
        inps = sinps
        DEFAULT_TEMPLATE = """template:
                    ########## 1. Load Data (--load to exit after this step)
                    {}\n
                    {}\n
                    {}\n
                    {}\n
                    """.format(auto_path.isceTopsAutoPath,
                               auto_path.isceStripmapAutoPath,
                               auto_path.roipacAutoPath,
                               auto_path.gammaAutoPath)

        if inps.template_file:
            pass
        elif inps.print_example_template:
            raise SystemExit(DEFAULT_TEMPLATE)
        else:
            self.parser.print_usage()
            print(('{}: error: one of the following arguments are required:'
                   ' -t/--template, -H'.format(os.path.basename(__file__))))
            print('{} -H to show the example template file'.format(os.path.basename(__file__)))
            sys.exit(1)

        inps.out_file = [os.path.abspath(i) for i in inps.out_file]
        inps.out_dir = os.path.dirname(inps.out_file[0])
        return inps

    def out_minopy_wrapper(self, sinps):
        inps = sinps
        STEP_LIST = self.STEP_LIST
        template_file = os.path.join(os.path.abspath(os.getenv('MINOPY_HOME')), 'defaults/minopy_template.cfg')

        # print default template
        if inps.print_template:
            raise SystemExit(open(template_file, 'r').read(), )

        # print software version
        if inps.version:
            raise SystemExit(minopy.version.description)

        if (not inps.customTemplateFile
                and not os.path.isfile(os.path.basename(template_file))
                and not inps.generate_template):
            self.parser.print_usage()
            #print(EXAMPLE)
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
        inps.runSteps = STEP_LIST[idx0:idx1 + 1]

        # empty the step list for -g option
        if inps.generate_template:
            inps.runSteps = []

        # message - software version
        if len(inps.runSteps) <= 1:
            print(minopy.version.description)
        else:
            print(minopy.version.logo)

        # mssage - processing steps
        if len(inps.runSteps) > 0:
            print('--RUN-at-{}--'.format(datetime.datetime.now()))
            print('Run routine processing with {} on steps: {}'.format(os.path.basename(__file__), inps.runSteps))
            if inps.doStep:
                print('Remaining steps: {}'.format(STEP_LIST[idx0 + 1:]))
                print('--dostep option enabled, disable the plotting at the end of the processing.')
                inps.plot = False

        current_dir = os.getcwd()
        if not inps.workDir:
            if 'minopy' in current_dir:
                inps.workDir = current_dir.split('minopy')[0] + 'minopy'
            else:
                inps.workDir = os.path.join(current_dir, 'minopy')

        inps.workDir = os.path.abspath(inps.workDir)

        inps.project_name = None
        if inps.customTemplateFile and not os.path.basename(inps.customTemplateFile) == 'minopy_template.cfg':
            inps.project_name = os.path.splitext(os.path.basename(inps.customTemplateFile))[0]
            print('Project name:', inps.project_name)
        else:
            inps.project_name = os.path.dirname(inps.workDir)

        if not os.path.exists(inps.workDir):
            os.mkdir(inps.workDir)

        print('-' * 50)
        return inps

    @staticmethod
    def crop_image_parser():

        TEMPLATE = """template:
        ########## 1. Load Data
        ## auto - automatic path pattern for Univ of Miami file structure
        ## load_data.py -H to check more details and example inputs.
        ## compression to save disk usage for slcStack.h5 file:
        ## no   - save   0% disk usage, fast [default]
        ## lzf  - save ~57% disk usage, relative slow
        ## gzip - save ~62% disk usage, very slow [not recommend]
        MINOPY.load.processor      = auto  #[isce,snap,gamma,roipac], auto for isce
        MINOPY.load.updateMode     = auto  #[yes / no], auto for yes, skip re-loading if HDF5 files are complete
        MINOPY.load.compression    = auto  #[gzip / lzf / no], auto for no.
        ##---------for ISCE only:
        MINOPY.load.metaFile       = auto  #[path2metadata_file], i.e.: ./master/IW1.xml, ./masterShelve/data.dat
        MINOPY.load.baselineDir    = auto  #[path2baseline_dir], i.e.: ./baselines
        ##---------slc datasets:
        MINOPY.load.slcFile        = auto  #[path2slc]
        ##---------geometry datasets:
        MINOPY.load.demFile        = auto  #[path2hgt_file]
        MINOPY.load.lookupYFile    = auto  #[path2lat_file], not required for geocoded data
        MINOPY.load.lookupXFile    = auto  #[path2lon_file], not required for geocoded data
        MINOPY.load.incAngleFile   = auto  #[path2los_file], optional
        MINOPY.load.azAngleFile    = auto  #[path2los_file], optional
        MINOPY.load.shadowMaskFile = auto  #[path2shadow_file], optional
        MINOPY.load.waterMaskFile  = auto  #[path2water_mask_file], optional
        MINOPY.load.bperpFile      = auto  #[path2bperp_file], optional
        ##---------subset (optional):
        ## if both yx and lalo are specified, use lalo option unless a) no lookup file AND b) dataset is in radar coord
        mintpy.subset.yx   = auto    #[1800:2000,700:800 / no], auto for no
        mintpy.subset.lalo = auto    #[31.5:32.5,130.5:131.0 / no], auto for no
        """

        EXAMPLE = """example:
          crop_images.py -t GalapagosSenDT128.tempalte
          crop_images.py -t smallbaselineApp.cfg
          crop_images.py -t smallbaselineApp.cfg GalapagosSenDT128.tempalte --project GalapagosSenDT128
          crop_images.py -H #Show example input template for ISCE/ROI_PAC/GAMMA products
        """

        parser = argparse.ArgumentParser(description='Saving a stack of Interferograms to an HDF5 file',
                                         formatter_class=argparse.RawTextHelpFormatter,
                                         epilog=TEMPLATE + '\n' + EXAMPLE)
        parser.add_argument('-H', dest='print_example_template', action='store_true',
                            help='Print/Show the example template file for loading.')
        parser.add_argument('-t', '--template', type=str, nargs='+',
                            dest='template_file', help='template file with path info.')

        parser.add_argument('--project', type=str, dest='PROJECT_NAME',
                            help='project name of dataset for INSARMAPS Web Viewer')
        parser.add_argument('--processor', type=str, dest='processor',
                            choices={'isce', 'gamma', 'roipac'},
                            help='InSAR processor/software of the file (This version only supports isce)',
                            default='isce')
        parser.add_argument('--enforce', '-f', dest='updateMode', action='store_false',
                            help='Disable the update mode, or skip checking dataset already loaded.')
        parser.add_argument('--compression', choices={'gzip', 'lzf', None}, default=None,
                            help='compress loaded geometry while writing HDF5 file, default: None.')

        parser.add_argument('-o', '--output', type=str, nargs=3, dest='out_file',
                            default=['./inputs/slcStack.h5',
                                     './inputs/geometryRadar.h5',
                                     './inputs/geometryGeo.h5'],
                            help='output HDF5 file')
        return parser

    def phase_inversion_parser(self):
        parser = self.parser
        patch = parser.add_argument_group('Phase inversion option')
        patch.add_argument('-w', '--workDir', type=str, dest='work_dir', help='minopy directory')
        patch.add_argument('-r', '--rangeWin', type=int, dest='range_window', default=15,
                           help='range window size for shp finding')
        patch.add_argument('-a', '--azimuthWin', type=int, dest='azimuth_window', default=15,
                           help='azimuth window size for shp finding')
        patch.add_argument('-m', '--method', type=str, dest='inversion_method', default='EMI',
                           help='inversion method (EMI, EVD, PTA, sequential_EMI, ...)')
        patch.add_argument('-t', '--test', type=str, dest='shp_test', default='ks',
                           help='shp statistical test (ks, ad, ttest)')
        patch.add_argument('-p', '--patchSize', type=int, dest='patch_size', default=200,
                           help='azimuth window size for shp finding')
        patch.add_argument('-s', '--slcStack', type=str, dest='slc_stack', help='SLC stack file')
       
        par = parser.add_argument_group('parallel', 'parallel processing using dask')
        par.add_argument('-c', '--cluster', '--cluster-type', dest='cluster', type=str,
                         default='local', choices=CLUSTER_LIST + ['no'],
                         help='Cluster to use for parallel computing, no to turn OFF. (default: %(default)s).')
        par.add_argument('--num-worker', dest='numWorker', type=str, default='4',
                         help='Number of workers to use (default: %(default)s).')
        par.add_argument('--config', '--config-name', dest='config', type=str, default=None,
                         help='Configuration name to use in dask.yaml (default: %(default)s).')
        return parser

    @staticmethod
    def generate_ifgrams_parser():

        parser = argparse.ArgumentParser(description='Generate interferogram, spatial and temporal coherence from '
                                                     'inversion outputs')
        parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
        parser.add_argument('-w', '--workDir', dest='work_dir', type=str, required=True,
                            help='minopy directory (inversion results)')
        parser.add_argument('-i', '--ifgDir', dest='ifg_dir', type=str, required=True,
                            help='interferogram directory')
        parser.add_argument('-r', '--rangeWindow', dest='range_win', type=str, default='15'
                            , help='SHP searching window size in range direction. -- Default : 15')
        parser.add_argument('-a', '--azimuthWindow', dest='azimuth_win', type=str, default='15'
                            , help='SHP searching window size in azimuth direction. -- Default : 15')
        return parser

    @staticmethod
    def generate_interferograms_parser():

        parser = argparse.ArgumentParser(description='Generate interferogram, spatial and temporal coherence from '
                                                     'inversion outputs')
        parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
        parser.add_argument('-s', '--slcFile', dest='slc_file', type=str, required=True,
                            help='Inverted interferograms')
        parser.add_argument('-o', '--outDir', dest='ifg_dir', type=str, required=True,
                            help='interferogram directory')
        parser.add_argument('-b1', '--band_master', dest='band_master', type=int, default=1
                            , help='master band in rslc_ref file. -- default = 1')
        parser.add_argument('-b2', '--band_slave', dest='band_slave', type=int, default=None
                            , help='master band in rslc_ref file. -- default = None')
        parser.add_argument('-p', '--prefix', dest='prefix', type=str, default='tops'
                            , help='ISCE stack processor: options= tops, stripmap -- default = tops')
        return parser

    def minopy_wrapper_parser(self):

        STEP_LIST = [
            'crop',
            'inversion',
            'ifgrams',
            'unwrap',
            'load_int',
            'reference_point',
            'correct_unwrap_error',
            'write_to_timeseries',
            'correct_troposphere',
            'deramp',
            'correct_topography',
            'residual_RMS',
            'reference_date',
            'velocity',
            'geocode',
            'google_earth',
            'hdfeos5',
            'plot',
            'email', ]

        STEP_HELP = """Command line options for steps processing with names are chosen from the following list:
        {}
        {}
        {}

        In order to use either --start or --step, it is necessary that a
        previous run was done using one of the steps options to process at least
        through the step immediately preceding the starting step of the current run.
        """.format(STEP_LIST[0:7], STEP_LIST[7:14], STEP_LIST[14:])

        EXAMPLE = """example: 
              minopy_wrapper.py  <custom_template_file>              # run with default and custom templates
              minopy_wrapper.py  <custom_template_file>  --submit    # submit as job
              minopy_wrapper.py  -h / --help                       # help 
              minopy_wrapper.py  -H                                # print    default template options
              # Run with --start/stop/step options
              minopy_wrapper.py GalapagosSenDT128.template --dostep  crop       # run the step 'download' only
              minopy_wrapper.py GalapagosSenDT128.template --start crop         # start from the step 'download' 
              minopy_wrapper.py GalapagosSenDT128.template --stop  unwrap       # end after step 'interferogram'
              """

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
        parser.add_argument('-v', '--version', action='store_true', help='print software version and exit')

        parser.add_argument('--noplot', dest='plot', action='store_false',
                            help='do not plot results at the end of the processing.')

        parser.add_argument('--submit', dest='submit_flag', action='store_true', help='submits job')
        parser.add_argument('--walltime', dest='wall_time', default='None',
                            help='walltime for submitting the script as a job')
        parser.add_argument('--queue', dest='queue_name', default=None, help='Queue name')

        step = parser.add_argument_group('steps processing (start/end/dostep)', STEP_HELP)
        step.add_argument('--start', dest='startStep', metavar='STEP', default=STEP_LIST[0],
                          help='start processing at the named step, default: {}'.format(STEP_LIST[0]))
        step.add_argument('--end', '--stop', dest='endStep', metavar='STEP', default=STEP_LIST[-1],
                          help='end processing at the named step, default: {}'.format(STEP_LIST[-1]))
        step.add_argument('--dostep', dest='doStep', metavar='STEP',
                          help='run processing at the named step only')

        self.STEP_LIST = STEP_LIST

        return parser







