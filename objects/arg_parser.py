#! /usr/bin/env python3
###############################################################################
# Project: Argument parser for minopy
# Author: Sara Mirzaee
###############################################################################
import argparse
from minopy.defaults import auto_path
import mintpy
import os
import datetime

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

    def parse(self):

        if self.script == 'crop_images':
            self.parser = self.crop_image_parser()
        elif self.script == 'create_patch':
            self.parser = self.create_patch_parser()
        elif self.script == 'patch_inversion':
            self.parser = self.patch_inversion_parser()
        elif self.script == 'generate_ifgram':
            self.parser = self.generate_ifgrams_parser()
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
                    """.format(auto_path.isceAutoPath,
                               auto_path.roipacAutoPath,
                               auto_path.gammaAutoPath)

        if inps.template_file:
            pass
        elif inps.print_example_template:
            raise SystemExit(DEFAULT_TEMPLATE)
        else:
            parser.print_usage()
            print(('{}: error: one of the following arguments are required:'
                   ' -t/--template, -H'.format(os.path.basename(__file__))))
            print('{} -H to show the example template file'.format(os.path.basename(__file__)))
            sys.exit(1)

        inps.outfile = [os.path.abspath(i) for i in inps.outfile]
        inps.outdir = os.path.dirname(inps.outfile[0])
        return inps

    def out_minopy_wrapper(self, sinps):
        inps = sinps
        STEP_LIST = self.STEP_LIST
        template_file = os.path.join(os.path.dirname(os.getenv('MINOPY_HOME')), 'defaults/minopy_templates.cfg')

        # print default template
        if inps.print_template:
            raise SystemExit(open(template_file, 'r').read(), )

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
        inps.runSteps = STEP_LIST[idx0:idx1 + 1]

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
                print('Remaining steps: {}'.format(STEP_LIST[idx0 + 1:]))
                print('--dostep option enabled, disable the plotting at the end of the processing.')
                inps.plot = False

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
        mintpy.load.processor      = auto  #[isce,snap,gamma,roipac], auto for isce
        mintpy.load.updateMode     = auto  #[yes / no], auto for yes, skip re-loading if HDF5 files are complete
        mintpy.load.compression    = auto  #[gzip / lzf / no], auto for no.
        ##---------for ISCE only:
        mintpy.load.metaFile       = auto  #[path2metadata_file], i.e.: ./master/IW1.xml, ./masterShelve/data.dat
        mintpy.load.baselineDir    = auto  #[path2baseline_dir], i.e.: ./baselines
        ##---------slc datasets:
        mintpy.load.slcFile        = auto  #[path2slc]
        ##---------geometry datasets:
        mintpy.load.demFile        = auto  #[path2hgt_file]
        mintpy.load.lookupYFile    = auto  #[path2lat_file], not required for geocoded data
        mintpy.load.lookupXFile    = auto  #[path2lon_file], not required for geocoded data
        mintpy.load.incAngleFile   = auto  #[path2los_file], optional
        mintpy.load.azAngleFile    = auto  #[path2los_file], optional
        mintpy.load.shadowMaskFile = auto  #[path2shadow_file], optional
        mintpy.load.waterMaskFile  = auto  #[path2water_mask_file], optional
        mintpy.load.bperpFile      = auto  #[path2bperp_file], optional
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
                            choices={'isce', 'snap', 'gamma', 'roipac', 'doris', 'gmtsar'},
                            help='InSAR processor/software of the file (This version only supports isce)',
                            default='isce')
        parser.add_argument('--enforce', '-f', dest='updateMode', action='store_false',
                            help='Disable the update mode, or skip checking dataset already loaded.')
        parser.add_argument('--compression', choices={'gzip', 'lzf', None}, default=None,
                            help='compress loaded geometry while writing HDF5 file, default: None.')

        parser.add_argument('-o', '--output', type=str, nargs=3, dest='outfile',
                            default=['./inputs/slcStack.h5',
                                     './inputs/geometryRadar.h5',
                                     './inputs/geometryGeo.h5'],
                            help='output HDF5 file')
        return parser

    def create_patch_parser(self):
        parser = self.parser
        patch = parser.add_argument_group('Crop options')
        patch.add_argument('-w', '--workDir', type=str, dest='work_dir', help='minopy directory')
        patch.add_argument('-r', '--rangeWin', type=int, dest='range_window', default=15,
                           help='range window size for shp finding')
        patch.add_argument('-a', '--azimuthWin', type=int, dest='azimuth_window', default=15,
                           help='azimuth window size for shp finding')
        patch.add_argument('-s', '--patchSize', type=int, dest='patch_size', default=200, help='azimuth window size for shp finding')
        return parser

    def patch_inversion_parser(self):
        parser = self.parser
        patch = parser.add_argument_group('Patch inversion option')
        patch.add_argument('-w', '--workDir', type=str, dest='work_dir', help='minopy directory')
        patch.add_argument('-r', '--rangeWin', type=int, dest='range_window', default=15,
                           help='range window size for shp finding')
        patch.add_argument('-a', '--azimuthWin', type=int, dest='azimuth_window', default=15,
                           help='azimuth window size for shp finding')
        patch.add_argument('-m', '--method', type=str, dest='inversion_method', default='EMI',
                           help='inversion method (EMI, EVD, PTA, sequential_EMI, ...)')
        patch.add_argument('-t', '--test', type=str, dest='shp_test', default='ks',
                           help='shp statistical test (ks, ad, ttest)')
        patch.add_argument('-p', '--patch', type=str, dest='patch_dir', help='patch directory ex: patch1_4')
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
        parser.add_argument('-x', '--ifg_index', dest='ifg_index', type=str, required=True,
                            help='interferogram index in 3D array (inversion results)')
        parser.add_argument('-r', '--range_window', dest='range_win', type=str, default='15'
                            , help='SHP searching window size in range direction. -- Default : 15')
        parser.add_argument('-a', '--azimuth_window', dest='azimuth_win', type=str, default='15'
                            , help='SHP searching window size in azimuth direction. -- Default : 15')
        parser.add_argument('-q', '--acquisition_number', dest='n_image', type=str, default='20',
                            help='number of images acquired')
        parser.add_argument('-A', '--azimuth_looks', type=str, dest='azimuth_looks', default='3', help='azimuth looks')
        parser.add_argument('-R', '--range_looks', type=str, dest='range_looks', default='9', help='range looks')

        return parser

    def minopy_wrapper_parser(self):

        STEP_LIST = [
            'crop',
            'create_patch',
            'inversion',
            'ifgrams',
            'unwrap'
            'load_data',
            'modify_network',
            'reference_point',
            'write_to_timeseries',
            'correct_unwrap_error',
            'stack_interferograms',
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

        step = parser.add_argument_group('steps processing (start/end/dostep)', STEP_HELP)
        step.add_argument('--start', dest='startStep', metavar='STEP', default=STEP_LIST[0],
                          help='start processing at the named step, default: {}'.format(STEP_LIST[0]))
        step.add_argument('--end', '--stop', dest='endStep', metavar='STEP', default=STEP_LIST[-1],
                          help='end processing at the named step, default: {}'.format(STEP_LIST[-1]))
        step.add_argument('--dostep', dest='doStep', metavar='STEP',
                          help='run processing at the named step only')

        self.STEP_LIST = STEP_HELP

        return parser







