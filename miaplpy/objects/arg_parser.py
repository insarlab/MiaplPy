#! /usr/bin/env python3
###############################################################################
# Project: Argument parser for miaplpy
# Author: Sara Mirzaee
###############################################################################
import sys
import argparse
import miaplpy
from miaplpy.defaults import auto_path
import os
import datetime
pathObj = auto_path.PathFind()

CLUSTER_LIST = ['lsf', 'pbs', 'slurm', 'local']


class MiaplPyParser:

    def __init__(self, iargs=None, script=None):
        self.iargs = iargs
        self.script = script
        self.parser = argparse.ArgumentParser(description='MiaplPy scripts parser')
        commonp = self.parser.add_argument_group('General options:')
        commonp.add_argument('-v', '--version', action='store_true', help='Print software version and exit')
        #commonp.add_argument('--walltime', dest='wall_time', default='None',
        #                    help='Walltime for submitting the script as a job')
        #commonp.add_argument('--queue', dest='queue', default=None, help='Queue name')
        #commonp.add_argument('--submit', dest='submit_flag', action='store_true', help='submits job')

    def parse(self):

        if self.script == 'load_slc':
            self.parser = self.load_slc_parser()
        elif self.script == 'phase_linking':
            self.parser = self.phase_linking_parser()
        elif self.script == 'generate_interferograms':
            self.parser = self.generate_interferograms_parser()
        elif self.script == 'generate_mask':
            self.parser = self.generate_unwrap_mask_parser()
        elif self.script == 'unwrap_miaplpy':
            self.parser = self.unwrap_parser()
        elif self.script == 'generate_temporal_coherence':
            self.parser = self.generate_temporal_coherence_parser()
        elif self.script == 'invert_network':
            self.parser = self.network_inversion_parser()
        elif self.script == 'miaplpy_app':
            self.parser, self.STEP_LIST, EXAMPLE = self.miaplpy_app_parser()

        inps = self.parser.parse_args(args=self.iargs)

        if self.script == 'load_slc':
            inps = self.out_load_slc(inps)

        if self.script == 'miaplpy_app':
            inps = self.out_miaplpy_app(inps, EXAMPLE)

        return inps

    def out_load_slc(self, sinps):
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

        inps.project_dir = os.path.abspath(inps.project_dir)
        inps.PROJECT_NAME = os.path.basename(inps.project_dir)

        if inps.work_dir is None:
            inps.work_dir = os.path.abspath(os.path.join(inps.project_dir, 'miaplpy'))
        else:
            inps.work_dir = os.path.abspath(inps.work_dir)

        os.makedirs(inps.work_dir, exist_ok=True)
        inps.out_dir = os.path.join(inps.work_dir, 'inputs')
        os.makedirs(inps.out_dir, exist_ok=True)
        inps.out_file = [os.path.join(inps.out_dir, i) for i in inps.out_file]

        return inps

    def out_miaplpy_app(self, sinps, EXAMPLE):
        inps = sinps
        template_file = os.path.join(os.path.dirname(miaplpy.__file__), 'defaults/miaplpyApp.cfg')
        template_file_print = os.path.join(os.path.dirname(miaplpy.__file__), 'defaults/miaplpy_mintpy_print.cfg')

        # print default template
        if inps.print_template:
            raise SystemExit(open(template_file_print, 'r').read(), )

        # print software version
        if inps.version:
            raise SystemExit(miaplpy.version.description)

        if (not inps.customTemplateFile
                and not os.path.isfile(os.path.basename(template_file))
                and not inps.generate_template):
            self.parser.print_usage()
            print(EXAMPLE)
            msg = "ERROR: no template file found! It requires:"
            msg += "\n  1) input a custom template file, OR"
            msg += "\n  2) there is a default template 'miaplpyApp.cfg' in current directory."
            print(msg)
            raise SystemExit()

        # invalid input of custom template
        if inps.customTemplateFile:
            inps.customTemplateFile = os.path.abspath(inps.customTemplateFile)
            if not os.path.isfile(inps.customTemplateFile):
                raise FileNotFoundError(inps.customTemplateFile)

            # ignore if miaplpy_template.cfg is input as custom template
            if os.path.basename(inps.customTemplateFile) == os.path.basename(template_file):
                inps.templateFile = inps.customTemplateFile
                inps.customTemplateFile = None

        # check input --start/end/dostep
        inps = self.read_inps2run_steps(inps)

        return inps

    def read_inps2run_steps(self, inps):
        """read/get run_steps from input arguments."""

        # check input --start/end/dostep
        for key in ['startStep', 'endStep', 'doStep']:
            value = vars(inps)[key]
            if value and value not in self.STEP_LIST:
                if not value == 'multilook':
                    msg = 'Input step not found: {}'.format(value)
                    msg += '\nAvailable steps: {}'.format(self.STEP_LIST)
                    raise ValueError(msg)

        # ignore --start/end input if --dostep is specified
        if inps.doStep:
            inps.startStep = inps.doStep
            inps.endStep = inps.doStep

        # get list of steps to run
        idx0 = self.STEP_LIST.index(inps.startStep)
        idx1 = self.STEP_LIST.index(inps.endStep)

        if idx0 > idx1:
            msg = 'input start step "{}" is AFTER input end step "{}"'.format(inps.startStep, inps.endStep)
            raise ValueError(msg)

        inps.run_steps = self.STEP_LIST[idx0:idx1 + 1]

        # empty the step list for -g option
        if inps.generate_template:
            inps.run_steps = []

        print('-' * 50)
        # message - processing steps
        if len(inps.run_steps) > 0:
            # for single step - compact version info
            if len(inps.run_steps) == 1:
                print(miaplpy.version.version_description)
            else:
                print(miaplpy.version.logo)
            print('--RUN-at-{}--'.format(datetime.datetime.now()))
            print('Current directory: {}'.format(os.getcwd()))
            print('Run routine processing with {} on steps: {}'.format(os.path.basename(__file__), inps.run_steps))
            print('Remaining steps: {}'.format(self.STEP_LIST[idx0 + 1:]))

        if inps.workDir is None:
            if inps.customTemplateFile:
                path1 = os.path.dirname(inps.customTemplateFile)
                inps.workDir = path1 + '/miaplpy'
            else:
                inps.workDir = os.path.dirname(inps.templateFile)

        inps.workDir = os.path.abspath(inps.workDir)

        inps.projectName = None
        if inps.customTemplateFile and not os.path.basename(inps.customTemplateFile) == 'miaplpyApp.cfg':
            inps.projectName = os.path.splitext(os.path.basename(inps.customTemplateFile))[0]
            print('Project name:', inps.projectName)
        else:
            inps.projectName = os.path.basename(os.path.dirname(inps.workDir))

        if not os.path.exists(inps.workDir):
            os.mkdir(inps.workDir)

        print('-' * 50)
        return inps

    @staticmethod
    def load_slc_parser():

        TEMPLATE = """template:
        ########## 1. Load Data
        ## auto - automatic path pattern for Univ of Miami file structure
        ## load_slc.py -H to check more details and example inputs.
        ## compression to save disk usage for slcStack.h5 file:
        ## no   - save   0% disk usage, fast [default]
        ## lzf  - save ~57% disk usage, relative slow
        ## gzip - save ~62% disk usage, very slow [not recommend]
        
        miaplpy.load.processor      = auto  #[isce,snap,gamma,roipac], auto for isceTops
        miaplpy.load.updateMode     = auto  #[yes / no], auto for yes, skip re-loading if HDF5 files are complete
        miaplpy.load.compression    = auto  #[gzip / lzf / no], auto for no.
        miaplpy.load.autoPath       = auto    # [yes, no] auto for no
        
        miaplpy.load.slcFile        = auto  #[path2slc_file]
        miaplpy.load.startDate      = auto  #auto for first date
        miaplpy.load.endDate        = auto  #auto for last date
        ##---------for ISCE only:
        miaplpy.load.metaFile       = auto  #[path2metadata_file], i.e.: ./reference/IW1.xml, ./referenceShelve/data.dat
        miaplpy.load.baselineDir    = auto  #[path2baseline_dir], i.e.: ./baselines
        ##---------geometry datasets:
        miaplpy.load.demFile        = auto  #[path2hgt_file]
        miaplpy.load.lookupYFile    = auto  #[path2lat_file], not required for geocoded data
        miaplpy.load.lookupXFile    = auto  #[path2lon_file], not required for geocoded data
        miaplpy.load.incAngleFile   = auto  #[path2los_file], optional
        miaplpy.load.azAngleFile    = auto  #[path2los_file], optional
        miaplpy.load.shadowMaskFile = auto  #[path2shadow_file], optional
        miaplpy.load.waterMaskFile  = auto  #[path2water_mask_file], optional
        miaplpy.load.bperpFile      = auto  #[path2bperp_file], optional
        
        ##---------subset (optional):
        ## if both yx and lalo are specified, use lalo option unless a) no lookup file AND b) dataset is in radar coord
        miaplpy.subset.yx           = auto    #[y0:y1,x0:x1 / no], auto for no
        miaplpy.subset.lalo         = auto    #[S:N,W:E / no], auto for no
        """

        EXAMPLE = """example:
          load_slc.py -t PichinchaSenDT142.template
          load_slc.py -t miaplpyApp.cfg
          load_slc.py -t PichinchaSenDT142.template --project_dir $SCRATCH/PichinchaSenDT142
          load_slc.py -H #Show example input template for ISCE/ROI_PAC/GAMMA products
        """

        parser = argparse.ArgumentParser(description='Saving a stack of Interferograms to an HDF5 file',
                                         formatter_class=argparse.RawTextHelpFormatter,
                                         epilog=TEMPLATE + '\n' + EXAMPLE)
        parser.add_argument('-H', dest='print_example_template', action='store_true',
                            help='Print/Show the example template file for loading.')
        parser.add_argument('-t', '--template', type=str, nargs='+',
                            dest='template_file', help='Template file with path info.')

        parser.add_argument('-pj', '--project_dir', type=str, dest='project_dir',
                            help='Project directory of SLC dataset to read from')
        parser.add_argument('-d', '--work_dir', dest='work_dir', default=None,
                            help='Working directory of miaplpy (default ./miaplpy)')
        parser.add_argument('-pr', '--processor', type=str, dest='processor',
                            choices={'isce', 'gamma', 'roipac'},
                            help='InSAR processor/software of the file (This version only supports isce)',
                            default='isce')
        parser.add_argument('--enforce', '-f', dest='updateMode', action='store_false',
                            help='Disable the update mode, or skip checking dataset already loaded.')
        parser.add_argument('--compression', choices={'gzip', 'lzf', None}, default=None,
                            help='Compress loaded geometry while writing HDF5 file, default: None.')
        parser.add_argument('--no_metadata_check', dest='no_metadata_check', action='store_true',
                          help='Do not check for rsc files, when running via miaplpyApp.py')

        parser.add_argument('-o', '--output', type=str, nargs=3, dest='out_file',
                            default=['slcStack.h5',
                                     'geometryRadar.h5',
                                     'geometryGeo.h5'],
                            help='Output HDF5 file')
        return parser

    def phase_linking_parser(self):
        parser = self.parser
        patch = parser.add_argument_group('Phase inversion option')
        patch.add_argument('-w', '--work_dir', type=str, dest='work_dir', help='Working directory (miaplpy)')
        patch.add_argument('-r', '--range_window', type=int, dest='range_window', default=15,
                           help='Range window size for shp finding')
        patch.add_argument('-a', '--azimuth_window', type=int, dest='azimuth_window', default=15,
                           help='Azimuth window size for shp finding')
        patch.add_argument('-m', '--method', type=str, dest='inversion_method', default='EMI',
                           help='Inversion method (EMI, EVD, PTA, sequential_EMI, ...)')
        patch.add_argument('-l', '--time_lag', type=int, dest='time_lag', default=10,
                           help='Time lag in case StBAS is used')
        patch.add_argument('-t', '--test', type=str, dest='shp_test', default='ks',
                           help='Shp statistical test (ks, ad, ttest)')
        patch.add_argument('-psn', '--ps_num_shp', type=int, dest='ps_shp', default=10,
                           help='Number of SHPs for PS candidates')
        patch.add_argument('-p', '--patch_size', type=int, dest='patch_size', default=200,
                           help='Azimuth window size for shp finding')
        patch.add_argument('-mss', '--mini_stack_size', type=int, dest='ministack_size', default=10,
                           help='Number of images in each mini stack')
        patch.add_argument('-s', '--slc_stack', type=str, dest='slc_stack', help='SLC stack file')
        patch.add_argument('-ms', '--mask', type=str, dest='mask_file', default='None', help='mask file for inversion')
        patch.add_argument('-n', '--num_worker', dest='num_worker', type=int, default=1,
                           help='Number of parallel tasks (default: 1)')
        patch.add_argument('-i', '--index', dest='sub_index', type=str, default=None,
                           help='The list containing patches of i*num_worker:(i+1)*num_worker')
        patch.add_argument('-c', '--concatenate', dest='do_concatenate', action='store_false',
                           help='Concatenate all phase inverted patches')


        return parser

    @staticmethod
    def generate_unwrap_mask_parser():
        parser = argparse.ArgumentParser(description='Generate unwrap mask based on shadow mask and input custom mask')
        parser.add_argument('-g', '--geometry', type=str, dest='geometry_stack', required=True,
                            help='Geometry stack file with shadowMask in the datasets')
        parser.add_argument('-m', '--mask', type=str, dest='custom_mask', default=None,
                            help='Custom mask in HDF5 format')
        parser.add_argument('-o', '--output', type=str, dest='output_mask', default=None,
                            help='Output binary mask for unwrapping with snaphu')
        #parser.add_argument('-q', '--quality_type', type=str, dest='quality_type', default='full',
        #                    help='Temporal coherence type (full or average from mini-stacks)')
        parser.add_argument('-t', '--text_cmd', type=str, dest='text_cmd', default='',
                            help='Command before calling any script. exp: singularity run dockerimage.sif')

        return parser

    @staticmethod
    def generate_interferograms_parser():

        parser = argparse.ArgumentParser(description='Generate interferogram')
        parser.add_argument('-m', '--reference', type=str, dest='reference', required=True,
                            help='Reference image')
        parser.add_argument('-s', '--secondary', type=str, dest='secondary', required=True,
                            help='Secondary image')
        parser.add_argument('-t', '--stack', type=str, dest='stack_file', required=True,
                            help='Phase series stack file to read from')
        parser.add_argument('-o', '--output_dir', type=str, dest='out_dir', default='interferograms',
                            help='Prefix of output int and amp files')
        parser.add_argument('-a', '--azimuth_looks', type=int, dest='azlooks', default=1,
                            help='Azimuth looks')
        parser.add_argument('-r', '--range_looks', type=int, dest='rglooks', default=1,
                            help='Range looks')
        parser.add_argument('-f', '--filter_strength', type=float, dest='filter_strength', default=0.5,
                            help='filtering strength')
        parser.add_argument('-p', '--stack_prefix', dest='prefix', type=str, default='tops'
                            , help='ISCE stack processor: options= tops, stripmap -- default = tops')

        return parser


    @staticmethod
    def unwrap_parser():
        parser = argparse.ArgumentParser(description='Unwrap using snaphu')
        parser.add_argument('-f', '--ifg', dest='input_ifg', type=str, required=True,
                            help='Input wrapped interferogram')
        parser.add_argument('-c', '--coherence', dest='input_cor', type=str, required=True,
                            help='Input coherence file')
        parser.add_argument('-u', '--unwrapped_ifg', dest='unwrapped_ifg', type=str, required=True,
                            help='Output unwrapped interferogram')
        parser.add_argument('-m', '--mask', dest='unwrap_mask', type=str, default=None,
                            help='Output unwrapped interferogram')
        parser.add_argument('-sw', '--width', dest='ref_width', type=int, default=None,
                            help='Width of Reference .h5 file')
        parser.add_argument('-sl', '--length', dest='ref_length', type=int, default=None,
                            help='Length of .h5 file')
        parser.add_argument('-w', '--wavelength', dest='wavelength', type=str, default=None,
                            help='Wavelength')
        parser.add_argument('-ht', '--height', dest='height', type=str, default=None,
                            help='Altitude of satellite')
        parser.add_argument('-er', '--earth_radius', dest='earth_radius', type=str, default=None,
                            help='Earth Radius')
        parser.add_argument('-i', '--init_method', dest='init_method', type=str, default='MST',
                            help='Unwrap initialization algorithm (MST, MCF)')
        parser.add_argument('-d', '--max_discontinuity', dest='defo_max', type=float, default=1.2,
                            help='Maximum abrupt phase discontinuity (cycles)')
        parser.add_argument('-nt', '--num_tiles', dest='num_tiles', type=int, default=1,
                            help='Number of tiles for Unwrapping in parallel')
        parser.add_argument('--two-stage', dest='unwrap_2stage', action='store_true',
                            help='Use 2 stage unwrapping (from ISCE)')
        parser.add_argument('--rmfilter', dest='remove_filter_flag', action='store_true',
                            help='Remove filtering after unwrap')
        parser.add_argument('--tmp', dest='copy_to_tmp', action='store_true', help='Copy and process on tmp')

        return parser

    @staticmethod
    def generate_temporal_coherence_parser():
        parser = argparse.ArgumentParser(description='Convert phase to range time series')
        parser.add_argument('-d', '--work_dir', type=str, dest='work_dir', required=True,
                            help='Working directory (miaplpy)')
        parser.add_argument('--shadow_mask', dest='shadow_mask', action='store_true',
                            help='use shadow mask to mask final results')

        return parser

    @staticmethod
    def network_inversion_parser():
        parser = argparse.ArgumentParser(description='Convert phase to range time series')
        parser.add_argument('ifgramStackFile', help='interferograms stack file to be inverted')
        parser.add_argument('-d', '--work_dir', type=str, dest='work_dir', required=True,
                            help='Working directory (miaplpy)')
        parser.add_argument('-t', '--template', dest='template_file', type=str, default=None,
                            help='template file (default: smallbaselineApp.cfg)')
        parser.add_argument('--tcoh', '--temp_coh', dest='temp_coh', default=None,
                            help='use shadow mask to mask final results')
        parser.add_argument('--mask-thres', '--mask-threshold', '--mt', dest='maskThreshold', metavar='NUM', type=float,
                            default=0.5, help='threshold to generate mask for temporal coherence (default: %(default)s).')
        parser.add_argument('--shadow_mask', dest='shadow_mask', action='store_true',
                            help='use shadow mask to mask final results')
        parser.add_argument('--min-norm-phase', dest='minNormVelocity', action='store_false',
                            help=('Enable inversion with minimum-norm deformation phase,'
                                  ' instead of the default minimum-norm deformation velocity.'))
        parser.add_argument('--norm', dest='residualNorm', default='L1', choices=['L1', 'L2'],
                            help='Optimization mehtod, L1 or L2 norm. (default: %(default)s).')
        parser.add_argument('--smooth_factor', dest='L1_alpha', default=0.001,
                            help='Smoothing factor for L1 inversion [0-1] default: 0.01.')
        parser.add_argument('-w', '--weight-func', dest='weightFunc', default='var',
                            choices={'var', 'fim', 'coh', 'no'},
                            help='function used to convert coherence to weight for inversion:\n' +
                                 'var - inverse of phase variance due to temporal decorrelation (default)\n' +
                                 'fim - Fisher Information Matrix as weight' +
                                 'coh - spatial coherence\n' +
                                 'no  - no/uniform weight')

        return parser

    @staticmethod
    def miaplpy_app_parser():

        STEP_LIST = [
            'load_data',
            'phase_linking',
            'concatenate_patches',
            'generate_ifgram',
            'unwrap_ifgram',
            'load_ifgram',
            'ifgram_correction',
            'invert_network',
            'timeseries_correction']


        STEP_HELP = """Command line options for steps processing with names are chosen from the following list:
        {}
        {}

        In order to use either --start or --step, it is necessary that a
        previous run was done using one of the steps options to process at least
        through the step immediately preceding the starting step of the current run.
        """.format(STEP_LIST[0:4], STEP_LIST[4::])

        EXAMPLE = """example: 
              miaplpyApp.py  <custom_template_file>            # run with default and custom templates
              miaplpyApp.py  -h / --help                       # help 
              miaplpyApp.py  -H                                # print    default template options
              # Run with --start/stop/step options
              miaplpyApp.py PichinchaSenDT142.template --dostep  load_data       # run the step 'download' only
              miaplpyApp.py PichinchaSenDT142.template --start load_data         # start from the step 'download' 
              miaplpyApp.py PichinchaSenDT142.template --stop  unwrap_ifgram    # end after step 'interferogram'
              """
        parser = argparse.ArgumentParser(description='Routine Time Series Analysis for MiaplPy',
                                         formatter_class=argparse.RawTextHelpFormatter,
                                         epilog=EXAMPLE)

        parser.add_argument('customTemplateFile', nargs='?',
                            help='Custom template with option settings.\n' +
                                 "ignored if the default miaplpyApp.cfg is input.")
        parser.add_argument('--dir', '--work-dir', dest='workDir', default=None,
                            help='Work directory, (default: %(default)s).')

        parser.add_argument('-g', dest='generate_template', action='store_true',
                            help='Generate default template (if it does not exist) and exit.')
        parser.add_argument('-H', dest='print_template', action='store_true',
                            help='Print the default template file and exit.')
        parser.add_argument('-v', '--version', action='store_true', help='Print software version and exit')

        parser.add_argument('--walltime', dest='wall_time', default=None,
                             help='walltime for submitting the script as a job')
        parser.add_argument('--queue', dest='queue', default=None, help='Queue name')
        parser.add_argument('--jobfiles', dest='write_job', action='store_true',
                          help='Do not run the tasks, only write job files')
        parser.add_argument('--runfiles', dest='run_flag', action='store_true', help='Create run files for all steps')
        parser.add_argument('--tmp', dest='copy_to_tmp', action='store_true', help='Copy and process on tmp')

        step = parser.add_argument_group('steps processing (start/end/dostep)', STEP_HELP)
        step.add_argument('--start', dest='startStep', metavar='STEP', default=STEP_LIST[0],
                          help='Start processing at the named step, default: {}'.format(STEP_LIST[0]))
        step.add_argument('--end', '--stop', dest='endStep', metavar='STEP', default=STEP_LIST[-1],
                          help='End processing at the named step, default: {}'.format(STEP_LIST[-1]))
        step.add_argument('--dostep', dest='doStep', metavar='STEP',
                          help='Run processing at the named step only')


        return parser, STEP_LIST, EXAMPLE







