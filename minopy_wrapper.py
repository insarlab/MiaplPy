#!/usr/bin/env python3
########################
# Author: Sara Mirzaee
#######################
import os
import sys
import time
import datetime
import shutil
import numpy as np
import gdal
from mintpy.utils import readfile, utils as ut
from mintpy.smallbaselineApp import TimeSeriesAnalysis
from mintpy.objects import ifgramStack
from mintpy.ifgram_inversion import write2hdf5_file, read_unwrap_phase, mask_unwrap_phase
import minopy
import minopy.submit_jobs as js
from minopy_utilities import log_message
from minopy.objects.arg_parser import MinoPyParser
from minopy.objects.slcStack import slcStack
from minopy.objects.stack_int import MinopyRun
from minopy.defaults.auto_path import autoPath, PathFind

pathObj = PathFind()
###########################################################################################
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
    'email',]

supported_schedulers = ['LSF', 'PBS', 'SLURM']


##########################################################################
def main(iargs=None):
    start_time = time.time()
    Parser = MinoPyParser(iargs, script='minopy_wrapper')
    inps = Parser.parse()

    job_file_name = 'minopy_wrapper'
    job_name = job_file_name

    if inps.wall_time == 'None':
        inps.wall_time = '12:00'

    #########################################
    # Submit job
    #########################################

    if inps.submit_flag:
        js.submit_script(job_name, job_file_name, sys.argv[:], inps.workDir, new_wall_time)
        sys.exit(0)

    if not iargs is None:
        log_message(inps.workDir, os.path.basename(__file__) + ' ' + ' '.join(iargs[:]))
    else:
        log_message(inps.workDir, os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1::]))

    os.chdir(inps.workDir)

    app = minopyTimeSeriesAnalysis(inps.customTemplateFile, inps.workDir, inps)
    app.startup()
    if len(inps.runSteps) > 0:
        app.run(steps=inps.runSteps, plot=inps.plot)

    # Timing
    m, s = divmod(time.time()-start_time, 60)
    print('Time used: {:02.0f} mins {:02.1f} secs\n'.format(m, s))
    return


class minopyTimeSeriesAnalysis(TimeSeriesAnalysis):
    """ Routine processing workflow for time series analysis of InSAR stacks with MiNoPy
        """

    def __init__(self, customTemplateFile=None, workDir=None, inps=None):
        super().__init__(customTemplateFile, workDir)
        self.inps = inps

    def startup(self):

        self.custom_template_file = custom_template_file
        self.cwd = os.path.abspath(os.getcwd())
        self.workDir = workDir

        # 1. Get project name
        self.project_name = None
        if self.custom_template_file:
            self.project_name = os.path.splitext(os.path.basename(self.custom_template_file))[0]
            print('Project name:', self.project_name)

        # 2. Go to the work directory
        # 2.1 Get workDir
        if not self.workDir:
            if autoPath and 'SCRATCHDIR' in os.environ and self.project_name:
                self.workDir = os.path.join(os.getenv('SCRATCHDIR'), self.projectName, pathObj.minopydir)
            else:
                self.workDir = os.getcwd()
        self.workDir = os.path.abspath(self.workDir)

        # 2.2 Go to workDir
        if not os.path.isdir(self.workDir):
            os.makedirs(self.workDir)
            print('create directory:', self.workDir)
        os.chdir(self.workDir)
        print("Go to work directory:", self.workDir)

        # 3. Read templates
        # 3.1 Get default template file
        lfile = os.path.join(os.path.dirname(__file__), 'defaults/minopy_template.cfg')  # latest version
        cfile = os.path.join(self.workDir, 'minopy_template.cfg')  # current version
        if not os.path.isfile(cfile):
            print('copy default template file {} to work directory'.format(lfile))
            shutil.copy2(lfile, self.workDir)
        else:
            # cfile is obsolete if any key is missing
            ldict = readfile.read_template(lfile)
            cdict = readfile.read_template(cfile)
            if any([key not in cdict.keys() for key in ldict.keys()]):
                print('obsolete default template detected, update to the latest version.')
                shutil.copy2(lfile, self.workDir)
                # keep the existing option value from obsolete template file
                ut.update_template_file(cfile, cdict)
        self.templateFile = cfile

        self.run_dir = os.path.join(self.workDir, pathObj.rundir)
        self.patch_dir = os.path.join(self.workDir, pathObj.patchdir)
        self.ifgram_dir = os.path.join(self.workDir, pathObj.intdir)

        for directory in [self.run_dir, self.patch_dir, self.ifgram_dir]:
            if not os.path.isdir(directory):
                os.mkdir(directory)

        # 3.2 read (custom) template files into dicts
        super()._read_template()
        auto_template_file = os.path.join(os.path.dirname(__file__), 'defaults/minopy_template_defaults.cfg')
        self.template = ut.check_template_auto_value(self.template, auto_file=auto_template_file)

        # 4. Copy the plot shell file
        sh_file = os.path.join(os.getenv('MINTPY_HOME'), 'sh/plot_smallbaselineApp.sh')

        def grab_latest_update_date(fname, prefix='# Latest update:'):
            try:
                lines = open(fname, 'r').readlines()
                line = [i for i in lines if prefix in i][0]
                t = re.findall('\d{4}-\d{2}-\d{2}', line)[0]
                t = datetime.datetime.strptime(t, '%Y-%m-%d')
            except:
                t = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d')  # a arbitrary old date
            return t

        # 1) copy to work directory (if not existed yet).
        if not os.path.isfile(os.path.basename(sh_file)):
            print('copy {} to work directory: {}'.format(sh_file, self.workDir))
            shutil.copy2(sh_file, self.workDir)

        # 2) copy to work directory (if obsolete file detected) and rename the existing one
        elif grab_latest_update_date(os.path.basename(sh_file)) < grab_latest_update_date(sh_file):
            os.system('mv {f} {f}_obsolete'.format(f=os.path.basename(sh_file)))
            print('obsolete shell file detected, renamed it to: {}_obsolete'.format(os.path.basename(sh_file)))
            print('copy {} to work directory: {}'.format(sh_file, self.workDir))
            shutil.copy2(sh_file, self.workDir)

        self.plot_sh_cmd = './' + os.path.basename(sh_file)

        return

    def run_crop(self, sname):
        """ Cropping images using crop_sentinel.py script.
        """

        os.chdir(self.workDir)

        if self.template['mintpy.susbet.lalo'] == 'None' and self.template['mintpy.susbet.yx'] == 'None':
            print('WARNING: No crop area given in minopy.subset, the whole image is going to be used.')
            print('WARNING: May take days to process!')
        else:
            scp_args = '--template {}'.format(self.templateFile)
            print('crop_images.py ', scp_args)
            minopy.crop_images.main(scp_args.split())
        return

    def run_create_patch(self, sname):
        """ Dividing the area into patches.
        """
        scp_args = '--workDir {} --rangeWin {} --azimuthWin {} --patchSize {}'.\
            format(self.workDir, self.patch_dir,
                   int(self.template['minopy.range_window']),
                   int(self.template['minopy.azimuth_window']),
                   int(self.template['minopy.patch_size']))
        print('create_patch.py ', scp_args)
        minopy.create_patch.main(scp_args.split())

        return

    def run_phase_linking(self):
        """ Non-Linear phase inversion.
        """
        if not os.path.exists(self.run_dir):
            os.mkdir(self.run_dir)

        patch_list = glob.glob(self.patch_dir + '/patch*')
        run_minopy_inversion = os.path.join(self.run_dir, 'run_minopy_inversion')
        with open(run_minopy_inversion, 'w') as f:
            for item in patch_list:
                scp_srgs = '-w {a0} -r {a1} -a {a2} -m {a3} -t {a4} -p {a5}'.format(a0=self.workDir,
                                                                                     a1=self.template['minopy.range_window'],
                                                                                     a2=self.template['minopy.azimuth_window'],
                                                                                     a3=self.template['minopy.plmethod'],
                                                                                     a4=self.template['minopy.shp_test'],
                                                                                     a5=item.split('/')[-1])
                command = 'python_inversion.py ' + scp_srgs
                f.write(command)

        memorymax = '2000'
        walltime = '2:00'
        js.scheduler_job_submit(run_minopy_inversion, self.workDir, memorymax, walltime)

        return

    def get_phase_linking_coherence_mask(self):
        """Generate reliable pixel mask from temporal coherence"""

        tcoh_file = os.path.join(self.workDir, 'temporalCoherence.h5')
        mask_file = os.path.join(self.workDir, 'maskTempCoh.h5')

        tcoh_min = self.template['mintpy.networkInversion.minTempCoh']

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
        mask_threshold = self.template['mintpy.networkInversion.minTempCoh']

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

        return

    def run_interferogram(self):
        """ Export single master interferograms
        """
        run_file_int = os.path.join(self.run_dir, 'run_interferograms')

        if not os.path.exists(run_file_int):

            slc_file = os.path.join(self.workDir, 'inputs/slcStack.h5')
            slcObj = slcStack(slc_file)
            slcObj.open(print_msg=False)
            date_list = slcObj.get_date_list(dropIfgram=True)

            pairs = []
            for i in range(1, len(date_list)):
                pairs.append((date_list[0], date_list[i]))

            runObj = MinopyRun()
            runObj.configure(self.inps, 'run_interferograms')
            runObj.generateIfg(self.inps, pairs)
            runObj.finalize()

        memorymax = '4000'
        walltime = '2:00'

        js.scheduler_job_submit(run_file_int, self.workDir, memorymax, walltime)

        return

    def run_unwrap(self, config):
        """ Unwrapps single master interferograms
        """
        run_file_unwrap = os.path.join(self.run_dir, 'run_unwrap')

        if not os.path.exists(run_file_unwrap):

            slc_file = os.path.join(self.workDir, 'inputs/slcStack.h5')
            slcObj = slcStack(slc_file)
            slcObj.open(print_msg=False)
            date_list = slcObj.get_date_list(dropIfgram=True)

            pairs = []
            for i in range(1, len(date_list)):
                pairs.append((date_list[0], date_list[i]))

            runObj = MinopyRun()
            runObj.configure(self.inps, 'run_unwrap')
            runObj.generateIfg(self.inps, pairs)
            runObj.finalize()

        memorymax = '5000'
        walltime = '4:00'

        js.scheduler_job_submit(run_file_unwrap, self.workDir, memorymax, walltime)

        return

    def run_email_results(self):
        """ email Time series results
        """
        log_message(self.workDir, 'email_results.py {}'.format(self.custom_template_file))
        email_results.main([self.custom_template_file])
        return

    def run(self, steps=STEP_LIST, plot=True):
        for sname in steps:
            print('\n\n******************** step - {} ********************'.format(sname))

            if sname == 'crop':
                self.run_crop(sname)

            elif sname == 'patch':
                self.run_create_patch(sname)

            elif sname == 'inversion':
                self.run_phase_linking(sname)

            elif sname == 'ifgrams':
                self.run_interferogram(sname)

            elif sname == 'unwrap':
                self.run_unwrap(sname)

            elif sname == 'load_data':
                super().run_load_data(sname)

            elif sname == 'modify_network':
                super().run_network_modification(sname)

            elif sname == 'reference_point':
                super().run_reference_point(sname)

            elif sname == 'write_to_timeseries':
                self.write_to_timeseries(sname)

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

            elif sname == 'email':
                self.run_email_results(sname)


        # plot result (show aux visualization message more multiple steps processing)
        print_aux = len(steps) > 1
        super().plot_result(print_aux=print_aux, plot=plot)

        # go back to original directory
        print('Go back to directory:', self.cwd)
        os.chdir(self.cwd)

        # message
        msg = '\n###############################################################'
        msg += '\nNormal end of Non-Linear time series processing workflow!'
        msg += '\n##############################################################'
        print(msg)
        return


###########################################################################################
if __name__ == '__main__':
    main()
