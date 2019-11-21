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
import warnings
import glob
import h5py
from scipy import linalg
import mintpy
from mintpy.utils import writefile, readfile, utils as ut
from mintpy.smallbaselineApp import TimeSeriesAnalysis
from mintpy.objects import timeseries, ifgramStack
from mintpy.ifgram_inversion import read_unwrap_phase, mask_unwrap_phase, split2boxes, write2hdf5_auxFiles
import minopy
import minopy.workflow
import minopy.submit_jobs as js
from minopy_utilities import log_message, email_minopy
from minopy.objects.arg_parser import MinoPyParser
from minopy.objects.slcStack import slcStack
from minopy.objects.stack_int import MinopyRun
from minopy.defaults.auto_path import autoPath, PathFind
from minopy.objects.utils import check_template_auto_value
from minopy.defaults import auto_path

pathObj = PathFind()
###########################################################################################
STEP_LIST = [
    'crop',
    'create_patch',
    'inversion',
    'ifgrams',
    'unwrap',
    'load_int',
    'modify_network',
    'reference_point',
    'correct_unwrap_error',
    'stack_interferograms',
    'write_to_timeseries',
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
    'plot',
    'email',]

##########################################################################


def main(iargs=None):
    start_time = time.time()
    Parser = MinoPyParser(iargs, script='minopy_wrapper')
    inps = Parser.parse()

    job_file_name = 'minopy_wrapper'
    job_name = job_file_name

    if inps.wall_time == 'None':
        inps.wall_time = '24:00'

    #########################################
    # Submit job
    #########################################

    if inps.submit_flag:
        js.submit_script(job_name, job_file_name, sys.argv[:], inps.workDir, inps.wall_time, queue_name=inps.queue_name)
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

        self.customTemplateFile = customTemplateFile
        self.cwd = os.path.abspath(os.getcwd())
        self.workDir = workDir

    def startup(self):

        # 1. Get project name
        self.project_name = None
        if self.customTemplateFile and not os.path.basename(self.customTemplateFile) == 'minopy_template.cfg':
            self.project_name = os.path.splitext(os.path.basename(self.customTemplateFile))[0]
            print('Project name:', self.project_name)
        else:
            self.project_name = os.path.dirname(self.workDir)

        # 2. Go to the work directory
        # 2.1 Get workDir
        if not self.workDir:
            if autoPath and 'SCRATCHDIR' in os.environ and self.project_name:
                self.workDir = os.path.join(os.getenv('SCRATCHDIR'), self.project_name, pathObj.minopydir)
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
        self._read_template()

        if (auto_path.autoPath
                and 'SCRATCHDIR' in os.environ
                and self.project_name is not None
                and self.template['mintpy.load.slcFile'] == 'auto'):
            print(('check auto path setting for Univ of Miami users'
                   ' for processor: {}'.format(self.template['mintpy.load.processor'])))
            self.template = auto_path.get_auto_path(processor=self.template['mintpy.load.processor'],
                                                    project_name=self.project_name,
                                                    template=self.template)

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

    def _read_template(self):
        # read custom template, to:
        # 1) update default template
        # 2) add metadata to ifgramStack file and HDF-EOS5 file
        self.customTemplate = None
        if self.customTemplateFile:
            cfile = self.customTemplateFile
            # Copy custom template file to inputs directory for backup
            inputs_dir = os.path.join(self.workDir, 'inputs')
            if not os.path.isdir(inputs_dir):
                os.makedirs(inputs_dir)
                print('create directory:', inputs_dir)
            if ut.run_or_skip(out_file=os.path.join(inputs_dir, os.path.basename(cfile)),
                              in_file=cfile,
                              check_readable=False) == 'run':
                shutil.copy2(cfile, inputs_dir)
                print('copy {} to inputs directory for backup.'.format(os.path.basename(cfile)))

            # Read custom template
            print('read custom template file:', cfile)
            cdict = readfile.read_template(cfile)

            # correct some loose type errors
            standardValues = {'def':'auto', 'default':'auto',
                              'y':'yes', 'on':'yes', 'true':'yes',
                              'n':'no', 'off':'no', 'false':'no'
                             }
            for key, value in cdict.items():
                if value in standardValues.keys():
                    cdict[key] = standardValues[value]

            for key in ['mintpy.deramp', 'mintpy.troposphericDelay.method']:
                if key in cdict.keys():
                    cdict[key] = cdict[key].lower().replace('-', '_')

            if 'processor' in cdict.keys():
                cdict['mintpy.load.processor'] = cdict['processor']

            # these metadata are used in load_data.py only, not needed afterwards
            # (in order to manually add extra offset when the lookup table is shifted)
            # (seen in ROI_PAC product sometimes)
            for key in ['SUBSET_XMIN', 'SUBSET_YMIN']:
                if key in cdict.keys():
                    cdict.pop(key)

            self.customTemplate = dict(cdict)

            # Update default template file based on custom template
            print('update default template based on input custom template')
            self.templateFile = ut.update_template_file(self.templateFile, self.customTemplate)

        print('read default template file:', self.templateFile)
        self.template = readfile.read_template(self.templateFile)
        auto_template_file = os.path.join(os.path.dirname(__file__), 'defaults/minopy_template_defaults.cfg')
        self.template = check_template_auto_value(self.template, auto_file=auto_template_file)

        # correct some loose setup conflicts
        if self.template['mintpy.geocode'] is False:
            for key in ['mintpy.save.hdfEos5', 'mintpy.save.kmz']:
                if self.template[key] is True:
                    self.template['mintpy.geocode'] = True
                    print('Turn ON mintpy.geocode in order to run {}.'.format(key))
                    break
        return

    def run_crop(self, sname):
        """ Cropping images using crop_sentinel.py script.
        """

        os.chdir(self.workDir)

        if self.template['mintpy.subset.lalo'] == 'None' and self.template['mintpy.subset.yx'] == 'None':
            print('WARNING: No crop area given in minopy.subset, the whole image is going to be used.')
            print('WARNING: May take days to process!')
        else:
            scp_args = '--template {}'.format(self.templateFile)
            if self.customTemplateFile:
                scp_args += ' {}'.format(self.customTemplateFile)
            if self.project_name:
                scp_args += ' --project {}'.format(self.project_name)

            print('crop_images.py ', scp_args)
            minopy.crop_images.main(scp_args.split())
        return

    def run_create_patch(self, sname):
        """ Dividing the area into patches.
        """

        scp_args = '--workDir {} --rangeWin {} --azimuthWin {} --patchSize {}'.\
            format(self.workDir,
                   int(self.template['mintpy.inversion.range_window']),
                   int(self.template['mintpy.inversion.azimuth_window']),
                   int(self.template['mintpy.inversion.patch_size']))
        print('create_patch.py ', scp_args)
        minopy.create_patch.main(scp_args.split())

        return

    def run_patch_inversion(self, sname):
        """ Non-Linear phase inversion.
        """
        if not os.path.exists(self.run_dir):
            os.mkdir(self.run_dir)
        patch_list = glob.glob(self.patch_dir + '/patch*')
        run_minopy_inversion = os.path.join(self.run_dir, 'run_minopy_inversion')

        with open(run_minopy_inversion, 'w') as f:
            for item in patch_list:
                scp_srgs = '-w {a0} -r {a1} -a {a2} -m {a3} -t {a4} -p {a5}\n'.format(
                    a0=self.workDir,
                    a1=self.template['mintpy.inversion.range_window'],
                    a2=self.template['mintpy.inversion.azimuth_window'],
                    a3=self.template['mintpy.inversion.plmethod'],
                    a4=self.template['mintpy.inversion.shp_test'],
                    a5=item.split('/')[-1])
                command = 'patch_inversion.py ' + scp_srgs
                f.write(command)

        memorymax = '2000'
        walltime = '6:00'
        js.scheduler_job_submit(run_minopy_inversion, self.workDir, memorymax, walltime, queuename=self.inps.queue_name)

        return

    def run_interferogram(self, sname):
        """ Export single master interferograms
        """
        run_file_int = os.path.join(self.run_dir, 'run_interferograms')

        slc_file = os.path.join(self.workDir, 'inputs/slcStack.h5')
        slcObj = slcStack(slc_file)
        slcObj.open(print_msg=False)
        date_list = slcObj.get_date_list()

        if self.template['mintpy.interferograms.type'] == 'sequential':
            master_ind = True
        else:
            master_ind = False

        pairs = []
        for i in range(1, len(date_list)):
            if master_ind:
                pairs.append((date_list[i - 1], date_list[i]))
            else:
                pairs.append((date_list[0], date_list[i]))

        inps = self.inps
        inps.run_dir = self.run_dir
        inps.patch_dir = self.patch_dir
        inps.ifgram_dir = self.ifgram_dir
        inps.template = self.template

        runObj = MinopyRun()
        runObj.configure(inps, 'run_interferograms')
        runObj.generateIfg(inps, pairs)
        runObj.finalize()

        os.system('chmod +x {}'.format(self.workDir+'/configs/*'))

        del runObj, slcObj

        memorymax = '4000'
        walltime = '2:00'

        js.scheduler_job_submit(run_file_int, self.workDir, memorymax, walltime, queuename=self.inps.queue_name)

        return

    def run_unwrap(self, sname):
        """ Unwrapps single master interferograms
        """
        run_file_unwrap = os.path.join(self.run_dir, 'run_unwrap')

        slc_file = os.path.join(self.workDir, 'inputs/slcStack.h5')
        slcObj = slcStack(slc_file)
        slcObj.open(print_msg=False)
        date_list = slcObj.get_date_list()

        if self.template['mintpy.interferograms.type'] == 'sequential':
            master_ind = True
        else:
            master_ind = False

        pairs = []
        for i in range(1, len(date_list)):
            if master_ind:
                pairs.append((date_list[i - 1], date_list[i]))
            else:
                pairs.append((date_list[0], date_list[i]))

        inps = self.inps
        inps.run_dir = self.run_dir
        inps.patch_dir = self.patch_dir
        inps.ifgram_dir = self.ifgram_dir
        inps.template = self.template

        runObj = MinopyRun()
        runObj.configure(inps, 'run_unwrap')
        runObj.unwrap(inps, pairs)
        runObj.finalize()

        os.system('chmod +x {}'.format(self.workDir + '/configs/*'))
        
        memorymax = '5000'
        walltime = '4:00'

        js.scheduler_job_submit(run_file_unwrap, self.workDir, memorymax, walltime, queuename=self.inps.queue_name)

        return

    def run_load_int(self, step_name):
        """Load InSAR stacks into HDF5 files in ./inputs folder.
        It 1) copy auxiliary files into work directory (for Unvi of Miami only)
           2) load all interferograms stack files into mintpy/inputs directory.
           3) check loading result
           4) add custom metadata (optional, for HDF-EOS5 format only)
        """
        os.chdir(self.workDir)

        # 1) copy aux files (optional)
        self.projectName = self.project_name
        super()._copy_aux_file()

        # 2) loading data
        scp_args = '--template {}'.format(self.templateFile)
        if self.customTemplateFile:
            scp_args += ' {}'.format(self.customTemplateFile)
        if self.projectName:
            scp_args += ' --project {}'.format(self.projectName)
        scp_args += ' --output {}'.format(self.workDir + '/inputs/ifgramStack.h5')
        # run
        print("load_int.py", scp_args)
        minopy.load_int.main(scp_args.split())

        # 3) check loading result
        load_complete, stack_file, geom_file = ut.check_loaded_dataset(self.workDir, print_msg=True)[0:3]

        # 4) add custom metadata (optional)
        if self.customTemplateFile:
            print('updating {}, {} metadata based on custom template file: {}'.format(
                os.path.basename(stack_file),
                os.path.basename(geom_file),
                os.path.basename(self.customTemplateFile)))
            # use ut.add_attribute() instead of add_attribute.py because of
            # better control of special metadata, such as SUBSET_X/YMIN
            ut.add_attribute(stack_file, self.customTemplate)
            ut.add_attribute(geom_file, self.customTemplate)

        # 5) if not load_complete, plot and raise exception
        if not load_complete:
            # plot result if error occured
            self.plot_result(print_aux=False, plot=plot)

            # go back to original directory
            print('Go back to directory:', self.cwd)
            os.chdir(self.cwd)

            # raise error
            msg = 'step {}: NOT all required dataset found, exit.'.format(step_name)
            raise RuntimeError(msg)
        return

    def get_phase_linking_coherence_mask(self):
        """Generate reliable pixel mask from temporal coherence"""

        tcoh_file = os.path.join(self.workDir, 'temporalCoherence.h5')
        mask_file = os.path.join(self.workDir, 'maskTempCoh.h5')

        tcoh_min = float(self.template['mintpy.networkInversion.minTempCoh'])

        scp_args = '{} -m {} --nonzero -o {} --update'.format(tcoh_file, tcoh_min, mask_file)
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

    def write_to_timeseries(self, sname):

        inps = self.inps

        inps.timeseriesFile = os.path.join(self.workDir, 'timeseries.h5')
        inps.tempCohFile = os.path.join(self.workDir, 'temporalCoherence.h5')
        inps.timeseriesFiles = [os.path.join(self.workDir, 'timeseries.h5')]  # all ts files
        inps.outfile = [os.path.join(self.workDir, 'timeseries.h5'),
                        os.path.join(self.workDir, 'temporalCoherence.h5')]

        ifgram_file = os.path.join(self.workDir, 'inputs/ifgramStack.h5')

        stack_obj = ifgramStack(ifgram_file)
        stack_obj.open(print_msg=False)

        date12_list = stack_obj.get_date12_list(dropIfgram=True)
        A = stack_obj.get_design_matrix4timeseries(date12_list=date12_list)[0]

        pbase = stack_obj.get_perp_baseline_timeseries(dropIfgram=False)
        date_list = stack_obj.get_date_list(dropIfgram=False)
        length, width = stack_obj.length, stack_obj.width
        num_date = len(date_list)
        num_ifgram = num_date - 1

        box_list = split2boxes(dataset_shape=stack_obj.get_size(), chunk_size=100e6)
        num_box = len(box_list)

        unwDatasetName = [i for i in ['unwrapPhase_bridging', 'unwrapPhase'] if i in stack_obj.datasetNames][0]
        mask_dataset_name = self.template['mintpy.networkInversion.maskDataset']
        mask_threshold = float(self.template['mintpy.networkInversion.maskThreshold'])

        ref_phase = stack_obj.get_reference_phase(unwDatasetName=unwDatasetName,
                                                  skip_reference=True,
                                                  dropIfgram=False)

        # File 1 - timeseries.h5

        metadata = dict(stack_obj.metadata)
        metadata['REF_DATE'] = date_list[0]
        metadata['FILE_TYPE'] = 'timeseries'
        metadata['UNIT'] = 'm'

        suffix = ''
        ts_file = '{}{}.h5'.format(suffix, os.path.splitext(inps.outfile[0])[0])
        ts_obj = timeseries(ts_file)

        # A dictionary of the datasets which we like to have in the timeseries
        dsNameDict = {
            "date": (np.dtype('S8'), (num_date,)),
            "bperp": (np.float32, (num_date,)),
            "timeseries": (np.float32, (num_date, length, width)),
        }

        # layout the HDF5 file for the datasets and the metadata
        ts_obj.layout_hdf5(dsNameDict, metadata)

        phase2range = -1 * float(metadata['WAVELENGTH']) / (4. * np.pi)

        gfilename = os.path.join(self.workDir, 'avgSpatialCoh.h5')
        f = h5py.File(gfilename, 'r')
        spatial_coh = f['coherence'][:, :]
        f.close()

        gfilename = os.path.join(self.workDir, 'inputs/geometryRadar.h5')
        f = h5py.File(gfilename, 'r')
        temp_coh = f['quality'][:, :]
        f.close()
        mask_spCoh = (spatial_coh >= mask_threshold)
        temp_coh *= mask_spCoh

        threshold_tempCoh = float(self.template['mintpy.networkInversion.minTempCoh'])

        for i in range(num_box):
            box = box_list[i]
            num_row = box[3] - box[1]
            num_col = box[2] - box[0]
            num_pixel = num_row * num_col

            mask_Coh = (temp_coh[box[1]:box[3], box[0]:box[2]] >= threshold_tempCoh).reshape(num_pixel,)

            if num_box > 1:
                print('\n------- Processing Patch {} out of {} --------------'.format(i + 1, num_box))

                # Read/Mask unwrapPhase
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

                # Mask for pixels to invert
                mask = np.ones(num_pixel, np.bool_)

                # Mask for Zero Phase in ALL ifgrams
                print('skip pixels with zero/nan value in all interferograms')
                with warnings.catch_warnings():
                    # ignore warning message for all-NaN slices
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    phase_stack = np.nanmean(pha_data, axis=0)
                mask *= np.multiply(~np.isnan(phase_stack), phase_stack != 0.)
                del phase_stack

                ts = np.zeros((num_date, num_pixel), np.float32)
                num_pixel2inv = int(np.sum(mask))
                idx_pixel2inv = np.where(mask)[0]

                if num_pixel2inv < 1:
                    ts = ts.reshape(num_date, num_row, num_col)
                else:

                    # Mask for Non-Zero Phase in ALL ifgrams (share one B in sbas inversion)
                    mask_all_net = np.all(pha_data, axis=0)
                    mask_all_net *= mask
                    mask_all_net *= mask_Coh
                    idx_pixel2inv = np.where(mask_all_net)[0]

                    if np.sum(mask_all_net) > 0:
                        tsi = linalg.lstsq(A, pha_data[:, mask_all_net], cond=1e-5)[0]

            ts[1:, idx_pixel2inv] = tsi
            ts = ts.reshape(num_date, num_row, num_col)

            print('converting phase to range')
            ts *= phase2range
            block = [0, num_date, box[1], box[3], box[0], box[2]]
            ts_obj.write2hdf5_block(ts, datasetName='timeseries', block=block)

        print('-' * 50)
        date_list_utf8 = [dt.encode('utf-8') for dt in date_list]
        ts_obj.write2hdf5_block(date_list_utf8, datasetName='date')
        ts_obj.write2hdf5_block(pbase, datasetName='bperp')

        # reference pixel
        ref_y = int(stack_obj.metadata['REF_Y'])
        ref_x = int(stack_obj.metadata['REF_X'])
        temp_coh[ref_y, ref_x] = 1.

        num_inv_ifg = np.zeros((length, width), np.int16) + num_ifgram
        write2hdf5_auxFiles(metadata, temp_coh, num_inv_ifg, suffix='', inps=inps)

        self.get_phase_linking_coherence_mask()

        return

    def run(self, steps=STEP_LIST, plot=True):
        for sname in steps:
            print('\n\n******************** step - {} ********************'.format(sname))

            if sname == 'crop':
                self.run_crop(sname)

            elif sname == 'create_patch':
                self.run_create_patch(sname)

            elif sname == 'inversion':
                self.run_patch_inversion(sname)

            elif sname == 'ifgrams':
                self.run_interferogram(sname)

            elif sname == 'unwrap':
                self.run_unwrap(sname)

            elif sname == 'load_int':
                self.run_load_int(sname)

            elif sname == 'modify_network':
                super().run_network_modification(sname)

            elif sname == 'reference_point':

                ifgram_file = os.path.join(self.workDir, 'inputs/ifgramStack.h5')
                with h5py.File(ifgram_file, 'a') as f:
                    f.attrs['mintpy.reference.yx'] = self.template['mintpy.reference.yx']
                    f.attrs['mintpy.reference.lalo'] = self.template['mintpy.reference.lalo']
                f.close()

                super().run_reference_point(sname)

            elif sname == 'correct_unwrap_error':

                if self.template['mintpy.unwrapError.method']:
                    self.template['mintpy.unwrapError.method'] = 'bridging'
                ifgram_file = os.path.join(self.workDir, 'inputs/ifgramStack.h5')
                f = h5py.File(ifgram_file, 'a')
                if 'unwrapPhase_bridging' in f.keys():
                    del f['unwrapPhase_bridging']
                f.close()

                super().run_unwrap_error_correction(sname)

            elif sname == 'stack_interferograms':
                super().run_ifgram_stacking(sname)

            elif sname == 'write_to_timeseries':
                self.write_to_timeseries(sname)

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

            elif sname == 'plot':
                # plot result (show aux visualization message more multiple steps processing)
                print_aux = len(steps) > 1
                super().plot_result(print_aux=print_aux, plot=plot)

            #elif sname == 'email':
            #    email_minopy(self.workDir)

        # go back to original directory
        print('Go back to directory:', self.cwd)
        os.chdir(self.cwd)

        # message
        msg = '\n###############################################################'
        msg += '\nNormal end of Non-Linear time series processing workflow!'
        msg += '\n##############################################################'
        print(msg)
        return

'''
def mask_unwrap_phase(pha_data, msk_data, mask_threshold=0.5):
    # Read/Generate Mask
    msk_data[np.isnan(msk_data)] = 0
    msk_data = 1 * (msk_data >= float(mask_threshold))
    for ind in range(pha_data.shape[0]):
        pha_data[ind, :] = msk_data * pha_data[ind, :]
    return pha_data
'''
###########################################################################################


if __name__ == '__main__':
    main()
