#!/usr/bin/env python3
########################
# Author: Sara Mirzaee
#######################
import logging
import warnings

warnings.filterwarnings("ignore")

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import os
import sys
import time
import shutil
import math

from mintpy.utils import writefile, readfile, utils as ut
from mintpy.smallbaselineApp import TimeSeriesAnalysis
from minopy.objects.arg_parser import MinoPyParser
from minopy.defaults.auto_path import autoPath, PathFind
from minopy.find_short_baselines import find_baselines, plot_baselines
from minopy.objects.utils import (check_template_auto_value,
                                  log_message, get_latest_template_minopy,
                                  read_initial_info)

pathObj = PathFind()
###########################################################################################
STEP_LIST = [
    'load_slc',
    'phase_inversion',
    'generate_ifgram',
    'unwrap_ifgram',
    'load_ifgram',
    'correct_unwrap_error',
    'phase_to_range',
    'mintpy_corrections']

RUN_FILES = {'load_slc': 'run_01_minopy_load_slc',
             'phase_inversion': 'run_02_minopy_phase_inversion',
             'generate_ifgram': 'run_03_minopy_generate_ifgram',
             'unwrap_ifgram': 'run_04_minopy_unwrap_ifgram',
             'load_ifgram': 'run_05_minopy_load_ifgram',
             'correct_unwrap_error': 'run_06_mintpy_correct_unwrap_error',
             'phase_to_range': 'run_07_minopy_phase_to_range',
             'mintpy_corrections': 'run_08_mintpy_corrections'}

##########################################################################


def main(iargs=None):
    start_time = time.time()

    Parser = MinoPyParser(iargs, script='minopy_app')
    inps = Parser.parse()

    if not iargs is None:
        log_message(inps.workDir, os.path.basename(__file__) + ' ' + ' '.join(iargs[:]))
    else:
        log_message(inps.workDir, os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1::]))

    os.chdir(inps.workDir)

    if not os.path.exists(os.path.join(inps.workDir, 'conf.full')):
        CONFIG_FILE = os.path.dirname(os.path.abspath(__file__)) + '/defaults/conf.full'
        shutil.copyfile(CONFIG_FILE, os.path.join(inps.workDir, 'conf.full'))

    app = minopyTimeSeriesAnalysis(inps.customTemplateFile, inps.workDir, inps)
    app.open()

    if inps.run_flag or inps.write_job and not inps.generate_template:
        app.create_run_files()
    else:
        if len(inps.run_steps) > 0:
            app.run(steps=inps.run_steps)

    app.close()

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
        self.write_job = inps.write_job
        self.run_flag = inps.run_flag
        self.copy_to_tmp = False
        if 'copy_to_tmp' in inps:
            self.copy_to_tmp = inps.copy_to_tmp

    def open(self):
        super().open()

        self.templateFile_mintpy = self.templateFile
        self.template_mintpy = self.template

        # Read minopy templates and add to mintpy template
        # 1. Get default template file
        self.templateFile = get_latest_template_minopy(self.workDir)
        # 2. read (custom) template files into dicts
        self._read_template_minopy()

        self.projectName = self.inps.projectName

        self.run_dir = os.path.join(self.workDir, pathObj.rundir)
        os.makedirs(self.run_dir, exist_ok=True)
        #self.ifgram_dir = os.path.join(self.workDir, pathObj.intdir)

        self.azimuth_look = 1
        self.range_look = 1

        self.hostname = os.getenv('HOSTNAME')
        if not self.hostname is None and self.hostname.startswith('login'):
            self.hostname = 'login'

        self.text_cmd = self.template['minopy.textCmd']
        if self.text_cmd in [None, 'None']:
            self.text_cmd = ''

        self.num_workers = int(self.template['minopy.compute.numWorker'])
        if not self.write_job:
            num_cpu = os.cpu_count()
            if self.num_workers > num_cpu:
                self.num_workers = num_cpu
                print('There are {a} workers available, numWorker is changed to {a}'. format(a=num_cpu))

        if not self.inps.generate_template:
            self.date_list, self.num_pixels, self.metadata = read_initial_info(self.workDir, self.templateFile)
            self.num_images = len(self.date_list)
            self.date_list_text = os.path.join(self.workDir, 'inputs/date_list.txt')
            with open(self.date_list_text, 'w+') as fr:
                fr.write("\n".join(self.date_list))

            if 'box' in self.metadata:
                self.metadata['LENGTH'] = self.metadata['box'][3] - self.metadata['box'][1]
                self.metadata['WIDTH'] = self.metadata['box'][2] - self.metadata['box'][0]
            else:
                self.metadata['LENGTH'] = int(self.metadata['LENGTH'])
                self.metadata['WIDTH'] = int(self.metadata['WIDTH'])

            self.ifgram_dir, self.pairs = self.get_interferogram_pairs()

        os.chdir(self.workDir)
        return

    def _read_template_minopy(self):

        if self.customTemplateFile:
            # Update default template file based on custom template
            print('update default template based on input custom template')
            self.templateFile = ut.update_template_file(self.templateFile, self.customTemplate)

        # 2) backup custome/default template file in inputs/pic folder
        for backup_dirname in ['inputs', 'pic']:
            backup_dir = os.path.join(self.workDir, backup_dirname)
            # create directory
            os.makedirs(backup_dir, exist_ok=True)

            # back up to the directory
            for tfile in [self.customTemplateFile, self.templateFile]:
                if tfile and ut.run_or_skip(out_file=os.path.join(backup_dir, os.path.basename(tfile)),
                                            in_file=tfile,
                                            check_readable=False,
                                            print_msg=False) == 'run':
                    shutil.copy2(tfile, backup_dir)
                    print('copy {} to {:<8} directory for backup.'.format(os.path.basename(tfile),
                                                                          os.path.basename(backup_dir)))

        # 3) read default template file
        print('read default template file:', self.templateFile)
        self.template = readfile.read_template(self.templateFile)
        auto_template_file = os.path.join(os.path.dirname(__file__), 'defaults/minopyApp_auto.cfg')
        self.template = check_template_auto_value(self.template, self.template_mintpy, auto_file=auto_template_file,
                                                  templateFile=self.templateFile)

        return

    def create_run_files(self):
        print('-' * 50)
        print('Create run files for all the steps')
        os.makedirs(self.run_dir, exist_ok=True)
        if self.write_job:
            from minsar.job_submission import JOB_SUBMIT
            inps = self.inps
            inps.custom_template_file = self.customTemplateFile
            if self.customTemplateFile is None:
                inps.custom_template_file = self.templateFile
            inps.work_dir = self.run_dir
            inps.out_dir = self.run_dir
            inps.num_data = self.num_images
            job_obj = JOB_SUBMIT(inps)
        else:
            job_obj = None
        self.run_load_slc('load_slc', job_obj)
        self.run_phase_inversion('phase_inversion', job_obj)
        self.run_interferogram('generate_ifgram', job_obj)
        self.run_unwrap('unwrap_ifgram', job_obj)
        self.run_load_ifg('load_ifgram', job_obj)
        self.run_correct_unwrap_error('correct_unwrap_error', job_obj)
        self.run_phase_to_range('phase_to_range', job_obj)
        self.run_write_correction_job('mintpy_corrections', job_obj)
        del job_obj

        return

    def run_load_slc(self, sname, job_obj):
        """ Loading images using load_slc.py script and crop is subsets are given.
        """
        run_file_load_slc = os.path.join(self.run_dir, RUN_FILES[sname])
        print('Generate {}'.format(run_file_load_slc))

        if self.template['minopy.subset.lalo'] == 'None' and self.template['minopy.subset.yx'] == 'None':
            print('WARNING: No crop area given in mintpy.subset, the whole image is going to be used.')
            print('WARNING: May take days to process!')

        scp_args = '--template {}'.format(self.templateFile)
        scp_args += ' --project_dir {}'.format(os.path.dirname(self.workDir))

        run_commands = ['{} load_slc.py {} --no_metadata_check\n'.format(self.text_cmd.strip("'"), scp_args)]
        run_commands = [cmd.lstrip() for cmd in run_commands]

        with open(run_file_load_slc, 'w+') as frun:
            frun.writelines(run_commands)

        if self.write_job or not job_obj is None:
            job_obj.num_bursts = 1
            job_obj.write_batch_jobs(batch_file=run_file_load_slc)

        return

    def run_phase_inversion(self, sname, job_obj):
        """ Non-Linear phase inversion.
        """
        run_inversion = os.path.join(self.run_dir, RUN_FILES[sname])
        print('Generate {}'.format(run_inversion))

        num_length_patch = math.ceil(self.metadata['LENGTH'] / int(self.template['minopy.inversion.patchSize']))
        num_width_patch = math.ceil(self.metadata['WIDTH'] / int(self.template['minopy.inversion.patchSize']))
        num_patches = num_length_patch * num_width_patch
        print('Total number of PATCHES: {}'.format(num_patches))
        number_of_nodes = math.ceil(num_patches / self.num_workers)
        print('Number of tasks for step 2: {}'.format(number_of_nodes))
        num_bursts = int(self.template['minopy.inversion.patchSize'])**2 // 40000

        slc_stack = os.path.join(self.workDir, 'inputs/slcStack.h5')
        if self.copy_to_tmp:
            tmp_slc_stack = '/tmp/slcStack.h5'
        else:
            tmp_slc_stack = slc_stack

        run_commands = []

        scp_args = '--work_dir {a0} --range_window {a1} --azimuth_window {a2} --method {a3} --test {a4} ' \
                   '--patch_size {a5} --num_worker {a6} --mini_stack_size {a7} --time_lag {a8} --ps_num_shp {a9}'.format(
            a0=self.workDir, a1=self.template['minopy.inversion.rangeWindow'],
            a2=self.template['minopy.inversion.azimuthWindow'], a3=self.template['minopy.inversion.phaseLinkingMethod'],
            a4=self.template['minopy.inversion.shpTest'], a5=self.template['minopy.inversion.patchSize'],
            a6=self.num_workers, a7=self.template['minopy.inversion.ministackSize'],
            a8=self.template['minopy.inversion.stbas_time_lag'],
            a9=self.template['minopy.inversion.PsNumShp'])

        if number_of_nodes > 1:
            for i in range(number_of_nodes):
                scp_args1 = scp_args + ' --index {}'.format(i)
                command_line = '{a} phase_inversion.py {b} --slc_stack {c}\n'.format(a=self.text_cmd.strip("'"),
                                                                                     b=scp_args1,
                                                                                     c=tmp_slc_stack)
                run_commands.append(command_line)
        else:
            command_line = '{a} phase_inversion.py {b} --slc_stack {c}\n'.format(a=self.text_cmd.strip("'"),
                                                                                 b=scp_args,
                                                                                 c=tmp_slc_stack)
            run_commands.append(command_line)

        with open(run_inversion, 'w+') as frun:
            frun.writelines(run_commands)

        if self.write_job or not job_obj is None:
            job_obj.num_bursts = num_bursts
            if self.copy_to_tmp:
                job_obj.write_batch_jobs(batch_file=run_inversion, num_cores_per_task=self.num_workers,
                                         distribute=slc_stack)
            else:
                job_obj.write_batch_jobs(batch_file=run_inversion, num_cores_per_task=self.num_workers)

        return

    def get_interferogram_pairs(self):

        ifg_dir_names = {'mini_stacks': 'mini_stacks',
                         'single_reference': 'single_reference',
                         'delaunay': 'delaunay',
                         'sequential': 'sequential'}

        ifgram_dir = os.path.join(self.workDir, 'inverted/interferograms')
        baseline_dir = self.template['minopy.load.baselineDir']
        if not os.path.exists(baseline_dir):
            baseline_dir = os.path.join(self.workDir, 'inputs/baselines')
        short_baseline_ifgs = os.path.join(self.workDir, 'short_baseline_ifgs.txt')

        if not self.template['minopy.interferograms.list'] in [None, 'None', 'auto']:
            ifgram_dir = ifgram_dir + '_list'
        else:
            ifgram_dir = ifgram_dir + '_{}'.format(ifg_dir_names[self.template['minopy.interferograms.type']])

        os.makedirs(ifgram_dir, exist_ok='True')

        if self.template['minopy.interferograms.referenceDate']:
            reference_date = self.template['minopy.interferograms.referenceDate']
        else:
            index = int(self.num_images // 2)
            reference_date = self.date_list[index]

        if self.template['minopy.interferograms.type'] == 'delaunay' and \
            self.template['minopy.interferograms.list'] in [None, 'None']:
            scp_args = ' -b {} -o {} --temporalBaseline {} --perpBaseline {} --date_list {}'.format(
                baseline_dir, short_baseline_ifgs, self.template['minopy.interferograms.delaunayTempThresh'],
                self.template['minopy.interferograms.delaunayPerpThresh'], self.date_list_text)
            find_baselines(scp_args.split())
            print('Successfully created short_baseline_ifgs.txt ')
            self.template['minopy.interferograms.list'] = short_baseline_ifgs

        pairs = []
        ind1 = []
        ind2 = []

        if not self.template['minopy.interferograms.list'] in [None, 'None']:
            with open(self.template['minopy.interferograms.list'], 'r') as f:
                lines = f.readlines()
            for line in lines:
                pairs.append((line.split('_')[0], line.split('\n')[0].split('_')[1]))
        else:
            if self.template['minopy.interferograms.type'] == 'sequential':
                for i in range(1, len(self.date_list)):
                        pairs.append((self.date_list[i-1], self.date_list[i]))

            if self.template['minopy.interferograms.type'] == 'single_reference':
                indx = self.date_list.index(reference_date)
                for i in range(0, len(self.date_list)):
                    if not indx == i:
                        pairs.append((self.date_list[indx], self.date_list[i]))
           
            if self.template['minopy.interferograms.type'] == 'mini_stacks':
                total_num_mini_stacks = self.num_images // int(self.template['minopy.interferograms.ministackSize'])
                #indx_ref_0 = None
                for i in range(total_num_mini_stacks):
                    indx_ref = i * int(self.template['minopy.interferograms.ministackSize'])
                    last_ind = indx_ref + int(self.template['minopy.interferograms.ministackSize']) + 1
                    if i == total_num_mini_stacks - 1:
                        last_ind = self.num_images
                    #indx_ref_0 = indx_ref
                    indx_ref_1 = (last_ind - indx_ref) // 2 + indx_ref
                    for t in range(indx_ref, last_ind):
                        if t != indx_ref_1:
                            pairs.append((self.date_list[indx_ref_1], self.date_list[t]))
                    #if not indx_ref_0 is None:
                    #    pairs.append((self.date_list[indx_ref_0], self.date_list[indx_ref_1]))
                    #indx_ref_0 = indx_ref_1
        for pair in pairs:
            ind1.append(self.date_list.index(pair[0]))
            ind2.append(self.date_list.index(pair[1]))
        plot_baselines(ind1=ind1, ind2=ind2, dates=self.date_list, baseline_dir=baseline_dir, out_dir=self.workDir)

        return ifgram_dir, pairs


    def run_interferogram(self, sname, job_obj):
        """ Export single reference interferograms
        """
        run_ifgs = os.path.join(self.run_dir, RUN_FILES[sname])
        print('Generate {}'.format(run_ifgs))

        if 'sensor_type' in self.metadata:
            sensor_type = self.metadata['sensor_type']
        else:
            sensor_type = 'tops'

        #  command for generating unwrap mask
        cmd_generate_unwrap_mask = '{} generate_unwrap_mask.py --geometry {} '.format(
            self.text_cmd.strip("'"), os.path.join(self.workDir, 'inputs/geometryRadar.h5'))

        if not self.template['minopy.unwrap.mask'] in ['None', None]:
            cmd_generate_unwrap_mask += '--mask {}'.format(self.template['minopy.unwrap.mask'])
        cmd_generate_unwrap_mask = cmd_generate_unwrap_mask.lstrip() + '\n'

        #  Add all commands to run file
        run_commands = []
        run_commands.append(cmd_generate_unwrap_mask)

        phase_series = os.path.join(self.workDir, 'inverted/phase_series.h5')
        if self.copy_to_tmp:
            tmp_phase_series = '/tmp/phase_series.h5'
        else:
            tmp_phase_series = phase_series
        num_cpu = os.cpu_count()
        num_lin = 0
        for pair in self.pairs:
            out_dir = os.path.join(self.ifgram_dir, pair[0] + '_' + pair[1])
            os.makedirs(out_dir, exist_ok='True')

            scp_args = '--reference {a1} --secondary {a2} --output_dir {a3} --azimuth_looks {a4} ' \
                       '--range_looks {a5} --filter_strength {a6} ' \
                       '--stack_prefix {a7} --stack {a8}'.format(a1=pair[0],
                                                           a2=pair[1],
                                                           a3=out_dir, a4=self.azimuth_look,
                                                           a5=self.range_look,
                                                           a6=self.template['minopy.interferograms.filterStrength'],
                                                           a7=sensor_type,
                                                           a8=tmp_phase_series)

            cmd = '{} generate_ifgram.py {}'.format(self.text_cmd.strip("'"), scp_args)
            cmd = cmd.lstrip()

            if not self.write_job:
                cmd = cmd + ' &\n'
                run_commands.append(cmd)
                num_lin += 1
                if num_lin == num_cpu:
                    run_commands.append('wait\n\n')
                    num_lin = 0
            else:
                cmd = cmd + '\n'
                run_commands.append(cmd)

        run_commands.append('wait\n\n')

        with open(run_ifgs, 'w+') as frun:
            frun.writelines(run_commands)

        if self.write_job or not job_obj is None:
            job_obj.num_bursts = self.num_pixels // 30000000
            if self.copy_to_tmp:
                job_obj.write_batch_jobs(batch_file=run_ifgs, distribute=phase_series)
            else:
                job_obj.write_batch_jobs(batch_file=run_ifgs)

        return

    def run_unwrap(self, sname, job_obj):
        """ Unwrapps single reference interferograms
        """
        run_file_unwrap = os.path.join(self.run_dir, RUN_FILES[sname])
        print('Generate {}'.format(run_file_unwrap))

        length = int(self.metadata['LENGTH'])
        width = int(self.metadata['WIDTH'])
        wavelength = self.metadata['WAVELENGTH']
        earth_radius = self.metadata['EARTH_RADIUS']
        height = self.metadata['HEIGHT']

        run_commands = []
        num_cpu = os.cpu_count()
        ntiles = self.num_pixels // 4000000
        if ntiles == 0:
            ntiles = 1
        num_cpu = num_cpu // ntiles

        num_lin = 0

        #corr_file = os.path.join(self.workDir, 'inverted/quality_{}'.format(self.template['minopy.timeseries.tempCohType']))
        corr_file = os.path.join(self.workDir, 'inverted/quality_average')
        unwrap_mask = os.path.join(self.workDir, 'inverted/mask_unwrap')
        #unwrap_mask = os.path.abspath(self.template['minopy.unwrap.mask'])

        for pair in self.pairs:
            out_dir = os.path.join(self.ifgram_dir, pair[0] + '_' + pair[1])
            if float(self.template['minopy.interferograms.filterStrength']) > 0:
                corr_file = os.path.join(out_dir, 'filt_fine.cor')
            os.makedirs(out_dir, exist_ok='True')

            scp_args = '--ifg {a1} --coherence {a2} --unwrapped_ifg {a3} '\
                       '--max_discontinuity {a4} --init_method {a5} --length {a6} ' \
                       '--width {a7} --height {a8} --num_tiles {a9} --earth_radius {a10} ' \
                       ' --wavelength {a11}'.format(a1=os.path.join(out_dir, 'filt_fine.int'),
                                                             a2=corr_file,
                                                             a3=os.path.join(out_dir, 'filt_fine.unw'),
                                                             a4=self.template['minopy.unwrap.maxDiscontinuity'],
                                                             a5=self.template['minopy.unwrap.initMethod'],
                                                             a6=length, a7=width, a8=height, a9=ntiles,
                                                             a10=earth_radius, a11=wavelength)
            if self.template['minopy.unwrap.mask']:
                scp_args += ' -m {a12}'.format(a12=unwrap_mask)
            if float(self.template['minopy.interferograms.filterStrength']) > 0 and self.template['minopy.unwrap.removeFilter']:
                scp_args += ' --rmfilter'
            if self.copy_to_tmp:
                scp_args += ' --tmp'
            cmd = '{} unwrap_ifgram.py {}'.format(self.text_cmd.strip("'"), scp_args)
            cmd = cmd.lstrip()

            if not self.write_job:
                cmd = cmd + ' &\n'
                run_commands.append(cmd)
                num_lin += 1
                if num_lin == num_cpu:
                    run_commands.append('wait\n\n')
                    num_lin = 0
            else:
                cmd = cmd + '\n'
                run_commands.append(cmd)

        run_commands.append('wait\n\n')

        with open(run_file_unwrap, 'w+') as frun:
            frun.writelines(run_commands)

        if self.write_job or not job_obj is None:
            job_obj.num_bursts = self.num_pixels // 3000000
            job_obj.write_batch_jobs(batch_file=run_file_unwrap, num_cores_per_task=ntiles)

        return

    def run_load_ifg(self, sname, job_obj):
        """Load InSAR stacks into HDF5 files in ./inputs folder.
        """

        run_file_load_ifg = os.path.join(self.run_dir, RUN_FILES[sname])
        print('Generate {}'.format(run_file_load_ifg))

        # 1) copy aux files (optional)
        super()._copy_aux_file()

        # 2) loading data
        scp_args = '--template {}'.format(self.templateFile)
        if self.customTemplateFile:
            scp_args += ' {}'.format(self.customTemplateFile)

        if self.projectName:
            scp_args += ' --project {}'.format(self.projectName)
        scp_args += ' --output {}'.format(self.workDir + '/inputs/ifgramStack.h5')

        run_commands = ['{} load_ifgram.py {}\n'.format(self.text_cmd.strip("'"), scp_args)]
        run_commands = run_commands[0].lstrip()
        os.makedirs(self.run_dir, exist_ok=True)

        with open(run_file_load_ifg, 'w+') as frun:
            frun.writelines(run_commands)

        if self.write_job or not job_obj is None:
            job_obj.num_bursts = 1
            job_obj.write_batch_jobs(batch_file=run_file_load_ifg)

        return

    def run_correct_unwrap_error(self, sname, job_obj):

        run_file_correct_unwrap = os.path.join(self.run_dir, RUN_FILES[sname])
        print('Generate {}'.format(run_file_correct_unwrap))

        run_commands = ['{} smallbaselineApp.py {} '.format(self.text_cmd.strip("'"), self.templateFile_mintpy) +
                        '--start reference_point --stop correct_unwrap_error --dir {}\n'.format(self.workDir)]
        run_commands = run_commands[0].lstrip()
        os.makedirs(self.run_dir, exist_ok=True)

        with open(run_file_correct_unwrap, 'w+') as frun:
            frun.writelines(run_commands)

        if self.write_job or not job_obj is None:
            job_obj.num_bursts = 1
            job_obj.write_batch_jobs(batch_file=run_file_correct_unwrap)

        return

    def run_phase_to_range(self, sname, job_obj):

        run_file_phase_to_range = os.path.join(self.run_dir, RUN_FILES[sname])
        print('Generate {}'.format(run_file_phase_to_range))

        run_commands = ['{} phase_to_range.py --work_dir {} --num_worker {}\n'.format(self.text_cmd.strip("'"),
                                                                                      self.workDir,
                                                                                      self.template['minopy.compute.numWorker'])]
        run_commands = run_commands[0].lstrip()
        os.makedirs(self.run_dir, exist_ok=True)

        with open(run_file_phase_to_range, 'w+') as frun:
            frun.writelines(run_commands)

        if self.write_job or not job_obj is None:
            job_obj.num_bursts = 1
            job_obj.write_batch_jobs(batch_file=run_file_phase_to_range)

        return

    def run_write_correction_job(self, sname, job_obj):
        run_file_corrections = os.path.join(self.run_dir, RUN_FILES[sname])
        print('Generate {}'.format(run_file_corrections))

        run_commands = ['{} smallbaselineApp.py {} --start correct_LOD --dir {}\n'.format(self.text_cmd.strip("'"),
                                                                                          self.templateFile_mintpy,
                                                                                          self.workDir)]

        run_commands = run_commands[0].lstrip()
        os.makedirs(self.run_dir, exist_ok=True)

        with open(run_file_corrections, 'w+') as frun:
            frun.writelines(run_commands)

        if self.write_job or not job_obj is None:
            job_obj.num_bursts = 1
            job_obj.write_batch_jobs(batch_file=run_file_corrections)

        return

    def run(self, steps=STEP_LIST):
        #import subprocess
        for sname in steps:
            if not sname in ['correct_unwrap_error', 'mintpy_corrections']:
                print('\n\n******************** step - {} ********************'.format(sname))
            job_obj = None
            if sname == 'load_slc':
                self.run_load_slc('load_slc', job_obj)
            elif sname == 'phase_inversion':
                self.run_phase_inversion('phase_inversion', job_obj)
            elif sname == 'generate_ifgram':
                self.run_interferogram('generate_ifgram', job_obj)
            elif sname == 'unwrap_ifgram':
                self.run_unwrap('unwrap_ifgram', job_obj)
            elif sname == 'load_ifgram':
                self.run_load_ifg('load_ifgram', job_obj)
            elif sname == 'correct_unwrap_error':
                self.run_correct_unwrap_error('correct_unwrap_error', job_obj)
            elif sname == 'phase_to_range':
                self.run_phase_to_range('phase_to_range', job_obj)
            elif sname == 'mintpy_corrections':
                self.run_write_correction_job('mintpy_corrections', job_obj)

            run_file = os.path.join(self.run_dir, RUN_FILES[sname])
            os.system('bash {}'.format(run_file))

        # go back to original directory
        print('Go back to directory:', self.cwd)
        os.chdir(self.cwd)

        # message
        msg = '\n###############################################################'
        msg += '\nNormal end of Non-Linear time series processing workflow!'
        msg += '\n##############################################################'
        print(msg)
        return

    def close(self, normal_end=True):
        # go back to original directory
        print('Go back to directory:', self.cwd)
        os.chdir(self.cwd)
        # message
        if normal_end:
            if self.run_flag or self.write_job:
                msg = '\n################################################'
                msg += '\n   Normal end of minopyApp creating run files!'
                msg += '\n################################################'
                print(msg)
            else:
                msg  = '\n################################################'
                msg += '\n   Normal end of minopyApp processing!'
                msg += '\n################################################'
                print(msg)
        return


###########################################################################################


if __name__ == '__main__':
    main()
