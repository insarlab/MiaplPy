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
    'load_slc_geometry',
    'phase_inversion',
    'concatenate_patch'
    'generate_ifgram',
    'unwrap_ifgram',
    'load_ifgram',
    'ifgram_correction',
    'network_inversion',
    'timeseries_correction']

RUN_FILES = {'load_slc_geometry': 'run_01_minopy_load_slc_geometry',
             'phase_inversion': 'run_02_minopy_phase_inversion',
             'concatenate_patch': 'run_03_minopy_concatenate_patch',
             'generate_ifgram': 'run_04_minopy_generate_ifgram',
             'unwrap_ifgram': 'run_05_minopy_unwrap_ifgram',
             'load_ifgram': 'run_06_minopy_load_ifgram',
             'ifgram_correction': 'run_07_mintpy_ifgram_correction',
             'network_inversion': 'run_08_minopy_network_inversion',
             'timeseries_correction': 'run_09_mintpy_timeseries_correction'}

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

        network_inversion_options = {}
        update_flag = False
        if self.customTemplateFile:
            if not 'mintpy.networkInversion.weightFunc' in self.customTemplate:
                self.customTemplate['mintpy.networkInversion.weightFunc'] = 'no'
                update_flag = True
            if not 'mintpy.networkInversion.minNormVelocity' in self.customTemplate:
                self.customTemplate['mintpy.networkInversion.minNormVelocity'] = 'no'
                update_flag = True
            if update_flag:
                self.templateFile_mintpy = ut.update_template_file(self.templateFile_mintpy, self.customTemplate)

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

        self.num_workers = int(self.template['minopy.multiprocessing.numProcessor'])
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

        if 'sensor_type' in self.metadata:
            self.sensor_type = self.metadata['sensor_type']
        else:
            self.sensor_type = 'tops'

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
            inps.work_dir = os.path.dirname(self.run_dir)
            inps.out_dir = self.run_dir
            inps.num_data = self.num_images
            job_obj = JOB_SUBMIT(inps)
        else:
            job_obj = None
        self.run_load_slc_geometry('load_slc_geometry', job_obj)
        self.run_phase_inversion('phase_inversion', job_obj)
        self.run_phase_inversion('concatenate_patch', job_obj)
        self.run_interferogram('generate_ifgram', job_obj)
        self.run_unwrap('unwrap_ifgram', job_obj)
        self.run_load_ifg('load_ifgram', job_obj)
        self.run_ifgram_correction('ifgram_correction', job_obj)
        self.run_network_inversion('network_inversion', job_obj)
        self.run_timeseries_correction('timeseries_correction', job_obj)

        del job_obj

        run_file_list = []
        for key, value in RUN_FILES.items():
            run_file_list.append(value)
        with open(self.workDir + '/run_files_list', 'w') as run_file:
            for item in run_file_list:
                run_file.writelines(item + '\n')

        return

    def run_load_slc_geometry(self, sname, job_obj):
        """ Loading images using load_slc.py script and crop is subsets are given.
        """
        run_file_load_slc = os.path.join(self.run_dir, RUN_FILES[sname])
        print('Generate {}'.format(run_file_load_slc))

        if self.template['minopy.subset.lalo'] == 'None' and self.template['minopy.subset.yx'] == 'None':
            print('WARNING: No crop area given in mintpy.subset, the whole image is going to be used.')
            print('WARNING: May take days to process!')

        scp_args = '--template {}'.format(self.templateFile)
        scp_args += ' --project_dir {} --work_dir {} '.format(os.path.dirname(self.workDir), self.workDir)

        run_commands = ['{} load_slc_geometry.py {} --no_metadata_check\n'.format(self.text_cmd.strip("'"), scp_args)]
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

        number_of_nodes = math.ceil(num_patches / self.num_workers)
        num_bursts = int(self.template['minopy.inversion.patchSize'])**2 // 40000

        slc_stack = os.path.join(self.workDir, 'inputs/slcStack.h5')
        if self.copy_to_tmp:
            tmp_slc_stack = '/tmp/slcStack.h5'
        else:
            tmp_slc_stack = slc_stack

        run_commands = []

        scp_args = '--work_dir {a0} --range_window {a1} --azimuth_window {a2} --patch_size {a3}'.format(
            a0=self.workDir, a1=self.template['minopy.inversion.rangeWindow'],
            a2=self.template['minopy.inversion.azimuthWindow'], a3=self.template['minopy.inversion.patchSize'])

        if sname == 'concatenate_patch':
            command_line = '{a} phase_inversion.py {b} --slc_stack {c} --concatenate\n'.format(
                a=self.text_cmd.strip("'"), b=scp_args, c=slc_stack)

            run_commands.append(command_line)
        else:
            print('Total number of PATCHES: {}'.format(num_patches))
            print('Number of tasks for step phase inversion: {}'.format(number_of_nodes))

            scp_args += ' --method {a1} --test {a2} --num_worker {a3} ' \
                        '--mini_stack_size {a4} --time_lag {a5} --ps_num_shp {a6}'.format(
                a1=self.template['minopy.inversion.phaseLinkingMethod'],
                a2=self.template['minopy.inversion.shpTest'],
                a3=self.num_workers, a4=self.template['minopy.inversion.ministackSize'],
                a5=self.template['minopy.inversion.stbas_time_lag'],
                a6=self.template['minopy.inversion.PsNumShp'])

            if not self.template['minopy.inversion.mask'] in [None, 'None']:
                scp_args += ' --mask {}'.format(os.path.abspath(self.template['minopy.inversion.mask']))

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

        return slc_stack

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
                num_seq = int(self.template['minopy.interferograms.numSequential'])
                for t in range(0, num_seq-1):
                    for l in range(t + 1, num_seq):
                        pairs.append((self.date_list[t], self.date_list[l]))
                for i in range(num_seq, len(self.date_list)):
                    for t in range(1, num_seq + 1):
                        pairs.append((self.date_list[i - t], self.date_list[i]))

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
                                                           a7=self.sensor_type,
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
        ntiles = self.num_pixels // 40000000
        if ntiles == 0:
            ntiles = 1
        num_cpu = num_cpu // ntiles

        num_lin = 0

        #corr_file = os.path.join(self.workDir, 'inverted/tempCoh_{}'.format(self.template['minopy.timeseries.tempCohType']))
        corr_file = os.path.join(self.workDir, 'inverted/tempCoh_average')
        unwrap_mask = os.path.join(self.workDir, 'inverted/mask_unwrap')
        #unwrap_mask = os.path.abspath(self.template['minopy.unwrap.mask'])

        for pair in self.pairs:
            out_dir = os.path.join(self.ifgram_dir, pair[0] + '_' + pair[1])
            #if float(self.template['minopy.interferograms.filterStrength']) > 0:
            #    corr_file = os.path.join(out_dir, 'filt_fine.cor')
            os.makedirs(out_dir, exist_ok='True')

            scp_args = '--ifg {a1} --coherence {a2} --unwrapped_ifg {a3} '\
                       '--max_discontinuity {a4} --init_method {a5} --length {a6} ' \
                       '--width {a7} --height {a8} --num_tiles {a9} --earth_radius {a10} ' \
                       ' --wavelength {a11}'.format(a1=os.path.join(out_dir, 'filt_fine.int'),
                                                             a2=corr_file,
                                                             a3=os.path.join(out_dir, 'filt_fine.unw'),
                                                             a4=self.template['minopy.unwrap.snaphu.maxDiscontinuity'],
                                                             a5=self.template['minopy.unwrap.snaphu.initMethod'],
                                                             a6=length, a7=width, a8=height, a9=ntiles,
                                                             a10=earth_radius, a11=wavelength)
            if self.template['minopy.unwrap.mask']:
                scp_args += ' -m {a12}'.format(a12=unwrap_mask)
            if float(self.template['minopy.interferograms.filterStrength']) > 0 and self.template['minopy.unwrap.removeFilter']:
                scp_args += ' --rmfilter'
            if self.copy_to_tmp:
                scp_args += ' --tmp'
            if self.template['minopy.unwrap.two-stage'] == 'yes':
                scp_args += ' --two-stage'
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

    def run_ifgram_correction(self, sname, job_obj):

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

    def run_network_inversion(self, sname, job_obj):

        run_file_network_inversion = os.path.join(self.run_dir, RUN_FILES[sname])
        print('Generate {}'.format(run_file_network_inversion))

        run_commands = ['{} network_inversion.py --template {} --work_dir {}\n'.format(self.text_cmd.strip("'"),
                                                                                       self.templateFile_mintpy,
                                                                                       self.workDir)]
        run_commands = run_commands[0].lstrip()
        os.makedirs(self.run_dir, exist_ok=True)

        with open(run_file_network_inversion, 'w+') as frun:
            frun.writelines(run_commands)

        if self.write_job or not job_obj is None:
            job_obj.num_bursts = 1
            job_obj.write_batch_jobs(batch_file=run_file_network_inversion)

        return

    def run_timeseries_correction(self, sname, job_obj):
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
            if sname == 'load_slc_geometry':
                self.run_load_slc_geometry('load_slc_geometry', job_obj)
            elif sname == 'phase_inversion':
                slc_stack = self.run_phase_inversion('phase_inversion', job_obj)
                if self.copy_to_tmp:
                    os.system('cp {} /tmp'.format(slc_stack))
            elif sname == 'concatenate_patch':
                self.run_phase_inversion('concatenate_patch', job_obj)
            elif sname == 'generate_ifgram':
                self.run_interferogram('generate_ifgram', job_obj)
            elif sname == 'unwrap_ifgram':
                self.run_unwrap('unwrap_ifgram', job_obj)
            elif sname == 'load_ifgram':
                self.run_load_ifg('load_ifgram', job_obj)
            elif sname == 'ifgram_correction':
                self.run_ifgram_correction('ifgram_correction', job_obj)
            elif sname == 'network_inversion':
                self.run_network_inversion('network_inversion', job_obj)
            elif sname == 'timeseries_correction':
                self.run_timeseries_correction('timeseries_correction', job_obj)

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
