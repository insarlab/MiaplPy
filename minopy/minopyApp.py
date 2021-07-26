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
import datetime
import shutil
import h5py
import re
import math

import minopy
import minopy.workflow
from mintpy.utils import writefile, readfile, utils as ut

import mintpy
from mintpy.smallbaselineApp import TimeSeriesAnalysis
import minopy.minopy_utilities as mut
from minopy.objects.arg_parser import MinoPyParser
from minopy.objects.slcStack import slcStack
from minopy.defaults.auto_path import autoPath, PathFind
from minopy.objects.utils import check_template_auto_value

pathObj = PathFind()
###########################################################################################
STEP_LIST = [
    'load_slc',
    'inversion',
    'ifgrams',
    'unwrap',
    'load_ifg',
    'correct_unwrap_error',
    'phase_to_range',
    'mintpy_corrections']

RUN_FILES = {'load_slc': 'run_01_minopy_load_slc',
             'inversion': 'run_02_minopy_inversion',
             'ifgrams': 'run_03_minopy_ifgrams',
             'unwrap': 'run_04_minopy_unwrap',
             'load_ifg': 'run_05_minopy_load_ifg',
             'correct_unwrap_error': 'run_06_mintpy_correct_unwrap_error',
             'phase_to_range': 'run_07_minopy_phase_to_range',
             'mintpy_corrections': 'run_08_mintpy_corrections'}

##########################################################################


def main(iargs=None):
    start_time = time.time()

    Parser = MinoPyParser(iargs, script='minopy_app')
    inps = Parser.parse()

    if not iargs is None:
        mut.log_message(inps.workDir, os.path.basename(__file__) + ' ' + ' '.join(iargs[:]))
    else:
        mut.log_message(inps.workDir, os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1::]))

    os.chdir(inps.workDir)

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

    def open(self):
        super().open()

        self.templateFile_mintpy = self.templateFile
        self.template_mintpy = self.template

        # Read minopy templates and add to mintpy template
        # 1. Get default template file
        self.templateFile = mut.get_latest_template_minopy(self.workDir)
        # 2. read (custom) template files into dicts
        self._read_template_minopy()

        self.projectName = self.inps.projectName

        self.run_dir = os.path.join(self.workDir, pathObj.rundir)
        # self.patch_dir = os.path.join(self.workDir, pathObj.patchdir)
        self.ifgram_dir = os.path.join(self.workDir, pathObj.intdir)

        self.azimuth_look = 1
        self.range_look = 1

        self.hostname = os.getenv('HOSTNAME')
        if not self.hostname is None and self.hostname.startswith('login'):
            self.hostname = 'login'

        self.text_cmd = self.template['minopy.textCmd']
        if self.text_cmd in [None, 'None']:
            self.text_cmd = ''

        self.num_workers = int(self.template['minopy.compute.numWorker'])
        self.num_nodes = int(self.template['minopy.compute.numNode'])

        if not self.inps.generate_template:
            self.date_list, self.num_pixels, self.metadata = mut.read_initial_info(self.workDir, self.templateFile)
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
        self.template = check_template_auto_value(self.template, self.template_mintpy, auto_file=auto_template_file)

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
            job_obj = JOB_SUBMIT(inps)
        else:
            job_obj = None
        self.run_load_slc('load_slc', job_obj)
        self.run_phase_inversion('inversion', job_obj)
        self.run_interferogram('ifgrams', job_obj)
        self.run_unwrap('unwrap', job_obj)
        self.run_load_ifg('load_ifg', job_obj)
        self.run_correct_unwrap_error('correct_unwrap_error', job_obj)
        self.run_phase_to_range('phase_to_range', job_obj)
        self.run_write_correction_job('mintpy_corrections', job_obj)
        del job_obj

        return

    def run_load_slc(self, sname, job_obj):
        """ Loading images using load_slc.py script and crop is subsets are given.
        """
        run_file_load_slc = os.path.join(self.run_dir, RUN_FILES[sname])

        if self.template['minopy.subset.lalo'] == 'None' and self.template['minopy.subset.yx'] == 'None':
            print('WARNING: No crop area given in mintpy.subset, the whole image is going to be used.')
            print('WARNING: May take days to process!')

        scp_args = '--template {}'.format(self.templateFile)
        scp_args += ' --project_dir {}'.format(os.path.dirname(self.workDir))

        print('{} load_slc.py '.format(self.text_cmd.strip("'")), scp_args)
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

        scp_args = '--work_dir {a0} --range_window {a1} --azimuth_window {a2} --method {a3} --test {a4} ' \
                   '--patch_size {a5} --num_worker {a6}'.format(a0=self.workDir,
                                                              a1=self.template['minopy.inversion.rangeWindow'],
                                                              a2=self.template['minopy.inversion.azimuthWindow'],
                                                              a3=self.template['minopy.inversion.phaseLinkingMethod'],
                                                              a4=self.template['minopy.inversion.shpTest'],
                                                              a5=self.template['minopy.inversion.patchSize'],
                                                              a6=self.num_workers)

        command_line1 = '{} phase_inversion.py {}'.format(self.text_cmd.strip("'"), scp_args)

        command_line = 'cp {a} /tmp; unset LD_PRELOAD; '.format(a=os.path.join(self.workDir, 'inputs/slcStack.h5'))
        command_line += '{a} --slc_stack {b}; '.format(a=command_line1, b='/tmp/slcStack.h5')
        command_line += 'rm /tmp/slcStack.h5\n'

        run_commands = [command_line]

        with open(run_inversion, 'w+') as frun:
            frun.writelines(run_commands)

        if self.write_job or not job_obj is None:
            job_obj.num_bursts = self.num_pixels // 40000 // self.num_workers
            job_obj.write_batch_jobs(batch_file=run_inversion)

        return

    def run_interferogram(self, sname, job_obj):
        """ Export single reference interferograms
        """
        run_ifgs = os.path.join(self.run_dir, RUN_FILES[sname])

        ifgram_dir = os.path.join(self.workDir, 'inverted/interferograms')
        if not self.template['minopy.interferograms.list'] in [None, 'None', 'auto']:
            ifgram_dir = ifgram_dir + '_list'
        else:
            ifgram_dir = ifgram_dir + '_{}'.format(self.template['minopy.interferograms.type'])

        os.makedirs(ifgram_dir, exist_ok='True')

        if 'sensor_type' in self.metadata:
            sensor_type = self.metadata['sensor_type']
        else:
            sensor_type = 'tops'

        if self.template['minopy.interferograms.referenceDate']:
            reference_date = self.template['minopy.interferograms.referenceDate']
        else:
            reference_date = self.date_list[0]

        if self.template['minopy.interferograms.type'] == 'sequential':
            reference_ind = None
        elif self.template['minopy.interferograms.type'] == 'combine':
            reference_ind = 'multi'
        else:
            reference_ind = self.date_list.index(reference_date)

        pairs = []
        if not self.template['minopy.interferograms.list'] in [None, 'None']:
            with open(self.template['minopy.interferograms.list'], 'r') as f:
                lines = f.readlines()
            for line in lines:
                pairs.append((line.split('_')[0], line.split('\n')[0].split('_')[1]))
        else:
            if reference_ind == 'multi':
                indx = self.date_list.index(reference_date)
                for i in range(0, len(self.date_list)):
                    if not indx == i:
                        pairs.append((self.date_list[indx], self.date_list[i]))
                    if not i == 0:
                        pairs.append((self.date_list[i - 1], self.date_list[i]))
            else:
                for i in range(0, len(self.date_list)):
                    if not reference_ind is None:
                        if not reference_ind == i:
                            pairs.append((self.date_list[reference_ind], self.date_list[i]))
                    else:
                        if not i == 0:
                            pairs.append((self.date_list[i - 1], self.date_list[i]))
        pairs = list(set(pairs))

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
        num_cpu = os.cpu_count()
        num_lin = 0
        for pair in pairs:
            out_dir = os.path.join(ifgram_dir, pair[0] + '_' + pair[1])
            os.makedirs(out_dir, exist_ok='True')

            scp_args = '--reference {a1} --secondary {a2} --output_dir {a3} --azimuth_looks {a4} ' \
                       '--range_looks {a5} --filter_strength {a6} ' \
                       '--stack_prefix {a7} --stack {a8}'.format(a1=pair[0],
                                                           a2=pair[1],
                                                           a3=out_dir, a4=self.azimuth_look,
                                                           a5=self.range_look,
                                                           a6=self.template['minopy.interferograms.filterStrength'],
                                                           a7=sensor_type,
                                                           a8=phase_series)

            cmd = '{} generate_interferograms.py {}'.format(self.text_cmd.strip("'"), scp_args)
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
            job_obj.write_batch_jobs(batch_file=run_ifgs)

        return

    def run_unwrap(self, sname, job_obj):
        """ Unwrapps single reference interferograms
        """
        run_file_unwrap = os.path.join(self.run_dir, RUN_FILES[sname])

        length = int(self.metadata['LENGTH'])
        width = int(self.metadata['WIDTH'])
        wavelength = self.metadata['WAVELENGTH']
        earth_radius = self.metadata['EARTH_RADIUS']
        height = self.metadata['HEIGHT']

        if self.template['minopy.interferograms.referenceDate']:
            reference_date = self.template['minopy.interferograms.referenceDate']
        else:
            reference_date = self.date_list[0]

        if self.template['minopy.interferograms.type'] == 'sequential':
            reference_ind = None
        elif self.template['minopy.interferograms.type'] == 'combine':
            reference_ind = 'multi'
        else:
            reference_ind = self.date_list.index(reference_date)

        pairs = []
        if not self.template['minopy.interferograms.list'] in [None, 'None']:
            with open(self.template['minopy.interferograms.list'], 'r') as f:
                lines = f.readlines()
            for line in lines:
                pairs.append((line.split('_')[0], line.split('\n')[0].split('_')[1]))
        else:
            if reference_ind == 'multi':
                indx = self.date_list.index(reference_date)
                for i in range(0, len(self.date_list)):
                    if not indx == i:
                        pairs.append((self.date_list[indx], self.date_list[i]))
                    if not i == 0:
                        pairs.append((self.date_list[i - 1], self.date_list[i]))
            else:
                for i in range(0, len(self.date_list)):
                    if not reference_ind is None:
                        if not reference_ind == i:
                            pairs.append((self.date_list[reference_ind], self.date_list[i]))
                    else:
                        if not i == 0:
                            pairs.append((self.date_list[i - 1], self.date_list[i]))
        pairs = list(set(pairs))

        if not self.template['minopy.interferograms.list'] in [None, 'None', 'auto']:
            self.ifgram_dir = self.ifgram_dir + '_list'
        else:
            self.ifgram_dir = self.ifgram_dir + '_{}'.format(self.template['minopy.interferograms.type'])

        run_commands = []
        num_cpu = os.cpu_count()
        ntiles = self.num_pixels // 4000000
        if ntiles == 0:
            ntiles = 1
        num_cpu = num_cpu // ntiles

        num_lin = 0

        corr_file = os.path.join(self.workDir, 'inverted/quality') + '_msk'
        unwrap_mask = os.path.join(self.workDir, 'inverted/mask_unwrap')

        for pair in pairs:
            out_dir = os.path.join(self.ifgram_dir, pair[0] + '_' + pair[1])
            os.makedirs(out_dir, exist_ok='True')

            scp_args = '--ifg {a1} --coherence {a2} --unwrapped_ifg {a3} '\
                       '--max_discontinuity {a4} --init_method {a5} --length {a6} ' \
                       '--width {a7} --height {a8} --num_tiles {a9} --earth_radius {a10} ' \
                       ' --wavelength {a11} -m {a12}'.format(a1=os.path.join(out_dir, 'filt_fine.int'),
                                                             a2=corr_file,
                                                             a3=os.path.join(out_dir, 'filt_fine.unw'),
                                                             a4=self.template['minopy.unwrap.maxDiscontinuity'],
                                                             a5=self.template['minopy.unwrap.initMethod'],
                                                             a6=length, a7=width, a8=height, a9=ntiles,
                                                             a10=earth_radius, a11=wavelength, a12=unwrap_mask)
            cmd = '{} unwrap_minopy.py {}'.format(self.text_cmd.strip("'"), scp_args)
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

        # 1) copy aux files (optional)
        super()._copy_aux_file()

        # 2) loading data
        scp_args = '--template {}'.format(self.templateFile)
        if self.customTemplateFile:
            scp_args += ' {}'.format(self.customTemplateFile)

        if self.projectName:
            scp_args += ' --project {}'.format(self.projectName)
        scp_args += ' --output {}'.format(self.workDir + '/inputs/ifgramStack.h5')

        run_commands = ['{} load_ifg.py {}\n'.format(self.text_cmd.strip("'"), scp_args)]
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

        run_commands = ['{} phase_to_range.py --work_dir {}\n'.format(self.text_cmd.strip("'"), self.workDir)]
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
            elif sname == 'inversion':
                self.run_phase_inversion('inversion', job_obj)
            elif sname == 'ifgrams':
                self.run_interferogram('ifgrams', job_obj)
            elif sname == 'unwrap':
                self.run_unwrap('unwrap', job_obj)
            elif sname == 'load_ifg':
                self.run_load_ifg('load_ifg', job_obj)
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
            msg  = '\n################################################'
            msg += '\n   Normal end of minopyApp processing!'
            msg += '\n################################################'
            print(msg)
        return


###########################################################################################


if __name__ == '__main__':
    main()
