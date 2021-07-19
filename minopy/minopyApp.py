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
    'load_int',
    'reference_point',
    'quick_overview',
    'correct_unwrap_error',
    'write_to_timeseries',
    'correct_SET',
    'correct_troposphere',
    'deramp',
    'correct_topography',
    'residual_RMS',
    'reference_date',
    'velocity',
    'geocode',
    'google_earth',
    'hdfeos5']


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
    app.startup
    if len(inps.runSteps) > 0:
        app.run(steps=inps.runSteps)

    if inps.plot or (app.template['mintpy.plot'] and len(inps.runSteps) > 1):
        if inps.runSteps[-1] in STEP_LIST[5::]:
            app.plot_result()

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

        self.customTemplateFile = customTemplateFile
        self.cwd = os.path.abspath(os.getcwd())
        # 1. Go to the work directory
        # 1.1 Get workDir

        # 2. Get project name
        self.project_name = None
        if self.customTemplateFile and not os.path.basename(self.customTemplateFile) == 'minopy_template.cfg':
            self.project_name = os.path.splitext(os.path.basename(self.customTemplateFile))[0]
            print('Project name:', self.project_name)
        else:
            self.project_name = os.path.dirname(self.workDir)

        self.run_dir = os.path.join(self.workDir, pathObj.rundir)
        # self.patch_dir = os.path.join(self.workDir, pathObj.patchdir)
        self.ifgram_dir = os.path.join(self.workDir, pathObj.intdir)
        self.templateFile = ''

        self.plot_sh_cmd = ''

        self.status = False
        self.azimuth_look = 1
        self.range_look = 1

        self.hostname = os.getenv('HOSTNAME')
        if not self.hostname is None and self.hostname.startswith('login'):
            self.hostname = 'login'

    @property
    def startup(self):

        # 2.2 Go to workDir
        os.makedirs(self.workDir, exist_ok=True)
        os.chdir(self.workDir)
        print("Go to work directory:", self.workDir)

        # 3. Read templates
        # 3.1 Get default template file
        self.templateFile = mut.get_latest_template(self.workDir)
        # 3.2 read (custom) template files into dicts
        self._read_template()

        # 4. Copy the plot shell file
        sh_file = os.path.join(os.getenv('MINTPY_HOME'), 'mintpy/sh/plot_smallbaselineApp.sh')

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

        self.range_look = int(self.template['MINOPY.interferograms.range_look'])
        self.azimuth_look = int(self.template['MINOPY.interferograms.azimuth_look'])

        self.text_cmd = self.template['MINOPY.textCmd']
        if self.text_cmd in [None, 'None']:
            self.text_cmd = ''

        self.num_workers = int(self.template['MINOPY.compute.num_workers'])
        self.num_nodes = int(self.template['MINOPY.compute.num_nodes'])

        slc_file = os.path.join(self.workDir, 'inputs/slcStack.h5')

        if os.path.exists(slc_file):
            slcObj = slcStack(slc_file)
            slcObj.open(print_msg=False)
            self.date_list = slcObj.get_date_list()
            self.metadata = slcObj.get_metadata()
            self.num_pixels = int(self.metadata['LENGTH']) * int(self.metadata['WIDTH'])
        else:

            scp_args = '--template {}'.format(self.templateFile)
            if self.project_name:
                scp_args += ' --project_dir {}'.format(os.path.dirname(self.workDir))

            Parser_LoadSlc = MinoPyParser(scp_args.split(), script='load_slc')
            inps_loadSlc = Parser_LoadSlc.parse()

            iDict = minopy.load_slc.read_inps2dict(inps_loadSlc)
            minopy.load_slc.prepare_metadata(iDict)
            self.metadata = minopy.load_slc.read_subset_box(iDict)
            box = self.metadata['box']
            self.num_pixels = (box[2] - box[0]) * (box[3] - box[1])
            stackObj = minopy.load_slc.read_inps_dict2slc_stack_dict_object(iDict)
            self.date_list = stackObj.get_date_list()

        return

    def _read_template(self):
        # read custom template, to:
        # 1) update default template
        # 2) add metadata to ifgramStack file and HDF-EOS5 file
        self.customTemplate = None
        if self.customTemplateFile:
            cfile = self.customTemplateFile
            # Copy custom template file to inputs directory for backup

            for backup_dirname in ['inputs', 'pic']:
                backup_dir = os.path.join(self.workDir, backup_dirname)
                # create directory
                os.makedirs(backup_dir, exist_ok=True)
            
            inputs_dir = os.path.join(self.workDir, 'inputs')
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
                cdict['MINOPY.load.processor'] = cdict['processor']

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

        minopy_template = self.template.copy()
        for key, value in minopy_template.items():
            key2 = key.replace('minopy', 'mintpy')
            self.template[key2] = value

        if not 'processor' in self.template:
            print('WARNING: "processor" not defined in template, it is set to "isce" by default')
            self.template['processor'] = 'isce'
        return

    def run_load_slc(self, sname):
        """ Loading images using load_slc.py script and crop is subsets are given.
        """

        os.chdir(self.workDir)

        if self.template['mintpy.subset.lalo'] == 'None' and self.template['mintpy.subset.yx'] == 'None':
            print('WARNING: No crop area given in mintpy.subset, the whole image is going to be used.')
            print('WARNING: May take days to process!')

        scp_args = '--template {}'.format(self.templateFile)
        if self.project_name:
            scp_args += ' --project_dir {}'.format(os.path.dirname(self.workDir))

        print('{} load_slc.py '.format(self.text_cmd.strip("'")), scp_args)

        os.makedirs(self.run_dir, exist_ok=True)
        run_file_load_slc = os.path.join(self.run_dir, 'run_01_minopy_load_slc')
        run_commands = ['{} load_slc.py {}\n'.format(self.text_cmd.strip("'"), scp_args)]
        run_commands = [cmd.lstrip() for cmd in run_commands]

        with open(run_file_load_slc, 'w+') as frun:
            frun.writelines(run_commands)

        if self.write_job == False and self.hostname != 'login':
            minopy.load_slc.main(scp_args.split())
        else:
            from minsar.job_submission import JOB_SUBMIT
            inps = self.inps
            inps.custom_template_file = self.customTemplateFile
            inps.work_dir = self.run_dir
            inps.out_dir = self.run_dir
            job_obj = JOB_SUBMIT(inps)
            job_obj.write_batch_jobs(batch_file=run_file_load_slc)

        return

    def run_phase_inversion(self, sname):
        """ Non-Linear phase inversion.
        """
        inps = self.inps
        inps.work_dir = self.run_dir
        inps.out_dir = self.run_dir
        inps.custom_template_file = self.customTemplateFile

        scp_args = '--work_dir {a0} --range_window {a1} --azimuth_window {a2} --method {a3} --test {a4} ' \
                   '--patch_size {a5} --num_worker {a6}'.format(a0=self.workDir,
                                                              a1=self.template['MINOPY.inversion.range_window'],
                                                              a2=self.template['MINOPY.inversion.azimuth_window'],
                                                              a3=self.template['MINOPY.inversion.plmethod'],
                                                              a4=self.template['MINOPY.inversion.shp_test'],
                                                              a5=self.template['MINOPY.inversion.patch_size'],
                                                              a6=self.num_workers)

        command_line1 = '{} phase_inversion.py {}'.format(self.text_cmd.strip("'"), scp_args)

        if self.write_job == False and self.hostname != 'login':
            scp_args = scp_args + ' --slc_stack {a}\n'.format(a=os.path.join(self.workDir, 'inputs/slcStack.h5'))
            print('{} phase_inversion.py '.format(self.text_cmd.strip("'"), scp_args))
            minopy.phase_inversion.main(scp_args.split())
        else:
            from minsar.job_submission import JOB_SUBMIT

            inps.num_bursts = self.num_pixels // 40000 // self.num_workers

            job_obj = JOB_SUBMIT(inps)
            os.makedirs(self.run_dir, exist_ok=True)
            print('{} phase_inversion.py '.format(self.text_cmd.strip("'"), scp_args))
            command_line0 = '\ncp {} /tmp\nunset LD_PRELOAD\n'.format(os.path.join(self.workDir, 'inputs/slcStack.h5'))
            command_line1 = command_line1 + ' --slc_stack {a}\n'.format(a='/tmp/slcStack.h5')
            command_line2 = '\nrm /tmp/slcStack.h5\n'
            run_commands = command_line0 + command_line1 + command_line2
            job_name = os.path.join(self.run_dir, 'run_02_minopy_inversion.job')

            job_obj.get_memory_walltime(job_name)
            job_obj.write_single_job_file(job_name=job_name, job_file_name='run_02_minopy_inversion',
                                          command_line=run_commands, work_dir=self.run_dir,
                                          number_of_nodes=self.num_nodes)

            del job_obj

        return


    def run_interferogram(self, sname):
        """ Export single reference interferograms
        """

        ifgram_dir = os.path.join(self.workDir, 'inverted/interferograms')
        if not self.template['MINOPY.interferograms.list'] in [None, 'None', 'auto']:
            ifgram_dir = ifgram_dir + '_list'
        else:
            ifgram_dir = ifgram_dir + '_{}'.format(self.template['MINOPY.interferograms.type'])

        os.makedirs(ifgram_dir, exist_ok='True')

        if 'sensor_type' in self.metadata:
            sensor_type = self.metadata['sensor_type']
        else:
            sensor_type = 'tops'

        if self.template['MINOPY.interferograms.referenceDate']:
            reference_date = self.template['MINOPY.interferograms.referenceDate']
        else:
            reference_date = self.date_list[0]

        if self.template['MINOPY.interferograms.type'] == 'sequential':
            reference_ind = None
        elif self.template['MINOPY.interferograms.type'] == 'combine':
            reference_ind = 'multi'
        else:
            reference_ind = self.date_list.index(reference_date)

        pairs = []
        if not self.template['MINOPY.interferograms.list'] in [None, 'None']:
            with open(self.template['MINOPY.interferograms.list'], 'r') as f:
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

        inps = self.inps
        inps.run_dir = self.run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        inps.ifgram_dir = self.ifgram_dir
        inps.template = self.template
        inps.num_bursts = self.num_pixels // 30000000
        run_ifgs = os.path.join(inps.run_dir, 'run_03_minopy_ifgrams')
        run_commands = []

        rslc_ref = os.path.join(self.workDir, 'inverted/rslc_ref.h5')

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
                                                           a6=self.template['MINOPY.interferograms.filter_strength'],
                                                           a7=sensor_type,
                                                           a8=rslc_ref)

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

        if self.write_job == False and self.hostname != 'login':
            os.system('bash {}'.format(run_ifgs))
            #status = subprocess.Popen('bash {}'.format(run_ifgs), stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        else:
            from minsar.job_submission import JOB_SUBMIT
            inps.work_dir = inps.run_dir
            inps.out_dir = inps.run_dir
            inps.custom_template_file = self.customTemplateFile
            job_obj = JOB_SUBMIT(inps)
            job_obj.write_batch_jobs(batch_file=run_ifgs)
            del job_obj

        return

    def run_unwrap(self, sname):
        """ Unwrapps single reference interferograms
        """

        length = int(self.metadata['LENGTH'])
        width = int(self.metadata['WIDTH'])
        wavelength = self.metadata['WAVELENGTH']
        earth_radius = self.metadata['EARTH_RADIUS']
        height = self.metadata['HEIGHT']

        if self.template['MINOPY.interferograms.referenceDate']:
            reference_date = self.template['MINOPY.interferograms.referenceDate']
        else:
            reference_date = self.date_list[0]

        if self.template['MINOPY.interferograms.type'] == 'sequential':
            reference_ind = None
        elif self.template['MINOPY.interferograms.type']  == 'combine':
            reference_ind = 'multi'
        else:
            reference_ind = self.date_list.index(reference_date)

        pairs = []
        if not self.template['MINOPY.interferograms.list'] in [None, 'None']:
            with open(self.template['MINOPY.interferograms.list'], 'r') as f:
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

        # if reference_ind is False:
        #    pairs.append((date_list[0], date_list[-1]))

        inps = self.inps
        inps.run_dir = self.run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        inps.ifgram_dir = self.ifgram_dir

        if not self.template['MINOPY.interferograms.list'] in [None, 'None', 'auto']:
            inps.ifgram_dir = inps.ifgram_dir + '_list'
        else:
            inps.ifgram_dir = inps.ifgram_dir + '_{}'.format(self.template['MINOPY.interferograms.type'])

        inps.template = self.template
        run_file_unwrap = os.path.join(self.run_dir, 'run_04_minopy_un-wrap')
        run_commands = []

        num_cpu = os.cpu_count()
        ntiles = self.num_pixels / 4000000
        if ntiles > 1:
            x_tile = int(math.sqrt(ntiles)) + 1
            y_tile = x_tile
        else:
            x_tile = 1
            y_tile = 1
        num_cpu = min([num_cpu, x_tile * y_tile])

        num_lin = 0

        if self.azimuth_look * self.range_look > 1:
            corr_file = os.path.join(self.workDir, 'inverted/quality_ml')
        else:
            corr_file = os.path.join(self.workDir, 'inverted/quality')

        if not self.template['MINOPY.unwrap.mask'] in ['None', None]:
            mask_arg = ' {} -m {} --fill 0 -o {}'.format(corr_file,
                                                         self.template['MINOPY.unwrap.mask'],
                                                         corr_file + '_msk')
            mintpy.mask.main(mask_arg.split())
            corr_file = corr_file + '_msk'

        for pair in pairs:
            out_dir = os.path.join(inps.ifgram_dir, pair[0] + '_' + pair[1])
            os.makedirs(out_dir, exist_ok='True')

            #corr_file = os.path.join(out_dir, 'filt_fine.cor')

            scp_args = '--ifg {a1} --coherence {a2} --unwrapped_ifg {a3} '\
                       '--max_discontinuity {a4} --init_method {a5} ' \
                       '--length {a6} --width {a7} --height {a8} --num_tiles {a9} ' \
                       '--earth_radius {a10} --wavelength {a11}'.format(a1=os.path.join(out_dir, 'filt_fine.int'),
                                                                       a2=corr_file,
                                                                       a3=os.path.join(out_dir, 'filt_fine.unw'),
                                                                       a4=self.template['MINOPY.unwrap.max_discontinuity'],
                                                                       a5=self.template['MINOPY.unwrap.init_method'],
                                                                       a6=length, a7=width, a8=height, a9= num_cpu,
                                                                       a10=earth_radius, a11=wavelength)
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

            # print(cmd)
            # run_commands.append(cmd)

        run_commands.append('wait\n\n')

        with open(run_file_unwrap, 'w+') as frun:
            frun.writelines(run_commands)

        if self.write_job == False and self.hostname != 'login':
            os.system('bash {}'.format(run_file_unwrap))
        else:
            from minsar.job_submission import JOB_SUBMIT
            inps.work_dir = inps.run_dir
            inps.out_dir = inps.run_dir
            inps.custom_template_file = self.customTemplateFile
            inps.num_bursts = self.num_pixels // 3000000
            print('num bursts: {}'.format(inps.num_bursts))
            job_obj = JOB_SUBMIT(inps)
            job_obj.write_batch_jobs(batch_file=run_file_unwrap, num_cores_per_task=num_cpu)
            del job_obj

        return

    def write_correction_job(self, sname):
        from minsar.job_submission import JOB_SUBMIT
        run_commands = ['{} minopyApp.py {} --start load_int\n'.format(self.text_cmd.strip("'"), self.templateFile)]
        run_commands = run_commands[0].lstrip()
        os.makedirs(self.run_dir, exist_ok=True)
        run_file_corrections = os.path.join(self.run_dir, 'run_05_mintpy_corrections')

        with open(run_file_corrections, 'w+') as frun:
            frun.writelines(run_commands)

        inps = self.inps
        inps.custom_template_file = self.customTemplateFile
        inps.work_dir = self.run_dir
        inps.out_dir = self.run_dir
        job_obj = JOB_SUBMIT(inps)
        job_obj.write_batch_jobs(batch_file=run_file_corrections)

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
        print('{} load_int.py'.format(self.text_cmd.strip("'")), scp_args)
        minopy.load_int.main(scp_args.split())

        # 3) check loading result
        load_complete, stack_file, geom_file = ut.check_loaded_dataset(work_dir=self.workDir, print_msg=True)[0:3]

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
            self.plot_result(print_aux=False, plot='True')

            # go back to original directory
            print('Go back to directory:', self.cwd)
            os.chdir(self.cwd)

            # raise error
            msg = 'step {}: NOT all required dataset found, exit.'.format(step_name)
            raise RuntimeError(msg)
        return

    def write_to_timeseries(self, sname):
        if self.azimuth_look * self.range_look > 1:
            self.template['quality_file'] = os.path.join(self.workDir, 'inverted/quality_ml')
        else:
            self.template['quality_file'] = os.path.join(self.workDir, 'inverted/quality')
        mut.invert_ifgrams_to_timeseries(self.template, self.inps, self.workDir, writefile, self.num_workers)
        functions = [mintpy.generate_mask, readfile, ut.run_or_skip, ut.add_attribute]
        mut.get_phase_linking_coherence_mask(self.template, self.workDir, functions)

        return

    def run_topographic_residual_correction(self, sname):
        """step - correct_topography
        Topographic residual (DEM error) correction (optional).
        """
        geom_file = ut.check_loaded_dataset(self.workDir, print_msg=False)[2]
        fnames = self.get_timeseries_filename(self.template, self.workDir)[sname]
        in_file = fnames['input']
        out_file = fnames['output']

        if in_file != out_file:
            iargs = [in_file, '-t', self.templateFile, '-o', out_file, '--update',
                     '--cluster', 'local', '--num-worker', str(self.num_workers)]
            if self.template['mintpy.topographicResidual.pixelwiseGeometry']:
                iargs += ['-g', geom_file]
            print('\ndem_error.py', ' '.join(iargs))
            mintpy.dem_error.main(iargs)

        else:
            print('No topographic residual correction.')
        return

    def run(self, steps=STEP_LIST):
        for sname in steps:
            print('\n\n******************** step - {} ********************'.format(sname))

            if sname == 'load_slc':
                self.run_load_slc(sname)

            elif sname == 'inversion':
                self.run_phase_inversion(sname)

            elif sname == 'ifgrams':
                self.run_interferogram(sname)

            elif sname == 'unwrap':
                self.run_unwrap(sname)

            elif sname == 'mintpy_corrections':
                self.write_correction_job(sname)

            elif sname == 'load_int':
                self.run_load_int(sname)

            elif sname == 'reference_point':

                ifgram_file = os.path.join(self.workDir, 'inputs/ifgramStack.h5')
                with h5py.File(ifgram_file, 'a') as f:
                    f.attrs['mintpy.reference.yx'] = self.template['mintpy.reference.yx']
                    f.attrs['mintpy.reference.lalo'] = self.template['mintpy.reference.lalo']
                f.close()
                super().run_network_modification(sname)
                super().run_reference_point(sname)

            elif sname == 'quick_overview':
                super().run_quick_overview(sname)

            elif sname == 'correct_unwrap_error':
                super().run_unwrap_error_correction(sname)

            elif sname == 'write_to_timeseries':
                self.write_to_timeseries(sname)

            elif sname == 'correct_SET':
                super().run_solid_earth_tides_correction(sname)

            elif sname == 'correct_troposphere':
                super().run_tropospheric_delay_correction(sname)

            elif sname == 'deramp':
                super().run_phase_deramping(sname)

            elif sname == 'correct_topography':
                self.run_topographic_residual_correction(sname)

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
