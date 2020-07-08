#!/usr/bin/env python3
########################
# Author: Sara Mirzaee
#######################
import os
import sys
import time
import datetime
import shutil
import h5py
import re
import minopy
import minopy.workflow

from mintpy.utils import writefile, readfile, utils as ut
from minsar.job_submission import JOB_SUBMIT
import mintpy
from mintpy.smallbaselineApp import TimeSeriesAnalysis

import minopy_utilities as mut
from minopy.objects.arg_parser import MinoPyParser
from minopy.objects.slcStack import slcStack
from minopy.objects.stack_int import MinopyRun
from minopy.defaults.auto_path import autoPath, PathFind
from minopy.objects.utils import check_template_auto_value

pathObj = PathFind()
###########################################################################################
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

    #if inps.submit_flag:
    #    js.submit_script(job_name, job_file_name, sys.argv[:], inps.workDir, inps.wall_time, queue_name=inps.queue_name)

    if not iargs is None:
        mut.log_message(inps.workDir, os.path.basename(__file__) + ' ' + ' '.join(iargs[:]))
    else:
        mut.log_message(inps.workDir, os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1::]))

    os.chdir(inps.workDir)

    app = minopyTimeSeriesAnalysis(inps.customTemplateFile, inps.workDir, inps)
    app.startup
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

        self.run_dir = os.path.join(self.workDir, pathObj.rundir)
        # self.patch_dir = os.path.join(self.workDir, pathObj.patchdir)
        self.ifgram_dir = os.path.join(self.workDir, pathObj.intdir)
        self.templateFile = ''

        self.plot_sh_cmd = ''

        self.status = False

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

            for key in ['minopy.deramp', 'minopy.troposphericDelay.method']:
                if key in cdict.keys():
                    cdict[key] = cdict[key].lower().replace('-', '_')

            if 'processor' in cdict.keys():
                cdict['minopy.load.processor'] = cdict['processor']

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
        if self.template['minopy.geocode'] is False:
            for key in ['minopy.save.hdfEos5', 'minopy.save.kmz']:
                if self.template[key] is True:
                    self.template['minopy.geocode'] = True
                    print('Turn ON minopy.geocode in order to run {}.'.format(key))
                    break

        minopy_template = self.template.copy()
        for key, value in minopy_template.items():
            key2 = key.replace('minopy', 'mintpy')
            self.template[key2] = value
        return

    def run_crop(self, sname):
        """ Cropping images using crop_sentinel.py script.
        """

        os.chdir(self.workDir)

        if self.template['minopy.subset.lalo'] == 'None' and self.template['minopy.subset.yx'] == 'None':
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

    def update_inversion_test(self):

        slcStackObj = slcStack(self.workDir + '/inputs/slcStack.h5')
        inverted_date_list = self.workDir + '/inverted/inverted_date_list.txt'

        if not os.path.exists(inverted_date_list):
            self.status = True
            updated_index = 0
        else:
            self.status, updated_index = mut.update_or_skip_inversion(inverted_date_list, slcStackObj)
        return updated_index

    def run_phase_inversion(self, sname):
        """ Non-Linear phase inversion.
        """

        updated_index = self.update_inversion_test()

        scp_args = '-w {a0} -r {a1} -a {a2} -m {a3} -t {a4} -p {a5} -s {a6} -i {a7} -c {a8} ' \
                   '--num-worker {a9} \n'.format(a0=self.workDir, a1=self.template['minopy.inversion.range_window'],
                                                 a2=self.template['minopy.inversion.azimuth_window'],
                                                 a3=self.template['minopy.inversion.plmethod'],
                                                 a4=self.template['minopy.inversion.shp_test'],
                                                 a5=self.template['minopy.inversion.patch_size'],
                                                 a6=os.path.join(self.workDir, 'inputs/slcStack.h5'),
                                                 a7=updated_index,
                                                 a8=self.template['mintpy.compute.cluster '],
                                                 a9=self.template['mintpy.compute.numWorker'])

        print('phase_inversion.py ', scp_args)
        if self.status:
            minopy.phase_inversion.main(scp_args.split())
        return

    def run_interferogram(self, sname):
        """ Export single master interferograms
        """

        inverted_slc = os.path.join(self.workDir, 'inverted/rslc_ref')
        ifgram_dir = os.path.join(self.workDir, 'inverted/interferograms')
        os.makedirs(ifgram_dir, exist_ok='True')

        slc_file = os.path.join(self.workDir, 'inputs/slcStack.h5')
        slcObj = slcStack(slc_file)
        slcObj.open(print_msg=False)
        date_list = slcObj.get_date_list()
        metadata = slcObj.get_metadata()
        if 'sensor_type' in metadata:
            sensor_type = metadata['sensor_type']
        else:
            sensor_type = 'tops'

        if self.template['minopy.interferograms.masterDate']:
            master_date = self.template['minopy.interferograms.masterDate']
        else:
            master_date = date_list[0]

        if self.template['minopy.interferograms.type'] == 'sequential':
            master_ind = None
        else:
            master_ind = date_list.index(master_date)

        pairs = []
        for i in range(0, len(date_list)):
            if not master_ind is None:
                if not master_ind == i:
                    pairs.append((date_list[master_ind], date_list[i]))
            else:
                if not i == 0:
                    pairs.append((date_list[i - 1], date_list[i]))

        # if master_ind is False:
        #    pairs.append((date_list[0], date_list[-1]))

        for pair in pairs:
            out_dir = os.path.join(ifgram_dir, pair[0] + '_' + pair[1])
            os.makedirs(out_dir, exist_ok='True')
            scp_args = '-s {a1} -o {a2} -b1 {a3} -b2 {a4} -p {a5}\n'.format(a1=inverted_slc, a2=out_dir,
                                                                           a3=date_list.index(pair[0]),
                                                                           a4=date_list.index(pair[1]),
                                                                           a5=sensor_type)
            print('generate_interferograms.py ', scp_args)
            minopy.generate_interferograms.main(scp_args.split())

        return

    def run_unwrap(self, sname):
        """ Unwrapps single master interferograms
        """

        run_file_unwrap = os.path.join(self.run_dir, 'run_minopy_unwrap')

        slc_file = os.path.join(self.workDir, 'inputs/slcStack.h5')
        slcObj = slcStack(slc_file)
        slcObj.open(print_msg=False)
        date_list = slcObj.get_date_list()
        metadata = slcObj.get_metadata()
        if 'sensor_type' in metadata:
            sensor_type = metadata['sensor_type']
        else:
            sensor_type = 'tops'

        if self.template['minopy.interferograms.masterDate']:
            master_date = self.template['minopy.interferograms.masterDate']
        else:
            master_date = date_list[0]

        if self.template['minopy.interferograms.type'] == 'sequential':
            master_ind = None
        else:
            master_ind = date_list.index(master_date)

        pairs = []
        for i in range(0, len(date_list)):
            if not master_ind is None:
                if not master_ind == i:
                    pairs.append((date_list[master_ind], date_list[i]))
            else:
                if not i == 0:
                    pairs.append((date_list[i - 1], date_list[i]))

        # if master_ind is False:
        #    pairs.append((date_list[0], date_list[-1]))

        inps = self.inps
        inps.run_dir = self.run_dir
        inps.ifgram_dir = self.ifgram_dir
        inps.template = self.template

        runObj = MinopyRun()
        runObj.configure(inps, 'run_minopy_unwrap')
        if sensor_type == 'stripmap':
            runObj.unwrap_stripmap(inps, pairs)
        else:
            runObj.unwrap_tops(inps, pairs)

        runObj.finalize()

        os.system('chmod +x {}'.format(self.workDir + '/configs/*'))

        inps.work_dir = inps.run_dir
        inps.out_dir = inps.run_dir
        inps.memory = '5000'
        inps.wall_time = '4:00'
        job_obj = JOB_SUBMIT(inps)
        job_status = job_obj.submit_batch_jobs(batch_file=run_file_unwrap)

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
            self.plot_result(print_aux=False, plot='True')

            # go back to original directory
            print('Go back to directory:', self.cwd)
            os.chdir(self.cwd)

            # raise error
            msg = 'step {}: NOT all required dataset found, exit.'.format(step_name)
            raise RuntimeError(msg)
        return


    def run_reference_point(self, step_name):
        """Select reference point.
        It 1) generate mask file from common conn comp
           2) generate average spatial coherence and its mask
           3) add REF_X/Y and/or REF_LAT/LON attribute to stack file
        """
        self.template['mintpy.plot'] = self.template['minopy.plot']
        self.run_network_modification(step_name)
        self.generate_ifgram_aux_file()

        stack_file = ut.check_loaded_dataset(self.workDir, print_msg=False)[1]
        coh_file = 'avgSpatialCoh.h5'

        scp_args = '{} -t {} -c {} --method maxCoherence'.format(stack_file, self.templateFile, coh_file)
        print('reference_point.py', scp_args)
        mintpy.reference_point.main(scp_args.split())
        self.run_quick_overview(step_name)

        return

    def write_to_timeseries(self, sname):

        mut.invert_ifgrams_to_timeseries(self.template, self.inps, self.workDir, writefile)
        functions = [mintpy.generate_mask, readfile, ut.run_or_skip, ut.add_attribute]
        mut.get_phase_linking_coherence_mask(self.template, self.workDir, functions)

        return

    def run(self, steps=STEP_LIST, plot=True):
        for sname in steps:
            print('\n\n******************** step - {} ********************'.format(sname))

            if sname == 'crop':
                self.run_crop(sname)

            elif sname == 'inversion':
                self.run_phase_inversion(sname)

            elif sname == 'ifgrams':
                self.run_interferogram(sname)

            elif sname == 'unwrap':
                self.run_unwrap(sname)

            elif sname == 'load_int':
                self.run_load_int(sname)

            elif sname == 'reference_point':

                ifgram_file = os.path.join(self.workDir, 'inputs/ifgramStack.h5')
                with h5py.File(ifgram_file, 'a') as f:
                    f.attrs['minopy.reference.yx'] = self.template['minopy.reference.yx']
                    f.attrs['minopy.reference.lalo'] = self.template['minopy.reference.lalo']
                f.close()

                self.run_reference_point(sname)

            elif sname == 'correct_unwrap_error':

                if self.template['minopy.unwrapError.method']:
                    self.template['minopy.unwrapError.method'] = 'bridging'
                ifgram_file = os.path.join(self.workDir, 'inputs/ifgramStack.h5')
                with h5py.File(ifgram_file, 'a') as f:
                    if 'unwrapPhase_bridging' in f.keys():
                        del f['unwrapPhase_bridging']
                super().run_unwrap_error_correction(sname)

            elif sname == 'write_to_timeseries':
                self.write_to_timeseries(sname)

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
            #    mut.email_minopy(self.workDir)

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
