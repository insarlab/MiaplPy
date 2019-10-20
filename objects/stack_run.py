#!/usr/bin/env python3
#####################################
#Author: Sara Mirzaee
# based on stackSentinel.py
#####################################

import os
import sys
from argparse import Namespace
import shutil
import stackSentinel
from minsar.objects.stack_rsmas import rsmasRun
from minsar.utils.process_utilities import make_run_list
from minsar.objects.auto_defaults import PathFind
import contextlib
from minsar.objects import message_rsmas

pathObj = PathFind()
###########################################


class CreateRun:

    def __init__(self, inps):

        self.work_dir = inps.work_dir
        self.geo_master_dir = os.path.join(self.work_dir, pathObj.geomasterdir)
        self.minopy_dir = os.path.join(self.work_dir, pathObj.minopydir)

        self.inps = inps
        self.inps.custom_template_file = inps.custom_template_file

        self.command_options = []
        for item in inps.topsStack_template:
            if item == 'useGPU':
                if inps.topsStack_template[item] == 'True':
                    self.command_options = self.command_options + ['--' + item]
            elif not inps.topsStack_template[item] is None:
                self.command_options = self.command_options + ['--' + item] + [inps.topsStack_template[item]]

        clean_list = pathObj.isce_clean_list()
        for item in clean_list[0]:
            if os.path.isdir(os.path.join(inps.work_dir, item)):
                shutil.rmtree(os.path.join(inps.work_dir, item))

        return

    def run_stack_workflow(self):        # This part is for isceStack run_files

        message_rsmas.log(self.work_dir, 'stackSentinel.py' + ' ' + ' '.join(self.command_options))

        try:
            with open('out_stackSentinel.o', 'w') as f:
                with contextlib.redirect_stdout(f):
                    stackSentinel.main(self.command_options)
        except:
            with open('out_stackSentinel.e', 'w') as g:
                with contextlib.redirect_stderr(g):
                    stackSentinel.main(self.command_options)

        return

    def run_post_stack(self):

        inps = self.inps

        if inps.template['processingMethod'] == 'minopy' or inps.template['topsStack.workflow'] == 'slc':

            if not os.path.exists(self.minopy_dir):
                os.mkdir(self.minopy_dir)

            os.chdir(self.minopy_dir)
            inps_stack = stackSentinel.cmdLineParse(self.command_options)
            acquisitionDates, stackMasterDate, slaveDates, safe_dict, stackUpdate = stackSentinel.checkCurrentStatus(inps_stack)

            pairs_sm = []

            for i in range(len(acquisitionDates) - 1):
                pairs_sm.append((acquisitionDates[0], acquisitionDates[i + 1]))

            runObj = rsmasRun()
            runObj.configure(inps, 'run_single_master_interferograms')
            runObj.generateIfg(inps, pairs_sm)
            runObj.finalize()

            runObj = rsmasRun()
            runObj.configure(inps, 'run_unwrap')
            runObj.unwrap(inps, pairs_sm)
            runObj.finalize()

        return




