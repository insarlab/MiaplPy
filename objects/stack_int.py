#!/usr/bin/env python3
########################
# Author: Sara Mirzaee
#######################

import os
from minopy.defaults.auto_path import PathFind

noMCF = 'False'
defoMax = '2'
maxNodes = 72

pathObj = PathFind()
###################################

class MinopyConfig(object):
    """
       A class representing the config file
    """

    def __init__(self, config_path, outname):
        if not os.path.exists(config_path):
            os.makedirs(config_path)
        self.f = open(outname, 'w')
        self.f.write('[Common]' + '\n')
        self.f.write('')
        self.f.write('##########################' + '\n')

    def configure(self, inps):
        for k in inps.__dict__.keys():
            setattr(self, k, inps.__dict__[k])
        self.plot = 'False'
        self.misreg_az = None
        self.misreg_rng = None
        self.multilook_tool = None
        self.no_data_value = None
        self.cleanup = None

    def generate_igram(self, function):
        self.f.write('###################################' + '\n')
        self.f.write(function + '\n')
        self.f.write('generate_ifgram_sq : ' + '\n')
        self.f.write('workDir : ' + self.work_dir + '\n')
        self.f.write('ifgDir : ' + self.ifgDir + '\n')
        self.f.write('ifgIndex : ' + self.ifgIndex + '\n')
        self.f.write('rangeWindow : ' + self.rangeWindow + '\n')
        self.f.write('azimuthWindow : ' + self.azimuthWindow + '\n')
        self.f.write('acquisitionNumber : ' + self.acq_num + '\n')
        self.f.write('rangeLooks : ' + self.rangeLooks + '\n')
        self.f.write('azimuthLooks : ' + self.azimuthLooks + '\n')
        if 'geom_master' in self.ifgDir:
            self.f.write('plmethod : ' + self.plmethod + '\n')

    def unwrap(self, function):
        self.f.write('###################################' + '\n')
        self.f.write(function + '\n')
        self.f.write('unwrap : ' + '\n')
        self.f.write('ifg : ' + self.ifgName + '\n')
        self.f.write('unw : ' + self.unwName + '\n')
        self.f.write('coh : ' + self.cohName + '\n')
        self.f.write('nomcf : ' + self.noMCF + '\n')
        self.f.write('master : ' + self.master + '\n')
        self.f.write('defomax : ' + self.defoMax + '\n')
        self.f.write('alks : ' + self.azimuthLooks + '\n')
        self.f.write('rlks : ' + self.rangeLooks + '\n')
        self.f.write('method : ' + self.unwMethod + '\n')

    def unwrapSnaphu(self, function):
        self.f.write('###################################' + '\n')
        self.f.write(function + '\n')
        self.f.write('unwrapSnaphu : ' + '\n')
        self.f.write('ifg : ' + self.ifgName + '\n')
        self.f.write('unw : ' + self.unwName + '\n')
        self.f.write('coh : ' + self.cohName + '\n')
        self.f.write('nomcf : ' + self.noMCF + '\n')
        self.f.write('master : ' + self.master + '\n')
        self.f.write('defomax : ' + self.defoMax + '\n')
        self.f.write('alks : ' + self.azimuthLooks + '\n')
        self.f.write('rlks : ' + self.rangeLooks + '\n')

    def finalize(self):
        self.f.close()



################################################

class MinopyRun(object):
    """
       A class representing a run which may contain several functions
    """

    def configure(self, inps, runName):

        for k in inps.__dict__.keys():
            setattr(self, k, inps.__dict__[k])

        self.work_dir = inps.workDir

        self.run_dir = inps.run_dir

        self.ifgram_dir = inps.ifgram_dir

        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        self.run_outname = os.path.join(self.run_dir, runName)
        print('writing ', self.run_outname)

        self.config_path = os.path.join(self.work_dir, pathObj.configdir)

        self.runf = open(self.run_outname, 'w')

    def generateIfg(self, inps, pairs, n_image):
        for ifg in pairs:
            configName = os.path.join(self.config_path, 'config_generate_ifgram_{}_{}'.format(ifg[0], ifg[1]))
            configObj = MinopyConfig(self.config_path, configName)
            configObj.configure(self)
            configObj.work_dir = self.work_dir
            configObj.ifgDir = os.path.join(self.ifgram_dir, '{}_{}'.format(ifg[0], ifg[1]))
            configObj.ifgIndex = str(pairs.index(ifg))
            configObj.rangeWindow = inps.template['mintpy.inversion.range_window']
            configObj.azimuthWindow = inps.template['mintpy.inversion.azimuth_window']
            configObj.acq_num = str(n_image)
            configObj.rangeLooks = inps.template['topsStack.rangeLooks']
            configObj.azimuthLooks = inps.template['topsStack.azimuthLooks']
            configObj.generate_igram('[Function-1]')
            configObj.finalize()
            if inps.template['topsStack.textCmd'] is None or inps.template['topsStack.textCmd'] == 'None':
                self.runf.write(pathObj.wrappercommandtops + configName + '\n')
            else:
                self.runf.write(inps.template['topsStack.textCmd'] + pathObj.wrappercommandtops + configName + '\n')
        configName = os.path.join(self.config_path, 'config_generate_quality_map')
        configObj = MinopyConfig(self.config_path, configName)
        configObj.configure(self)
        configObj.work_dir = self.work_dir
        configObj.ifgDir = os.path.join(self.work_dir, 'inputs')
        configObj.ifgIndex = str(0)
        configObj.rangeWindow = inps.template['mintpy.inversion.range_window']
        configObj.azimuthWindow = inps.template['mintpy.inversion.azimuth_window']
        configObj.acq_num = str(n_image)
        configObj.rangeLooks = inps.template['topsStack.rangeLooks']
        configObj.azimuthLooks = inps.template['topsStack.azimuthLooks']
        configObj.plmethod = inps.template['mintpy.inversion.plmethod']
        configObj.generate_igram('[Function-1]')
        configObj.finalize()
        if inps.template['topsStack.textCmd'] is None or inps.template['topsStack.textCmd'] == 'None':
            self.runf.write(pathObj.wrappercommandtops + configName + '\n')
        else:
            self.runf.write(inps.template['topsStack.textCmd'] + pathObj.wrappercommandtops + configName + '\n')

    def unwrap(self, inps, pairs):
        for pair in pairs:
            master = pair[0]
            slave = pair[1]
            mergedDir = os.path.join(self.ifgram_dir, master + '_' + slave)
            configName = os.path.join(self.config_path, 'config_igram_unw_' + master + '_' + slave)
            configObj = MinopyConfig(self.config_path, configName)
            configObj.configure(self)
            configObj.ifgName = os.path.join(mergedDir, 'filt_fine.int')
            configObj.cohName = os.path.join(mergedDir, 'filt_fine.cor')
            configObj.unwName = os.path.join(mergedDir, 'filt_fine.unw')
            configObj.noMCF = noMCF
            configObj.master = os.path.join(self.work_dir, 'inputs/master')
            configObj.defoMax = defoMax
            configObj.unwMethod = inps.template['topsStack.unwMethod']
            configObj.rangeLooks = inps.template['mintpy.inversion.range_window']
            configObj.azimuthLooks = inps.template['mintpy.inversion.azimuth_window']
            configObj.unwrap('[Function-1]')
            configObj.finalize()
            if inps.template['topsStack.textCmd'] is None or inps.template['topsStack.textCmd'] == 'None':
                self.runf.write(pathObj.wrappercommandtops + configName + '\n')
            else:
                self.runf.write(inps.template['topsStack.textCmd'] + pathObj.wrappercommandtops + configName + '\n')

    def finalize(self):
        self.runf.close()

#######################################################


