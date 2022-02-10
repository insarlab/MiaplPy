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

    def __init__(self, outname):
        config_path = os.path.dirname(outname)
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

    def unwrap_tops(self, function):
        self.f.write('###################################' + '\n')
        self.f.write(function + '\n')
        self.f.write('unwrap : ' + '\n')
        self.f.write('ifg : ' + self.ifgName + '\n')
        self.f.write('unw : ' + self.unwName + '\n')
        self.f.write('coh : ' + self.cohName + '\n')
        self.f.write('nomcf : ' + self.noMCF + '\n')
        self.f.write('reference : ' + self.reference + '\n')
        self.f.write('defomax : ' + self.defoMax + '\n')
        self.f.write('alks : ' + self.azimuthLooks + '\n')
        self.f.write('rlks : ' + self.rangeLooks + '\n')
        self.f.write('method : ' + self.unwMethod + '\n')
        self.f.write('rmfilter: ' + self.rmFilter + '\n')

    def unwrap_stripmap(self, function):
        self.f.write('##########################' + '\n')
        self.f.write(function + '\n')
        self.f.write('unwrap : ' + '\n')
        self.f.write('ifg : ' + self.ifgName + '\n')
        self.f.write('coh : ' + self.cohName + '\n')
        self.f.write('unwprefix : ' + self.unwName + '\n')
        self.f.write('nomcf : ' + self.noMCF + '\n')
        self.f.write('reference : ' + self.reference + '\n')
        self.f.write('defomax : ' + self.defoMax + '\n')
        self.f.write('alks : ' + self.azimuthLooks + '\n')
        self.f.write('rlks : ' + self.rangeLooks + '\n')
        self.f.write('method : ' + self.unwMethod + '\n')
        self.f.write('##########################' + '\n')


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

        #self.quality_file = self.work_dir + '/inverted/quality'

        self.run_dir = inps.run_dir

        self.ifgram_dir = inps.ifgram_dir

        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        self.run_outname = os.path.join(self.run_dir, runName)
        print('writing ', self.run_outname)

        self.config_path = os.path.join(self.work_dir, pathObj.configdir)

        self.runf = open(self.run_outname, 'w')

    def unwrap_tops(self, inps, pairs):
        for pair in pairs:
            reference = pair[0]
            secondary = pair[1]
            mergedDir = os.path.join(self.ifgram_dir, reference + '_' + secondary)
            configName = os.path.join(self.config_path, 'config_unwrap_ifgram_' + reference + '_' + secondary)
            configObj = MinopyConfig(configName)
            configObj.configure(self)
            configObj.ifgName = os.path.join(mergedDir, 'filt_fine.int')
            configObj.cohName = os.path.join(mergedDir, 'filt_fine.cor') #self.quality_file
            configObj.unwName = os.path.join(mergedDir, 'filt_fine.unw')
            configObj.noMCF = noMCF
            configObj.reference = os.path.join(self.work_dir, 'inputs/reference')
            configObj.defoMax = defoMax
            configObj.unwMethod = inps.template['MINOPY.stack.unwMethod']
            configObj.rangeLooks = '1'
            configObj.azimuthLooks = '1'
            configObj.rmFilter = 'True'
            configObj.unwrap_tops('[Function-1]')
            configObj.finalize()
            if inps.template['MINOPY.stack.textCmd'] in [None, 'None']:
                self.runf.write(pathObj.wrappercommandtops + configName + '\n')
            else:
                self.runf.write(inps.template['MINOPY.stack.textCmd'] + pathObj.wrappercommandtops + configName + '\n')

    def unwrap_stripmap(self, inps, pairs):
        for pair in pairs:
            configName = os.path.join(self.config_path, 'config_unwrap_ifgram_{}_{}'.format(pair[0], pair[1]))
            configObj = MinopyConfig(configName)
            configObj.configure(self)
            configObj.azimuthLooks = '1'
            configObj.rangeLooks = '1'
            configObj.unwMethod = 'snaphu'
            configObj.outDir = os.path.join(self.ifgram_dir + '/' +pair[0] + '_' + pair[1])
            configObj.ifgName = configObj.outDir + '/filt_fine.int'
            configObj.unwName = configObj.outDir + '/filt_fine.unwrap'
            configObj.cohName = configObj.outDir + '/filt_fine.cor' # self.quality_file
            configObj.noMCF = noMCF
            configObj.reference = os.path.join(self.work_dir + '/inputs/reference/data')
            configObj.defoMax = defoMax
            configObj.unwrap_stripmap('[Function-1]')  ###
            configObj.finalize()
            if inps.template['MINOPY.stack.textCmd'] in [None, 'None']:
                self.runf.write(pathObj.wrappercommandstripmap + configName + '\n')
            else:
                self.runf.write(inps.template['MINOPY.stack.textCmd'] + pathObj.wrappercommandstripmap + configName + '\n')

    def finalize(self):
        self.runf.close()

#######################################################