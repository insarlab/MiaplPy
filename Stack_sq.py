#!/usr/bin/env python3
########################
#Author: Sara Mirzaee, Heresh Fattahi

#######################

import os, glob , sys
import subprocess as subp
import  datetime, glob
import copy
import shutil


noMCF = 'False'
defoMax = '2'
maxNodes = 72


class config(object):
    """
       A class representing the config file
    """
    def __init__(self, outname):
        self.f= open(outname,'w')
        self.f.write('[Common]'+'\n')
        self.f.write('')
        self.f.write('##########################'+'\n')

    def configure(self,inps):
        for k in inps.__dict__.keys():
            setattr(self, k, inps.__dict__[k])
        self.plot = 'False'
        self.misreg_az = None
        self.misreg_rng = None
        self.multilook_tool = None
        self.no_data_value = None
        self.cleanup = None ###SSS 7/2018: clean-up fine*int, if specified.


    def crop_sentinel(self, function):
        self.f.write('###################################' + '\n')
        self.f.write(function + '\n')
        self.f.write('crop_sentinel : ' + '\n')
        self.f.write('input : ' + self.input + '\n')
        self.f.write('output : ' + self.output + '\n')
        self.f.write('bbox : ' + self.bbox + '\n')
        self.f.write('multilook : ' + self.multi_look + '\n')
        self.f.write('range_looks : ' + self.rangeLooks + '\n')
        self.f.write('azimuth_looks : ' + self.azimuthLooks + '\n')
        self.f.write('multilook_tool : ' + self.multilook_tool + '\n')

    def create_patch(self, function):
        self.f.write('###################################' + '\n')
        self.f.write(function + '\n')
        self.f.write('create_patch : ' + '\n')
        self.f.write('slc_dir : ' + self.slcDir + '\n')
        self.f.write('squeesar_dir : ' + self.sqDir + '\n')
        self.f.write('patch_size : ' + self.patchSize + '\n')
        self.f.write('range_window : ' + self.rangeWindow + '\n')
        self.f.write('azimuth_window : ' + self.azimuthWindow + '\n')


    def phase_link(self, function):
        self.f.write('###################################' + '\n')
        self.f.write(function + '\n')
        self.f.write('PSQ_sentinel : ' + '\n')
        self.f.write('patch_dir : ' + self.patchDir + '\n')
        self.f.write('range_window : ' + self.rangeWindow + '\n')
        self.f.write('azimuth_window : ' + self.azimuthWindow + '\n')
        self.f.write('plmethod : ' + self.plmethod + '\n')


    def generate_igram(self, function):
        self.f.write('###################################' + '\n')
        self.f.write(function + '\n')
        self.f.write('generate_ifgram_sq : ' + '\n')
        self.f.write('squeesar_dir : ' + self.sqDir + '\n')
        self.f.write('ifg_dir : ' + self.ifgDir + '\n')
        self.f.write('ifg_index : ' + self.ifgIndex + '\n')
        self.f.write('range_window : ' + self.rangeWindow + '\n')
        self.f.write('azimuth_window : ' + self.azimuthWindow + '\n')
        self.f.write('acquisition_number : ' + self.acq_num + '\n')
        self.f.write('range_looks : ' + self.rangeLooks + '\n')
        self.f.write('azimuth_looks : ' + self.azimuthLooks + '\n')
        if 'geom_master' in self.ifgDir:
            self.f.write('plmethod : ' + self.plmethod + '\n')

    def unwrap(self, function):
        self.f.write('###################################'+'\n')
        self.f.write(function + '\n')
        self.f.write('unwrap : ' + '\n')
        self.f.write('ifg : ' + self.ifgName + '\n')
        self.f.write('unw : ' + self.unwName + '\n')
        self.f.write('coh : ' + self.cohName + '\n')
        self.f.write('nomcf : ' + self.noMCF + '\n')
        self.f.write('master : ' + self.master + '\n')
        #self.f.write('defomax : ' + self.defoMax + '\n')
        self.f.write('alks : ' + self.rangeLooks + '\n')
        self.f.write('rlks : ' + self.azimuthLooks + '\n')
        self.f.write('method : ' + self.unwMethod + '\n')

    def unwrapSnaphu(self, function):
        self.f.write('###################################'+'\n')
        self.f.write(function + '\n')
        self.f.write('unwrapSnaphu : ' + '\n')
        self.f.write('ifg : ' + self.ifgName + '\n')
        self.f.write('unw : ' + self.unwName + '\n')
        self.f.write('coh : ' + self.cohName + '\n')
        self.f.write('nomcf : ' + self.noMCF + '\n')
        self.f.write('master : ' + self.master + '\n')
        #self.f.write('defomax : ' + self.defoMax + '\n')
        self.f.write('alks : ' + self.rangeLooks + '\n')
        self.f.write('rlks : ' + self.azimuthLooks + '\n')

    def timeseries(self, function):
        self.f.write('###################################' + '\n')
        self.f.write(function + '\n')
        self.f.write('plApp : ' + '\n')
        self.f.write('template : ' + self.template + '\n')



########################################

    def maskLayover(self, function): ###SSS 7/2018: Add layover/water masking option.
        self.f.write('###################################'+'\n')
        self.f.write(function + '\n')
        self.f.write('maskLayover : ' + '\n')
        self.f.write('intname : ' + self.intname + '\n')
        if self.layovermsk: ###SSS 7/2018: layover mask, if specified.
            self.f.write('layovermaskname : ' + os.path.join(self.work_dir,'merged/geom_master/shadowMask.rdr') + '\n')
        if self.watermsk: ###SSS 7/2018: water mask, if specified.
            self.f.write('watermaskname : ' + os.path.join(self.work_dir,'merged/geom_master/waterMask.msk') + '\n')

    def createWbdMask(self, function): ###SSS 7/2018: Generate water mask.
        self.f.write('###################################'+'\n')
        self.f.write(function + '\n')
        self.f.write('createWbdMask : ' + '\n')
        self.f.write('lon : ' + os.path.join(self.work_dir,'merged/geom_master/lon.rdr') + '\n')
        self.f.write('lat : ' + os.path.join(self.work_dir,'merged/geom_master/lat.rdr') + '\n')
        self.f.write('workdir : ' + os.path.join(self.work_dir,'merged/geom_master') + '\n')
        self.f.write('bbox : ' + self.bbox + '\n')


    def finalize(self):
        self.f.close()
 

class run(object):
    """
       A class representing a run which may contain several functions
    """
    #def __init__(self):

    def configure(self, inps, runName):
        for k in inps.__dict__.keys():
            setattr(self, k, inps.__dict__[k])
        self.runDir = os.path.join(self.work_dir, 'run_files_SQ')
        if not os.path.exists(self.runDir):
            os.makedirs(self.runDir)

        self.run_outname = os.path.join(self.runDir, runName)
        print ('writing ', self.run_outname)

        self.config_path = os.path.join(self.work_dir,'configs_SQ')
        if not os.path.exists(self.config_path):
            os.makedirs(self.config_path)

        self.runf= open(self.run_outname,'w')



    def cropMergedSlc(self, acquisitions, inps):
        for slc in acquisitions:
            cropDir = os.path.join(self.work_dir, 'merged/SLC/' + slc)
            configName = os.path.join(self.config_path, 'config_crop_' + slc)
            configObj = config(configName)
            configObj.configure(self)
            configObj.input = os.path.join(cropDir, slc +'.slc.full')
            configObj.output = os.path.join(cropDir, slc + '.slc')
            configObj.bbox = inps.bbox_rdr
            configObj.multi_look = 'False'
            configObj.rangeLooks = inps.rangeLooks
            configObj.azimuthLooks = inps.azimuthLooks
            configObj.multilook_tool = 'gdal'
            configObj.crop_sentinel('[Function-1]')
            configObj.finalize()
            self.runf.write(self.text_cmd + 'SQWrapper.py -c ' + configName + '\n')

        list_geo = ['lat', 'lon', 'los', 'hgt', 'shadowMask', 'incLocal']
        multiookToolDict = {'lat*rdr': 'gdal', 'lon*rdr': 'gdal', 'los*rdr': 'gdal', 'hgt*rdr': "gdal",
                            'shadowMask*rdr': "isce", 'incLocal*rdr': "gdal"}
        for item in list_geo:
            pattern = item+'*rdr'
            geoDir = os.path.join(self.work_dir, 'merged/geom_master/')
            configName = os.path.join(self.config_path, 'config_crop_' + item)
            configObj = config(configName)
            configObj.configure(self)
            configObj.input = os.path.join(geoDir, item + '.rdr.full')
            configObj.output = os.path.join(geoDir, item + '.rdr')
            configObj.bbox = inps.bbox_rdr
            configObj.multi_look = 'False'
            configObj.rangeLooks = inps.rangeLooks
            configObj.azimuthLooks = inps.azimuthLooks
            configObj.multilook_tool = multiookToolDict[pattern]
            configObj.crop_sentinel('[Function-1]')
            configObj.finalize()
            self.runf.write(self.text_cmd + 'SQWrapper.py -c ' + configName + '\n')


    def createPatch(self, inps):
        configName = os.path.join(self.config_path, 'config_create_patch')
        configObj = config(configName)
        configObj.configure(self)
        configObj.slcDir = inps.slc_dirname
        configObj.sqDir = inps.squeesar_dir
        configObj.patchSize = inps.patch_size
        configObj.rangeWindow = inps.range_window
        configObj.azimuthWindow = inps.azimuth_window
        configObj.create_patch('[Function-1]')
        configObj.finalize()
        self.runf.write(self.text_cmd + 'SQWrapper.py -c ' + configName + '\n')


    def phaseLinking(self, inps):
        patchlist = glob.glob(inps.squeesar_dir+'/PATCH*')
        patchlist = [x.split('/')[-1] for x in patchlist]
        print(patchlist)
        for patch in patchlist:
            configName = os.path.join(self.config_path, 'config_phase_link_'+patch)
            configObj = config(configName)
            configObj.configure(self)
            configObj.patchDir = os.path.join(inps.squeesar_dir, patch)
            configObj.rangeWindow = inps.range_window
            configObj.azimuthWindow = inps.azimuth_window
            configObj.plmethod = inps.plmethod
            configObj.phase_link('[Function-1]')
            configObj.finalize()
            self.runf.write(self.text_cmd + 'SQWrapper.py -c ' + configName + '\n')


    def generateIfg(self, inps, acquisitions):
        ifgram_dir = os.path.dirname(inps.slc_dirname) + '/interferograms'
        if not os.path.isdir(ifgram_dir):
            os.mkdir(ifgram_dir)
        index = 0
        for ifg in acquisitions[1::]:
            index += 1
            configName = os.path.join(self.config_path, 'config_generate_ifgram_{}_{}'.format(acquisitions[0],ifg))
            configObj = config(configName)
            configObj.configure(self)
            configObj.sqDir = inps.squeesar_dir
            configObj.ifgDir = os.path.join(ifgram_dir, '{}_{}'.format(acquisitions[0],ifg))
            configObj.ifgIndex = str(index)
            configObj.rangeWindow = inps.range_window
            configObj.azimuthWindow = inps.azimuth_window
            configObj.acq_num = str(len(acquisitions))
            configObj.rangeLooks = inps.rangeLooks
            configObj.azimuthLooks = inps.azimuthLooks
            configObj.generate_igram('[Function-1]')
            configObj.finalize()
            self.runf.write(self.text_cmd + 'SQWrapper.py -c ' + configName + '\n')
        configName = os.path.join(self.config_path, 'config_generate_quality_map')
        configObj = config(configName)
        configObj.configure(self)
        configObj.sqDir = inps.squeesar_dir
        configObj.ifgDir = inps.geo_master_dir
        configObj.ifgIndex = str(0)
        configObj.rangeWindow = inps.range_window
        configObj.azimuthWindow = inps.azimuth_window
        configObj.acq_num = str(len(acquisitions))
        configObj.rangeLooks = inps.rangeLooks
        configObj.azimuthLooks = inps.azimuthLooks
        configObj.plmethod = inps.plmethod
        configObj.generate_igram('[Function-1]')
        configObj.finalize()
        self.runf.write(self.text_cmd + 'SQWrapper.py -c ' + configName + '\n')

    def unwrap(self, inps, pairs):
        for pair in pairs:
            master = pair[0]
            slave = pair[1]
            mergedDir = os.path.join(self.work_dir, 'merged/interferograms/' + master + '_' + slave)
            configName = os.path.join(self.config_path ,'config_igram_unw_' + master + '_' + slave)
            configObj = config(configName)
            configObj.configure(self)
            configObj.ifgName = os.path.join(mergedDir,'filt_fine.int')
            configObj.cohName = os.path.join(mergedDir,'filt_fine.cor')
            configObj.unwName = os.path.join(mergedDir,'filt_fine.unw')
            configObj.noMCF = noMCF
            configObj.master = os.path.join(self.work_dir,'master')
            configObj.defoMax = defoMax
            configObj.unwMethod = inps.unwMethod
            configObj.unwrap('[Function-1]')
            configObj.finalize()
            self.runf.write(self.text_cmd + 'SQWrapper.py -c ' + configName + '\n')


    def plAPP(self,inps):
        configName = os.path.join(self.config_path, 'config_corrections_and_velocity')
        configObj = config(configName)
        configObj.configure(self)
        configObj.template = inps.customTemplateFile
        configObj.timeseries('[Function-1]')
        configObj.finalize()
        self.runf.write(self.text_cmd + 'SQWrapper.py -c ' + configName + '\n')

################################################

    def create_wbdmask(self, pairs): ###SSS 7/2018: Generate water mask.
        configName = os.path.join(self.config_path ,'config_make_watermsk')
        configObj = config(configName)
        configObj.configure(self)
        if self.layovermsk:  ###SSS 7/2018: layover mask, if specified.
            configObj.layovermsk = 'True'
        if self.watermsk:  ###SSS 7/2018: water mask, if specified.
            configObj.watermsk = 'True'
        configObj.createWbdMask('[Function-1]')
        configObj.finalize()
        self.runf.write(self.text_cmd + 'SQWrapper.py -c ' + configName + '\n')

    def mask_layover(self, pairs): ###SSS 7/2018: Add layover/water masking option.
        for pair in pairs:
            master = pair[0]
            slave = pair[1]
            configName = os.path.join(self.config_path ,'config_igram_mask_' + master + '_' + slave)
            configObj = config(configName)
            configObj.configure(self)
            configObj.intname = os.path.join(self.work_dir,'merged/interferograms/'+master+'_'+slave,'filt_fine.int')
            if self.layovermsk:  ###SSS 7/2018: layover mask, if specified.
                configObj.layovermsk = 'True'
            if self.watermsk:  ###SSS 7/2018: water mask, if specified.
                configObj.watermsk = 'True'
            configObj.maskLayover('[Function-1]')
            configObj.finalize()
            self.runf.write(self.text_cmd + 'SQWrapper.py -c ' + configName + '\n')

    def finalize(self):
        self.runf.close()
        #writeJobFile(self.run_outname)

class sentinelSLC(object):
    """
        A Class representing the SLCs
    """
    def __init__(self, safe_file=None, orbit_file=None ,slc=None ):
        self.safe_file = safe_file
        self.orbit = orbit_file
        self.slc = slc

    def get_dates(self):
        datefmt = "%Y%m%dT%H%M%S"
        safe = os.path.basename(self.safe_file)
        fields = safe.split('_')
        self.platform = fields[0]
        self.start_date_time = datetime.datetime.strptime(fields[5], datefmt)
        self.stop_date_time = datetime.datetime.strptime(fields[6], datefmt)
        self.datetime = datetime.datetime.date(self.start_date_time)
        self.date = self.datetime.isoformat().replace('-','')

    def get_lat_lon(self):
        lats=[]
        lons=[]
        for safe in self.safe_file.split():
           from xml.etree import ElementTree as ET

           file=os.path.join(safe,'preview/map-overlay.kml')
           kmlFile = open( file, 'r' ).read(-1)
           kmlFile = kmlFile.replace( 'gx:', 'gx' )

           kmlData = ET.fromstring( kmlFile )
           document = kmlData.find('Document/Folder/GroundOverlay/gxLatLonQuad')
           pnts = document.find('coordinates').text.split()
           for pnt in pnts:
              lons.append(float(pnt.split(',')[0]))
              lats.append(float(pnt.split(',')[1]))
        self.SNWE=[min(lats),max(lats),min(lons),max(lons)]




    def getkmlQUAD(self,safe):
        # The coordinates in pnts must be specified in counter-clockwise order with the first coordinate corresponding to the lower-left corner of the overlayed image.
        # The shape described by these corners must be convex.
        # It appears this does not mean the coordinates are counter-clockwize in lon-lat reference.
        import zipfile
        from xml.etree import ElementTree as ET


        if safe.endswith('.zip'):
            zf = zipfile.ZipFile(safe,'r')
            fname = os.path.join(os.path.basename(safe).replace('zip','SAFE'), 'preview/map-overlay.kml')
            xmlstr = zf.read(fname)
            xmlstr=xmlstr.decode('utf-8')
            start = '<coordinates>'
            end = '</coordinates>'
            pnts = xmlstr[xmlstr.find(start)+len(start):xmlstr.find(end)].split()
        
        else:
            file=os.path.join(safe,'preview/map-overlay.kml')
            kmlFile = open( file, 'r' ).read(-1)
            kmlFile = kmlFile.replace( 'gx:', 'gx' )
            kmlData = ET.fromstring( kmlFile )
            document = kmlData.find('Document/Folder/GroundOverlay/gxLatLonQuad')
            pnts = document.find('coordinates').text.split()
    
        # convert the pnts to a list
        from scipy.spatial import distance as dist
        import numpy as np
        import cv2
        lats = []
        lons = []
        for pnt in pnts:
            lons.append(float(pnt.split(',')[0]))
            lats.append(float(pnt.split(',')[1]))
        pts = np.array([[a,b] for a,b in zip(lons,lats)])

        # The two points with most western longitude correspond to the left corners of the rectangle
        xSorted = pts[np.argsort(pts[:, 0]), :]
        leftMost = xSorted[:2, :]
        # the top right corner corresponds to the point which has the highest latitude of the two western most corners
        # the second point left will be the bottom left corner
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (bl, tl) = leftMost


        # the two points with the most eastern longitude correspond to the right cornersof the rectangle
        rightMost = xSorted[2:, :]

        '''print("left most")
        print(leftMost)
        print("")
        print("right most")
        print(rightMost)
        print("")'''


        # now that we have the top-left coordinate, use it as an
        # anchor to calculate the Euclidean distance between the
        # top-left and right-most points; by the Pythagorean
        # theorem, the point with the largest distance will be
        # our bottom-right point
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]
        
        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order
        temp = np.array([tl, tr, br, bl], dtype="float32")
        #print(temp)
        #print(pnts)
        pnts_new = [str(bl[0])+','+str(bl[1]),str(br[0])+','+str(br[1]) ,str(tr[0])+','+str(tr[1]),str(tl[0])+','+str(tl[1])]
        #print(pnts_new)
        #raise Exception ("STOP")
        return pnts_new

    def get_lat_lon_v2(self):

        import numpy as np
        lats = []
        lons = []

        # track the min lat and max lat in columns with the rows each time a different SAF file
        lat_frame_max = []
        lat_frame_min = []
        for safe in self.safe_file.split():
           safeObj=sentinelSLC(safe)
           pnts = safeObj.getkmlQUAD(safe)
           # The coordinates must be specified in counter-clockwise order with the first coordinate corresponding 
           # to the lower-left corner of the overlayed image
           counter=0
           for pnt in pnts:
              lons.append(float(pnt.split(',')[0]))
              lats.append(float(pnt.split(',')[1]))

              # only take the bottom [0] and top [3] left coordinates
              if counter==0:
                  lat_frame_min.append(float(pnt.split(',')[1]))
              elif counter==3:
                  lat_frame_max.append(float(pnt.split(',')[1]))
              counter+=1
        
        self.SNWE=[min(lats),max(lats),min(lons),max(lons)]

        # checking for missing gaps, by doing a difference between end and start of frames
        # will shift the vectors such that one can diff in rows to compare start and end
        # note need to keep temps seperate as otherwize one is using the other in calculation
        temp1 = max(lat_frame_max)
        temp2 = min(lat_frame_min)
        lat_frame_min.append(temp1)
        lat_frame_min.sort()
        lat_frame_max.append(temp2)
        lat_frame_max.sort()
        
        # combining the frame north and south left edge
        lat_frame_min = np.transpose(np.array(lat_frame_min))
        lat_frame_max = np.transpose(np.array(lat_frame_max))
        # if the differnce between top and bottom <=0 then there is overlap

        overlap_check = (lat_frame_min-lat_frame_max)<=0
        overlap_check = overlap_check.all()
        """if overlap_check:
            print(lat_frame_min)
            print(lat_frame_max)
            print(lat_frame_min-lat_frame_max)
            print("*********overlap")
        else:
            print(lat_frame_min)
            print(lat_frame_max)
            print(lat_frame_min-lat_frame_max)
            print("gap")"""
        
        #raise Exception("STOP")
        self.frame_nogap=overlap_check

    def get_lat_lon_v3(self,inps):
        from isceobj.Sensor.TOPS.Sentinel1 import Sentinel1
        lats=[]
        lons=[]

        for swathnum in inps.swath_num.split():
           obj = Sentinel1()
           obj.configure()
           obj.safe = self.safe_file.split()
           obj.swathNumber = int(swathnum)
           print(obj.polarization)
           # add by Minyan
           obj.polarization='vv'
          #obj.output = '{0}-SW{1}'.format(safe,swathnum)    
           obj.parse()

           s,n,w,e = obj.product.bursts[0].getBbox()
           lats.append(s);lats.append(n)
           lons.append(w);lons.append(e)

           s,n,w,e = obj.product.bursts[-1].getBbox()
           lats.append(s);lats.append(n)
           lons.append(w);lons.append(e)

        self.SNWE=[min(lats),max(lats),min(lons),max(lons)]

    def get_orbit(self, orbitDir, workDir):
        datefmt = "%Y%m%dT%H%M%S"
        orbit_files = glob.glob(os.path.join(orbitDir,  self.platform + '*.EOF'))
        if len(orbit_files) == 0:
            orbit_files = glob.glob(os.path.join(orbitDir, '*/{0}*.EOF'.format(self.platform)))

        match = False
        for orbit in orbit_files:
           orbit = os.path.basename(orbit)
           fields = orbit.split('_')
           orbit_start_date_time = datetime.datetime.strptime(fields[6].replace('V',''), datefmt)
           orbit_stop_date_time = datetime.datetime.strptime(fields[7].replace('.EOF',''), datefmt)

           if self.start_date_time > orbit_start_date_time and self.start_date_time < orbit_stop_date_time:
               self.orbit = os.path.join(orbitDir,orbit)
               self.orbitType = 'precise'
               match = True
               break
        if not match:
           print ("*****************************************")
           print (self.date)
           print ("orbit was not found in the "+orbitDir) # It should go and look online
           print ("downloading precise or restituted orbits ...")

           restitutedOrbitDir = os.path.join(workDir ,'orbits/' + self.date)
           if os.path.exists(restitutedOrbitDir):
              orbitFile = glob.glob(os.path.join(restitutedOrbitDir,'*.EOF'))[0]

              #fields = orbitFile.split('_')
              fields = os.path.basename(orbitFile).split('_')
              orbit_start_date_time = datetime.datetime.strptime(fields[6].replace('V',''), datefmt)
              orbit_stop_date_time = datetime.datetime.strptime(fields[7].replace('.EOF',''), datefmt)
              if self.start_date_time > orbit_start_date_time and self.start_date_time < orbit_stop_date_time and 'POEORB' in orbitFile:  ###SSS 8/18: Check orbit type
                  print ("precise orbit already exists.")
                  self.orbit =  orbitFile
                  self.orbitType = 'precise'
                  shutil.move(self.orbit,orbitDir)
                  os.rmdir(restitutedOrbitDir)
              else:
                  shutil.rmtree(restitutedOrbitDir)
                  os.makedirs(restitutedOrbitDir)
                  cmd = 'fetchOrbit.py -i ' + self.safe_file + ' -o ' + restitutedOrbitDir
                  print(cmd)
                  test = subp.run(['fetchOrbit.py','-i', self.safe_file, '-o', restitutedOrbitDir])
                  if test.returncode !=0:
                      raise RuntimeError('Error: Exception occurred during call to fetchOrbit.py')

                  orbitFile = glob.glob(os.path.join(restitutedOrbitDir,'*.EOF'))
                  self.orbit =  orbitFile[0]
                  if 'POEORB' in self.orbit: ###SSS 8/18: Check orbit type
                      print("downloaded precise orbit.")
                      self.orbitType = 'precise'
                      shutil.move(self.orbit,orbitDir)
                      os.rmdir(restitutedOrbitDir)
                  else:
                      print("downloaded restituted orbit.")
                      self.orbitType = 'restituted'


           #if not os.path.exists(restitutedOrbitDir):
           else:
              os.makedirs(restitutedOrbitDir)

              cmd = 'fetchOrbit.py -i ' + self.safe_file + ' -o ' + restitutedOrbitDir
              print(cmd)
              test = subp.run(['fetchOrbit.py','-i', self.safe_file, '-o', restitutedOrbitDir])
              if test.returncode !=0:
                  raise RuntimeError('Error: Exception occurred during call to fetchOrbit.py')

              orbitFile = glob.glob(os.path.join(restitutedOrbitDir,'*.EOF'))
              self.orbit =  orbitFile[0]
              if 'POEORB' in self.orbit: ###SSS 8/18: Check orbit type
                  print("downloaded precise orbit.")
                  self.orbitType = 'precise'
                  shutil.move(self.orbit,orbitDir)
                  os.rmdir(restitutedOrbitDir)
              else:
                  print("downloaded restituted orbit.")
                  self.orbitType = 'restituted'



# an example for writing job files when using clusters

"""
def writeJobFile(runFile):

  jobName = runFile + '.job'
  dirName = os.path.dirname(runFile)
  with open(runFile) as ff:
    nodes = len(ff.readlines())
  if nodes >maxNodes:
     nodes = maxNodes

  f = open (jobName,'w')
  f.write('#!/bin/bash '+ '\n')
  f.write('#PBS -N Parallel_GNU'+ '\n')
  f.write('#PBS -l nodes=' + str(nodes) + '\n')

  jobTxt='''#PBS -V
#PBS -l walltime=05:00:00
#PBS -q default
#PBS -m bae -M hfattahi@gps.caltech.edu

echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

echo Running on host `hostname`
echo Time is `date`

### Define number of processors
NPROCS=`wc -l < $PBS_NODEFILE`
echo This job has allocated $NPROCS cpus

# Tell me which nodes it is run on
echo " "
echo This jobs runs on the following processors:
echo `cat $PBS_NODEFILE`
echo " "

# 
# Run the parallel with the nodelist and command file
#

'''
  f.write(jobTxt+ '\n')
  f.write('parallel --sshloginfile $PBS_NODEFILE  -a ' + os.path.basename(runFile) + '\n')
  f.write('')
  f.close()
  
"""



