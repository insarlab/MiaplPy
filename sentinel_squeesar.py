#!/usr/bin/env python3

# Author: Sara Mirzaee


import isce
import isceobj
import numpy as np
import argparse
import os
import copy

import gdal
import subprocess
import sys
import glob

import logging
import pysar
from pysar.utils import utils
from pysar.utils import readfile

################


def readim(slcname):
    objSlc = isceobj.createSlcImage()
    objSlc.load(slcname +'.xml')
    ds = gdal.Open(slcname + '.vrt', gdal.GA_ReadOnly)
    Im = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    return Im

def main(argv):
  try:
    templateFileString = argv[1]
  except:

    print ("  ******************************************************************************************************")
    print ("  ******************************************************************************************************")
    print ("  ******************************************************************************************************")
    print (" ")
    print (" ")
    print ("  Usage:")
    print ("          writeSQ_sentinel.py templatefile	slcfile")
    print (" ")
    print ("  Example: ")
    print ("          writeSQ_sentinel.py LombokSenAT156VV.template	20170310/20170310.slc.full")
    print ("  ******************************************************************************************************")
    print ("  ******************************************************************************************************")
    print ("  ******************************************************************************************************")

    sys.exit(1)
  logger = logging.getLogger("process_sentinel_squeesar")
  logger.info(os.path.basename(sys.argv[0]) + " " + sys.argv[1])
  templateContents = readfile.read_template(templateFileString)
  projectName = os.path.basename(templateFileString).partition('.')[0]      
  scratchDir = os.getenv('SCRATCHDIR')
  projdir = scratchDir + '/' + projectName
  slavedir = projdir+'/merged/SLC'
  sqdir = projdir+'/SqueeSAR'
  patchDir = sqdir+'/PATCH'

  listslv = os.listdir(slavedir)

  wra = int(templateContents['squeesar.wsizerange'])
  waz = int(templateContents['squeesar.wsizeazimuth'])

  if os.path.isfile(projdir + '/merged/cropped.npy'):
      print('Already cropped')
  else:
      cmd = 'crop_sentinel.py ' + templateFileString
      status = subprocess.Popen(cmd, shell=True).wait()
      if status is not 0:
          logger.error('ERROR Cropping SLCs')
          raise Exception('ERROR Cropping SLCs')

    
  if not os.path.isdir(sqdir):
    os.mkdir(sqdir)

  slc = readim(slavedir + '/' + listslv[0] + '/' + listslv[0] + '.slc.full')  #
  nimage = len(listslv)
  lin = slc.shape[0]
  sam = slc.shape[1]
  del slc
  pr1 = np.ogrid[0:lin-50:200]
  pr2 = pr1+200
  pr2[-1] = lin
  pr1[1::] = pr1[1::] - 2*waz

  pc1 = np.ogrid[0:sam-50:200]
  pc2 = pc1+200
  pc2[-1] = sam
  pc1[1::] = pc1[1::] - 2*wra
  pr = [[pr1], [pr2]]
  pc = [[pc1], [pc2]]
  np.save(sqdir+'/rowpatch.npy',pr)
  np.save(sqdir+'/colpatch.npy',pc) 


  if os.path.isfile(sqdir + '/flag.npy'):
      print('patchlist exist')
  else:
      patchlist = []
      for n1 in range(len(pr1)):
          lin1 = pr2[n1] - pr1[n1]
          for n2 in range(len(pc1)):
              sam1 = pc2[n2] - pc1[n2]
              patchlist.append(str(n1) + '_' + str(n2))
              patn = patchDir + str(n1) + '_' + str(n2)
              if not os.path.isdir(patn) or not os.path.isfile(patn + '/count.npy'):
                  os.mkdir(patn)
                  logger.info("Making PATCH" + str(n1) + '_' + str(n2))
                  amp1 = np.empty((nimage, lin1, sam1))
                  ph1 = np.empty((nimage, lin1, sam1))
                  count = 0

                  for dirs in listslv:
                      logger.info("Reading image" + dirs)
                      dname = slavedir + '/' + dirs + '/' + dirs + '.slc.full'
                      slc = np.memmap(dname, dtype=np.complex64, mode='r', shape=(lin, sam))
                      amp1[count, :, :] = np.abs(slc[pr1[n1]:pr2[n1], pc1[n2]:pc2[n2]])
                      ph1[count, :, :] = np.angle(slc[pr1[n1]:pr2[n1], pc1[n2]:pc2[n2]])
                      count += 1
                      del slc
                  np.save(patn + '/' + 'Amplitude.npy', amp1)
                  np.save(patn + '/' + 'Phase.npy', ph1)
                  np.save(patn + '/count.npy', nimage)
              else:
                  print('Next patch...')
      if n1 == len(pr1) - 1 and n2 == len(pc1) - 1:
          np.save(sqdir + '/flag.npy', 'patchlist_created')
          np.save(sqdir + '/patchlist.npy', patchlist)

  cmd = '$SQUEESAR/patchlist_sentinel.py ' + templateFileString
  status = subprocess.Popen(cmd, shell=True).wait()
  if status is not 0:
      logger.error('ERROR patchlist not found')
      raise Exception('ERROR patchlist not found')

  ##################
  flag = np.load(sqdir + '/flag.npy')

  if flag == 'patchlist_created':

      cmd = '$INT_SCR/split_jobs.py -f ' + sqdir + '/run_PSQ_sentinel -w 40:00 -r 3700'
      status = subprocess.Popen(cmd, shell=True).wait()
      if status is not 0:
          logger.error('ERROR running PSQ_sentinel.py')
          raise Exception('ERROR running PSQ_sentinel.py')


  patchlist = np.load(sqdir + '/patchlist.npy')
  for d in patchlist:
      ff0 = np.str(d)
      d = ff0[2:-1]
      if os.path.isfile(sqdir + '/' + d + '/endflag.npy'):
          count = 'True'
      else:
          print(str(d))
          count = 'False'
  if count == 'True':

      cmd = '$SQUEESAR/wrslclist_sentinel.py ' + templateFileString
      status = subprocess.Popen(cmd, shell=True).wait()
      if status is not 0:
          logger.error('ERROR making run_writeSQ list')
          raise Exception('ERROR making run_writeSQ list')

      run_write = projdir + '/merged/run_writeSLC'
      cmd = '$INT_SCR/split_jobs.py -f ' + projdir + '/merged/run_writeSLC -w 1:00 -r 5000'
      status = subprocess.Popen(cmd, shell=True).wait()
      if status is not 0:
          logger.error('ERROR writing SLCs')
          raise Exception('ERROR writing SLCs')


if __name__ == '__main__':
  main(sys.argv[:])    
       

