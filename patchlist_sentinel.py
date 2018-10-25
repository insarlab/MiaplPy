#! /usr/bin/env python2
############################################################
# Author:  Sara Mirzaee                                  #
# Nov, 2017
############################################################

import sys
import os
import glob
import numpy as np
import logging
import pysar
from pysar.utils import readfile

logger = logging.getLogger("process_sentinel")

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
    print ("          patchlist.py templateFile")
    print (" ")
    print ("  Example: ")
    print ("          patchlist.py $TE/PichinchaSMT51TsxD.template")
    print ("  ******************************************************************************************************")
    print ("  ******************************************************************************************************")
    print ("  ******************************************************************************************************")

    sys.exit(1)

  logger.info(os.path.basename(sys.argv[0]) + " " + sys.argv[1])
  
  projectName = os.path.basename(templateFileString).partition('.')[0]

  scratchDirectory = os.getenv('SCRATCHDIR')

  sqDir = scratchDirectory + '/' + projectName + "/SqueeSAR"
  print(sqDir)
  dirlist0 = os.listdir(sqDir)
  dirlist = []
  for t in dirlist0:
     if t[0:3]=='PAT':
        dirlist.append(np.str(t))

  os.chdir(sqDir)


  run_PSQ_sentinel = sqDir + "/run_PSQ_sentinel"

  f = open(run_PSQ_sentinel,'w')
  for d in dirlist:
      cmd_coreg = 'PSQ_sentinel.py ' + templateFileString + '\t' + d + ' \n'
      f.write(cmd_coreg)
  f.close()
  print(dirlist)
  np.save(sqDir + '/patchlist.npy', dirlist)

  print "job file created: " + run_PSQ_sentinel
  sys.exit()



if __name__ == '__main__':
  main(sys.argv[:])


