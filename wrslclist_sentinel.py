#! /usr/bin/env python3
############################################################
# Author:  Sara Mirzaee                                  #
# Nov, 2017
############################################################

import sys
import os
import glob
# import re

import numpy as np

import logging

# import my_logger
# import __init__

# print(os.path.abspath("./PySAR"))
# sys.path.append(os.path.abspath("./PySAR"))

import pysar
from pysar.utils import utils
from pysar.utils import readfile

logger = logging.getLogger("process_sentinel")


##########################33
def main(argv):
    try:
        templateFileString = argv[1]
    except:

        print (
            "  ******************************************************************************************************")
        print (
            "  ******************************************************************************************************")
        print (
            "  ******************************************************************************************************")
        print (" ")
        print (" ")
        print ("  Usage:")
        print ("          wrslclist_sentinel.py templateFile")
        print (" ")
        print ("  Example: ")
        print ("          wrslclist_sentinel.py $TE/PichinchaSMT51TsxD.template")
        print (
            "  ******************************************************************************************************")
        print (
            "  ******************************************************************************************************")
        print (
            "  ******************************************************************************************************")

        sys.exit(1)

    logger.info(os.path.basename(sys.argv[0]) + " " + sys.argv[1])
    projectName = os.path.basename(templateFileString).partition('.')[0]

    scratchDir = os.getenv('SCRATCHDIR')
    projdir = scratchDir + '/' + projectName
    slavedir = projdir + '/merged/SLC'
    dates = os.listdir(slavedir)

    run_writeSLC = projdir + '/merged/run_writeSLC'

    f = open(run_writeSLC, 'w')
    for d in dates:
        cmd_coreg = 'writeSQ_sentinel.py ' + templateFileString + '\t' + d + '/' + d + '.slc.full' + ' \n'
        f.write(cmd_coreg)

    f.close()

    print ("job file created: " + " run_writeSLC")
    sys.exit()


if __name__ == '__main__':
    main(sys.argv[:])


