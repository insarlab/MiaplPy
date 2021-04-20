#########################################################################
# Program is part of PySAR                                              #
# Copyright(c) 2013-2018, Sara Mirzaee Zhang Yunjun, Heresh Fattahi     #
# Author:  Sara Mirzaee, 2019 Jan                                       #
#########################################################################

from __future__ import print_function
import sys
import os


minopy_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, minopy_path)
sys.path.insert(1, os.path.join(minopy_path, 'defaults'))
sys.path.insert(1, os.path.join(minopy_path, 'objects'))
sys.path.insert(1, os.path.join(minopy_path, 'lib'))


from minopy.version import *
__version__ = release_version

try:
    os.environ['MINOPY_HOME']
except KeyError:
    print('Using default MintPy Path: %s' % (minopy_path))
    os.environ['MINOPY_HOME'] = minopy_path

#os.environ['PATH'] = os.getenv('PATH') + ':$MINOPY_HOME/minopy/lib'