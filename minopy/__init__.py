#########################################################################
# Program is part of MiNoPy                                             #
# Author:  Sara Mirzaee                                                 #
#########################################################################

from __future__ import print_function
import sys
import os


minopy_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, minopy_path)
sys.path.insert(1, os.path.join(minopy_path, 'defaults'))
sys.path.insert(1, os.path.join(minopy_path, 'objects'))
sys.path.insert(1, os.path.join(minopy_path, 'lib'))
sys.path.insert(1, os.path.join(minopy_path, 'dev'))

from minopy.version import *
__version__ = release_version
#__logo__ = logo

try:
    os.environ['MINOPY_HOME']
except KeyError:
    print('Using default MintPy Path: %s' % (minopy_path))
    os.environ['MINOPY_HOME'] = minopy_path

#os.environ['PATH'] = os.getenv('PATH') + ':$MINOPY_HOME/minopy/lib'
