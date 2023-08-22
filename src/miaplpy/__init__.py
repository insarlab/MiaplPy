#########################################################################
# Program is part of MiaplPy                                             #
# Author:  Sara Mirzaee                                                 #
#########################################################################

from __future__ import print_function
import sys
import os


miaplpy_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, miaplpy_path)
sys.path.insert(1, os.path.join(miaplpy_path, 'defaults'))
sys.path.insert(1, os.path.join(miaplpy_path, 'objects'))
sys.path.insert(1, os.path.join(miaplpy_path, 'lib'))
sys.path.insert(1, os.path.join(miaplpy_path, 'dev'))

from miaplpy.version import *
__version__ = release_version
#__logo__ = logo

try:
    os.environ['MIAPLPY_HOME']
except KeyError:
    print('Using default MintPy Path: %s' % (miaplpy_path))
    os.environ['MIAPLPY_HOME'] = miaplpy_path

#os.environ['PATH'] = os.getenv('PATH') + ':$MIAPLPY_HOME/miaplpy/lib'
