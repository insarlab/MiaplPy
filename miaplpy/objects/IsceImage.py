############################################################
# Program is part of MiaplPy                                #
# Author:  Sara Mirzaee                                    #
############################################################

#from __future__ import print_function
import isceobj
from isceobj.Image.SlcImage import SlcImage


def create_slc_image(name=''):
    inst = SLC_Image(name=name)
    return inst


class SLC_Image(SlcImage):

    def __init__(self, family='', name=''):
        super(SlcImage, self).__init__(family, name)
        self.stackband = 1

        return

    def set_band(self, band=1):
        self.stackband = band

    def memMap(self, mode='r', band=None):
        super().memMap(mode=mode, band=self.stackband)
