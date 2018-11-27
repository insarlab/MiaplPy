#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import sys

#################################
EXAMPLE = """example:
  find_shp.py LombokSenAT156VV.template -p PATCH5_11
"""


def create_parser():
    """ Creates command line argument parser object. """

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, epilog=EXAMPLE)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('-f', dest='runfile', type=argparse.FileType('r'), required=True, help='file containing run commands')
    parser.add_argument('-n', dest='ncores', type=int, default=1, help='number of Cores')
    parser.add_argument('-q', dest='queue', type=str, default='general', help='job queue')
    parser.add_argument('-p', dest='project', type=str, default='insarlab', help='project name')
    parser.add_argument('-w', dest='walltime', type=str, default='1:00', help='wall time')
    parser.add_argument('-r', dest='memory', type=int, default=3600, help='memory use')
    

    return parser


def command_line_parse(args):
    """ Parses command line agurments into inps variable. """

    parser = create_parser()
    inps = parser.parse_args(args)
    return inps

###################################
def main(iargs=None):
    inps = command_line_parse(iargs)

    inps.project_name = os.path.basename(inps.custom_template_file).partition('.')[0]
    inps.project_dir = os.getenv('SCRATCHDIR') + '/' + inps.project_name
