#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import sys
from messageRsmas import Message as msg
import argparse

#################################
EXAMPLE = """example:
  find_shp.py LombokSenAT156VV.template -p PATCH5_11
"""


def create_parser():
    """ Creates command line argument parser object. """

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, epilog=EXAMPLE)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('-f', dest='runfile', type=str, required=True, help='file containing run commands')
    parser.add_argument('-n', dest='coreNum', type=int, default=1, help='number of Cores')
    parser.add_argument('-q', dest='queue', type=str, default='general', help='job queue')
    parser.add_argument('-p', dest='projectID', type=str, default='insarlab', help='project name')
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
    inps.work_dir = os.path.dirname(inps.runfile)
    jname = os.path.basename(inps.runfile)
    os.chdir(inps.work_dir)
    
    with open (inps.runfile,'r') as f:
        inps.runlist = f.readlines()
        
    jobsname = list(map(lambda x: jname + '_' + str(x), range(len(inps.runlist))))
    count = 0
    for jobn in jobsname:
        ##### Write job setting
        with open('z_input_'+jobn+'.job','w+') as fjob:
            fjob.write('#! /bin/tcsh')
            fjob.write('\n#BSUB -J '+jobn)
            fjob.write('\n#BSUB -P '+inps.projectID)
            fjob.write('\n#BSUB -o  z_output_.%J.o')
            fjob.write('\n#BSUB -e  z_output_.%J.e')
            fjob.write('\n#BSUB -W '+inps.walltime)
            fjob.write('\n#BSUB -q '+inps.queue)
            fjob.write('\n#BSUB -n '+str(coreNum))
            fjob.write('\n#BSUB -R "rusage[mem='+str(inps.memory)+']"')
            if inps.queue == 'parallel':
                fjob.write('\n#BSUB -R "span[ptile=16]"')
            fjob.write('\ncd '+inps.work_dir)
            fjob.write('\n'+inps.runlist[count])
        count += 1
        
        submitCmd = 'bsub -q ' + inps.queue+' < z_inp_' + jobn + '.job';   msg('\n'+submitCmd);   os.system(submitCmd) 
            
          
#####################################################################

if __name__ == '__main__':
  main(sys.argv[1:])
       
        
    
