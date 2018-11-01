#!/usr/bin/env python3
# Author Sara Mirzaee

import os
import sys
import subprocess
import argparse
from datetime import datetime
import shutil
import time
import glob
sys.path.insert(0, os.getenv('RSMAS_ISCE'))
import _process_utilities as putils
import generate_templates as gt
import _processSteps as prs
import logging

#################### LOGGERS AND LOGGING SETUP ####################

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

std_formatter = logging.Formatter("%(message)s")

general = logging.FileHandler(os.getenv('OPERATIONS')+'/LOGS/run_sequential.log', 'a+', encoding=None)
general.setLevel(logging.INFO)
general.setFormatter(std_formatter)
logger.addHandler(general)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(std_formatter)
logger.addHandler(console)

info_handler = None
error_handler = None
###################################################################
def create_parser():
    """ Creates command line argument parser object. """

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, epilog='')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('custom_template_file', nargs='?',
        help='custom template with option settings.\n')
    
    return parser


def command_line_parse(args):
    """ Parses command line agurments into inps variable. """

    global inps

    parser = create_parser()
    inps = parser.parse_args(args)


def create_ssara_options(custom_template_file):
	
	with open(custom_template_file, 'r') as template_file:
		options = ''
		for line in template_file:
			if 'ssaraopt' in line:
				options = line.strip('\n').rstrip().split("= ")[1]
				break;
					
	# Compute SSARA options to use
	options = options.split(' ')

	ssara_options = ['ssara_federated_query.py'] + options + ['--print']	
		
	return ssara_options

				
def compare_dates(old,new):
	old_data = old.split('ASF'))[1::]
	all_data = new.split('ASF'))[1::]
	a = len(all_data) - len(old_data)
	diffc = ''
	output_list = [diffc+li for li in all_data[len(old_data)::]]
	out_string = new.split('\nASF')[0] + '\n' + output_list
	return out_string
	 

def download_new_string(ssara_output):
	global stored_date, most_recent
	
	old_process_dir = inps.work_dir + '/old_process_dir'
	if os.path.isdir(old_process_dir) and os.path.isfile(old_process_dir+'/downloaded_dates.dates'):
		with open(old_process_dir+'/processed_dates.dates', 'r') as date_file:
			stored_date = date_file.read()
		nstored = len(stored_date.split('\n'))
		nnew = len(ssara_output.split('\n'))
		if nnew-nstored >= 10:
			data_to_download = compare_dates(stored_date,ssara_output)		
	else:
		stored_date = ''
		os.mkdir(old_process_dir)
		data_to_download = ssara_output
	        
        	
	with open(old_process_dir+'/downloaded_dates.dates', 'r+') as date_file:
		date_file.write(ssara_output)
	return data_to_download
	
	
def generate_files_csv(ssara_output):
	""" Generates a csv file of the files to download serially.
	
		Uses the `awk` command to generate a csv file containing the data files to be download
		serially. The output csv file is then sent through the `sed` command to remove the first five
		empty values to eliminate errors in download_ASF_serial.py.
	
	"""
	options = Template(inps.template).get_options()['ssaraopt']
	options = options.split(' ')
	
	filecsv_options = ssara_output+['|', 'awk', "'BEGIN{FS=\",\"; ORS=\",\"}{ print $14}'", '>', 'files.csv']
	csv_command = ' '.join(filecsv_options)
	subprocess.Popen(csv_command, shell=True).wait()
	sed_command = "sed 's/^.\{5\}//' files.csv > new_files.csv"
	
	subprocess.Popen(sed_command, shell=True).wait()
	
			
def call_ssara(slcDir):
        download_command = 'download_ASF_serial.py' + '-username' + password.asfuser + '-password' + password.asfpass + 'new_files.csv'
        command = 'ssh pegasus.ccs.miami.edu \"s.cgood;cd ' + slcDir + '; ' + os.getenv('PARENTDIR') + '/sources/rsmas_isce/' + download_command + '\"'
        messageRsmas.log(download_command)
        messageRsmas.log(command)
        status = subprocess.Popen(command, shell=True).wait()
	logger.log(loglevel.INFO, status)
	return status
	

def run_process():
	
	
###############################################################
stored_date = None			  			# previously stored date
most_recent = None						# date parsed from SSARA
inps = None;				       		        # command line arguments
date_format = "%Y-%m-%dT%H:%M:%S.%f"				# date format for reading and writing dates


if __name__ == "__main__":
    from datetime import datetime
    logger.info("RUN_Sequential for %s:\n", datetime.fromtimestamp(time.time()).strftime(date_format))
    command_line_parse(sys.argv[1:])
    inps.project_name = putils.get_project_name(custom_template_file=inps.custom_template_file)
    inps.work_dir = putils.get_work_directory(None, inps.project_name)
    
    # Generate SSARA Options to Use
    ssara_options = create_ssara_options(inps.custom_template_file)
    
    # Run SSARA and check output	
    ssara_output = subprocess.check_output(ssara_options).decode('utf-8');
    down_command = download_new_string(ssara_output)
    generate_files_csv(down_command)
    succesful = call_ssara(inps.work_dir + '/SLC')
    logger.log(loglevel.INFO, "DOWNLOADING SUCCESS: %s", str(succesful))
    logger.log(loglevel.INFO, "------------------------------------")	
    # Sets date variables for stored and most recent dates
    #set_dates(ssara_output)
    prs.step_runfiles(inps)
    prs.step_process(inps)


