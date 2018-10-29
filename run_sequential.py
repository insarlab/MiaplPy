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

import generate_templates as gt

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

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, epilog=EXAMPLE)
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


def set_dates(ssara_output):
	global stored_date, most_recent
	
	most_recent_data = ssara_output.split("\n")[-2]
	most_recent = datetime.strptime(most_recent_data.split(",")[3], date_format)

	# Write Most Recent Date to File
	with open(os.getenv('OPERATIONS')+'/stored_date.date', 'rb') as stored_date_file:
	
		try:
			date_line = subprocess.check_output(['grep', dataset, os.getenv('OPERATIONS')+'/stored_date.date']).decode('utf-8')
			stored_date = datetime.strptime(date_line.split(": ")[1].strip('\n'), date_format)
		except subprocess.CalledProcessError as e:
			
			stored_date = datetime.strptime("1970-01-01T12:00:00.000000", date_format)
			
			with open(os.getenv('OPERATIONS')+'/stored_date.date', 'a+') as date_file:
				data = str(dataset + ": "+str(datetime.strftime(most_recent, date_format))+"\n")
				date_file.write(data)


###############################################################
stored_date = None			  					            		# previously stored date
most_recent = None								              		# date parsed from SSARA
inps = None;				       						            	# command line arguments
date_format = "%Y-%m-%dT%H:%M:%S.%f"								# date format for reading and writing dates


if __name__ == "__main__":
    from datetime import datetime
    logger.info("RUN_Sequential for %s:\n", datetime.fromtimestamp(time.time()).strftime(date_format))
    command_line_parse(sys.argv[1:])
    
    # Generate SSARA Options to Use
    ssara_options = create_ssara_options(inps.custom_template_file)
    
    # Run SSARA and check output	
		ssara_output = subprocess.check_output(ssara_options).decode('utf-8');

    # Sets date variables for stored and most recent dates
    set_dates(ssara_output)



