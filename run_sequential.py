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


def set_dates(ssara_output):
	global stored_date, most_recent
	
	most_recent_data = ssara_output.split("\n")[-2]
	most_recent = datetime.strptime(most_recent_data.split(",")[3], date_format)

	# Write Most Recent Date to File
	with open(inps.work_dir+'/stored_date.date', 'rb') as stored_date_file:
	
		try:
			date_line = subprocess.check_output(['grep', dataset, inps.work_dir+'/stored_date.date']).decode('utf-8')
			stored_date = datetime.strptime(date_line.split(": ")[1].strip('\n'), date_format)
		except subprocess.CalledProcessError as e:
			
			stored_date = datetime.strptime("1970-01-01T12:00:00.000000", date_format)
			
			with open(inps.work_dir+'/stored_date.date', 'a+') as date_file:
				data = str(dataset + ": "+str(datetime.strftime(most_recent, date_format))+"\n")
				date_file.write(data)
				
				
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
			data_to_download = download_new_string(stored_date,ssara_output)		
	else:
		stored_date = ''
		os.mkdir(old_process_dir)
		data_to_download = ssara_output
	        
        	
	with open(old_process_dir+'/downloaded_dates.dates', 'r+') as date_file:
		date_file.write(ssara_output)
	return data_to_download
	
	
def run_ssara(down_command, run_number=1):
    """ Runs ssara_federated_query-cj.py and checks for download issues.
        Runs ssara_federated_query-cj.py and checks continuously for whether the data download has hung without
        comleting or exited with an error code. If either of the above occur, the function is run again, for a
        maxiumum of 10 times.
        Parameters: run_number: int, the current iteration the wrapper is on (maxiumum 10 before quitting)
        Returns: status_cod: int, the status of the donwload (0 for failed, 1 for success)
    """
    
    logger.log(loglevel.INFO, "RUN NUMBER: %s", str(run_number))
    if run_number > 10:
        return 0


    # Runs ssara_federated_query-cj.py with proper options
    ssara_options = data_to_download + ['--parallel', '10', '--print', '--download']
    ssara_process = subprocess.Popen(ssara_options)

    completion_status = ssara_process.poll()  # the completion status of the process
    hang_status = False  # whether or not the download has hung
    wait_time = 10  # wait time in 'minutes' to determine hang status
    prev_size = -1  # initial download directory size
    i = 0  # the iteration number (for logging only)

    # while the process has not completed
    while completion_status is None:

        i = i + 1

        # Computer the current download directory size
        curr_size = int(subprocess.check_output(['du', '-s', os.getcwd()]).split()[0].decode('utf-8'))

        # Compare the current and previous directory sizes to determine determine hang status
        if prev_size == curr_size:
            hang_status = True
            logger.log(loglevel.WARNING, "SSARA Hung")
            ssara_process.terminate()  # teminate the process beacause download hung
            break;  # break the completion loop 

        time.sleep(60 * wait_time)  # wait 'wait_time' minutes before continuing
        prev_size = curr_size
        completion_status = ssara_process.poll()
        logger.log(loglevel.INFO, "{} minutes: {:.1f}GB, completion_status {}".format(i * wait_time, curr_size / 1024 / 1024,
                                                                        completion_status))

    exit_code = completion_status  # get the exit code of the command
    logger.log(loglevel.INFO, "EXIT CODE: %s", str(exit_code))

    bad_codes = [137]

    # If the exit code is one that signifies an error, rerun the entire command
    if exit_code in bad_codes or hang_status:
        logger.log(loglevel.WARNING, "Something went wrong, running again")
        run_ssara(run_number=run_number + 1)

    return 1
				
###############################################################
stored_date = None			  					            		# previously stored date
most_recent = None								              		# date parsed from SSARA
inps = None;				       						            	# command line arguments
date_format = "%Y-%m-%dT%H:%M:%S.%f"								# date format for reading and writing dates


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
    succesful = run_ssara(down_command)
    logger.log(loglevel.INFO, "DOWNLOADING SUCCESS: %s", str(succesful))
    logger.log(loglevel.INFO, "------------------------------------")	
    # Sets date variables for stored and most recent dates
    #set_dates(ssara_output)



