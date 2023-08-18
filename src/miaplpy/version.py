#!/usr/bin/env python3
# grab version / date of the latest commit
import os
import subprocess
import collections


###########################################################################
Tag = collections.namedtuple('Tag', 'version date')
release_history = (
    Tag('0.2.0', '2021-09-14'),
    Tag('0.1.0', '2021-04-23'),
)
__version__ = release_version = release_history[0].version
release_date = release_history[0].date

###########################################################################
def get_version_info(version='v{}'.format(release_version), date=release_date):
    """Grab version and date of the latest commit from a git repository"""
    # go to the repository directory
    dir_orig = os.getcwd()
    os.chdir(os.path.dirname(os.path.dirname(__file__)))

    # grab git info into string
    try:
        cmd = "git describe --tags"
        version = subprocess.check_output(cmd.split(), stderr=subprocess.DEVNULL)
        version = version.decode('utf-8').strip()

        #if there are new commits after the latest release
        if '-' in version:
            version, num_commit = version.split('-')[:2]
            version += '-{}'.format(num_commit)

        cmd = "git log -1 --date=short --format=%cd"
        date = subprocess.check_output(cmd.split(), stderr=subprocess.DEVNULL)
        date = date.decode('utf-8').strip()
    except:
        pass

    # go back to the original directory
    os.chdir(dir_orig)
    return version, date

###########################################################################


version_num, version_date = get_version_info()
version_description = """MiaplPy version {v}, date {d}""".format(
    v=version_num,
    d=version_date,
)

# generate_from: http://patorjk.com/software/taag/
logo = """
_________________________________________________      

  /##      /## /##                     / ## /#######
 | ###    /###|__/|                    | ##| ##__  ##
 | ####  /#### /##|/ ##### /##         | ##| ##  \ ## /##   /##
 | ## ##/## ##| ##| ##__  ### /####### | ##| #######/| ##  | ##
 | ##  ###| ##| ##| ##__  ###| ##__  ##| ##| ##      | ##  | ##
 | ##\  # | ##| ##| ##   \###| ##   \##| ##| ##      | ##  | ##
 | ## \/  | ##| ##|  ##### ##| #######/| ##| ##      |  #######
 |__/     |__/|__/| \____/|_/| ##____/ |__/|__/       \____  ##
                             | ##                     /##  | ## 
                             | ##                    |  ######/         
                             |__/                     \______/
 Miami Non-Linear Phase Linking software in Python   
          MiaplPy {v}, {d}
_________________________________________________
""".format(v=release_version, d=release_date)

website = 'https://github.com/insarlab/MiaplPy'

description = 'Miami Non-Linear Phase Linking software in Python'
