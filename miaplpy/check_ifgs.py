#!/usr/bin/env python3
# Author: Sara Mirzaee

import os
import argparse


def cmd_line_parse(iargs=None):
    parser = argparse.ArgumentParser(description='Remove old interferograms not in the list')
    parser.add_argument('-i', '--ifgs', dest='ifg_file', type=str, help='ifg text file')
    parser.add_argument('-d', '--intDir', dest='int_dir', type=str, help='interferograms directory')
    parser.add_argument('--remove', dest='remove', action='store_true', help='remove extra ifgrams')
    inps = parser.parse_args(args=iargs)
    return inps

def main(iargs=None):
    inps = cmd_line_parse(iargs)

    with open(inps.ifg_file, 'r') as f:
        intlist = f.readlines()

    intlist = [x.split('\n')[0] for x in intlist]

    intold = os.listdir(inps.int_dir)

    diff = list(set(intold) - set(intlist))
    #diff = list(set(intlist) - set(intold))
    print(diff)

    for d in diff:
        file = os.path.join(inps.int_dir, d)
        print(file)
        if inps.remove:
            os.system('rm -rf {}'.format(file))

    return


if __name__ == '__main__':
    main()



