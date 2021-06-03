#!/usr/bin/env python3
# Author: Sara Mirzaee

import os
import numpy as np
import argparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
#import matplotlib.pyplot as plt

def cmd_line_parse(iargs=None):
    parser = argparse.ArgumentParser(description='find the minimum number of connected good interferograms')
    parser.add_argument('-b', '--baselineDir', dest='baseline_dir', type=str, help='baseline directory')
    parser.add_argument('-o', '--outFile', dest='out_file', type=str, default='./bestints.txt', help='output text file')
    parser.add_argument('-t', '--temporalBaseline', dest='threshold', type=int, help='temporal baseline threshold')
    inps = parser.parse_args(args=iargs)
    return inps


def main(iargs=None):
    inps = cmd_line_parse(iargs)

    bf = os.listdir(inps.baseline_dir)
    if not bf[0].endswith('txt'):
        bf2 = ['{}/{}.txt'.format(d, d) for d in bf]
    else:
        bf2 = bf
    baselines = []
    for d in bf2:
        print(d)
        with open(os.path.join(inps.baseline_dir, d), 'r') as f:
            lines = f.readlines()
            #print(d)
            if 'Bperp (average):' in lines[1]:
                baseline = lines[1].split('Bperp (average):')[1]
            else:
                baseline = lines[1].split('PERP_BASELINE_TOP')[1]
            baselines.append(baseline)

    baselines = [float(x.split('\n')[0].strip()) for x in baselines]
    reference = bf[0].split('_')[0]
    dates = [x.split('_')[1].split('.')[0] for x in bf]
    dates.append(reference)
    baselines.append(0)

    db_tuples = [(x, y) for x, y in zip(dates, baselines)]
    db_tuples.sort()

    dates = [x[0] for x in db_tuples]
    baselines = [x[1] for x in db_tuples]

    q = np.zeros([len(dates), len(dates)])

    for i, ds in enumerate(dates):
        t = np.arange(i - inps.threshold, i + inps.threshold)
        t = t[t >= 0]
        t = t[t < len(dates)]
        if len(t) > 0:
            for m in range(t[0], t[-1] + 1):
                q[i, m] = np.abs(baselines[i] - baselines[m])
                q[m, i] = q[i, m]
        
        #if len(t) > 3:
        #    ss = np.where(q[i, :] == np.max(q[i, :]))[0]
        #    q[i,ss]=0 

    X = csr_matrix(q)
    Tcsr = minimum_spanning_tree(X)
    A = Tcsr.toarray()
    
    A = q
    A[A>200] = 0

    for i in range(len(dates)):
        if len(np.nonzero(A[i,:])[0]) <= 1:
            A[i,:] = 0
            print(dates[i])
    
    A = np.triu(q) 
    
    ind1, ind2 = np.where(A > 0)
    intdates = ['{}_{}\n'.format(dates[g], dates[h]) for g, h in zip(ind1, ind2)]
    intdates_test = ['{}_{}, {}, {}, {}\n'.format(dates[g], dates[h], str(baselines[g]),
                                             str(baselines[h]), str(baselines[g] - baselines[h]))
                for g, h in zip(ind1, ind2)]

    #for x in intdates:
    #    print(x)

    with open(os.path.join(os.path.dirname(inps.out_file), 'test.txt'), 'w') as f:
        f.writelines(intdates_test)

    with open(inps.out_file, 'w') as f:
        f.writelines(intdates)

    return


if __name__ == '__main__':
    main()


