#!/usr/bin/env python3
# Author: Sara Mirzaee

import os
import datetime
import numpy as np
import argparse


def cmd_line_parse(iargs=None):
    parser = argparse.ArgumentParser(description='find the minimum number of connected good interferograms')
    parser.add_argument('-b', '--baselineDir', dest='baseline_dir', type=str, help='baseline directory')
    parser.add_argument('-o', '--outFile', dest='out_file', type=str, default='./bestints.txt', help='output text file')
    inps = parser.parse_args(args=iargs)
    return inps


def main(iargs=None):
    inps = cmd_line_parse(iargs)

    bf = os.listdir(inps.baseline_dir)
    baselines = []
    for d in bf:
        with open(os.path.join(inps.baseline_dir, d), 'r') as f:
            lines = f.readlines()
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

    dates_val = [datetime.datetime.strptime(x, '%Y%m%d') for x in dates]

    date11 = []
    date22 = []
    bs_ifgs = []

    for i, ds in enumerate(dates_val):
        t = np.arange(i + 1, i + 6)
        t = t[t >= 0]
        t = t[t <= len(dates_val)]
        if len(t) > 0:
            for m in range(t[0], t[-1]):
                if np.abs(ds - dates_val[m]).days < 100:
                    date11.append(ds)
                    date22.append(dates_val[m])
                    bs_ifgs.append(baselines[i] - baselines[m])

    date1 = [x.strftime("%Y%m%d") for x in date11]
    date2 = [x.strftime("%Y%m%d") for x in date22]
    ifg_tuple = [(x, y) for x, y in zip(date11, date22)]

    intd1 = []
    intd2 = []
    intd = []
    ind_taken = []

    intdates = []

    for g, tdate in enumerate(dates_val):
        later_check = np.any(np.array(intd2) > tdate)
        if later_check:
           indi = list(np.where(np.array(date11) == tdate)[0]) + list(np.where(np.array(date22) == tdate)[0])
        else:
           indi = list(np.where(np.array(date11) == tdate)[0])

        indi = list(set(indi) - set(ind_taken))

        bsm = np.array([bs_ifgs[i] for i in indi])
        bsm_sort = np.sort(np.abs(bsm))
        if len(bsm) == 0:
            continue

        ind2 = int(np.where(np.abs(bsm) == bsm_sort[0])[0])
        ind = indi[ind2]

        i = 0
        while date22[ind] == tdate:
            i += 1
            if i == len(bsm_sort):
                break
            ind2 = int(np.where(np.abs(bsm) == bsm_sort[i])[0])
            ind = indi[ind2]

        stat = True

        for ifg in intd:
            if ifg[0] < date11[ind] and ifg[1] > date11[ind]:
                dist1 = baselines[dates_val.index(ifg[0])] - baselines[dates.index(date1[ind])]
                dist2 = baselines[dates.index(date2[ind])] - baselines[dates.index(date1[ind])]
                if abs(dist1) < abs(dist2):
                    ind = ifg_tuple.index((ifg[0], date11[ind]))
                    intd1.append(date11[ind])
                    intd2.append(date22[ind])
                    intd.append((date11[ind], date22[ind]))
                    ind_taken.append(ind)
                    #print(date1[ind], date2[ind], str(bs_ifgs[ind]))
                    #intdates.append(
                    #    '{}_{}, {}, {}, {}\n'.format(date1[ind], date2[ind], str(baselines[dates.index(date1[ind])]),
                    #                                 str(baselines[dates.index(date2[ind])]), str(bs_ifgs[ind])))
                    intdates.append('{}_{}\n'.format(date1[ind], date2[ind]))
                    stat = False
                    break

        if stat:
            intd1.append(date11[ind])
            intd2.append(date22[ind])
            intd.append((date11[ind], date22[ind]))
            ind_taken.append(ind)
            #print(date1[ind], date2[ind], str(bs_ifgs[ind]))
            #intdates.append('{}_{}, {}, {}, {}\n'.format(date1[ind], date2[ind], str(baselines[dates.index(date1[ind])]),
            #                                             str(baselines[dates.index(date2[ind])]), str(bs_ifgs[ind])))
            intdates.append('{}_{}\n'.format(date1[ind], date2[ind]))

    with open(inps.out_file, 'w') as f:
        f.writelines(intdates)

    return


if __name__ == '__main__':
    main()


