"""
@author: hugonnet
Python script to check that aster l1a downloads errors and possibly regenerate a .csv list of Pulldirs to re-download

unfortunately we need to download l1a recursively on a https structure and if possible in parallel:
a lot easier with wget than curl, but checking errors is a bit tedious.
TODO: one solution would be to call from Python multiprocessing and check errors: mp wrapper commented at end of file
"""

from __future__ import print_function
import os
import csv


def check_wget_log(log_dir, out_csv):
    list_stderr = [os.path.join(log_dir, subdir, 'stderr') for subdir in os.listdir(log_dir)]
    list_fail = []
    for stderr in list_stderr:
        print('File:' + stderr)
        chk = 0
        with open(stderr) as s:
            text = s.readlines()

            for line in text:
                # if 'error' in line or 'Error' in line or 'ERROR' in line:
                if ('ERROR 404' not in line and 'robots.txt' not in line) and (
                        'error' in line or 'Error' in line or 'ERROR' in line or 'fail' in line or 'Fail' in line or 'FAIL' in line):
                    print(text.index(line))
                    print(line)

                    chk = 1

        if chk == 1:
            tmp_fail = os.path.split(stderr)[-2][-15:]
            list_fail.append(tmp_fail)

    print('List of failed downloads')
    print(list_fail)
    print('Representing ' + str(len(list_fail)))
    print('Out of:' + str(len(list_stderr)))

    list_Pulldirs = ['https://e4ftl01.cr.usgs.gov/PullDir/' + fail for fail in list_fail]

    with open(out_csv, 'wd') as file:
        writer = csv.writer(file, delimiter=',')
        for pdir in list_Pulldirs:
            writer.writerow([pdir])


if __name__ == '__main__':
    # if using parallel --results for log, path to log is: download_dir/1/
    download_dir = '/home/hugonnet/ww_l1a/'
    out_csv = '/home/hugonnet/worldwide/list_failed_pulldirs.csv'
    check_wget_log(os.path.join(download_dir, '1'), out_csv)

# wget_l1a + check_l1a could be merged in Python doing: (to tedious to check wget errors in bash)
#
# from subprocess import Popen
# import sys, os, shutil
# import time
# http_dir = ''
# tgt_dir = ''
#
# check = True
#     u=0
#     while check and u<5:
#         u=u+1
#         #if download already failed, sleep 10min before trying again
#         #also need to remove what was downloaded otherwise some error just loop back
#         if os.path.exists(tgt_dir)
#             shutil.rmtree(tgt_dir)
#         if u>1:
#             time.sleep(600)
#         #download strip data
#         cmd = 'wget -r -N -nH -np -R index.html* --cut-dirs=6 '+ http_dir +' -P '+tgt_dir
#         log = open(wget_log_file,'w')
#         p = Popen(cmd,stdout=log, stderr=log,shell=True)
#         p.wait()
#         log.close()
#         check = not check_wget(wget_log_file)
#
#     #if download keeps failing for some reason, abandon and append to specific log of failed downloads
#     if u==5:
#         print('Downloading for '+arc_tile_name+' failed.')
#         with open(wget_fail,'a') as fail:
#             fail.write(arc_tile_name+'\n')
