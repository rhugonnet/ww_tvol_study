"""
@author: hugonnet
retrieve stats of Cross (X), Along (A) and Jitter (J) corrections for Fig. S1
"""
from __future__ import print_function
from glob import glob
import os
import pandas as pd
import numpy as np
import csv

aster_dir = '/data/icesat/travail_en_cours/romain/data/dems/aster_corr/'
# arcticdem_dir = '/data/icesat/travail_en_cours/romain/data/dems/arcticdem/'
# rema_dir = '/data/icesat/travail_en_cours/romain/data/dems/rema_2m/'
# rema_dir_8 = '/data/icesat/travail_en_cours/romain/data/dems/rema_8m/'

list_aster_dem = glob(os.path.join(aster_dir,'**/*_final.zip'),recursive=True)

def read_rmse(fdir,subdir):

    fname = os.path.join(fdir,subdir,'stats.txt')

    if os.path.exists(fname):

        with open(fname, 'r') as f:
            lines = f.readlines()
        stats = lines[0].strip('[ ]\n').split(', ')
        after = [float(s) for s in lines[-1].strip('[ ]\n').split(', ')]

        stats_d = dict(zip(stats, after))

        return [stats_d['RMSE'], stats_d['COUNT']]
    else:
        return [np.nan,np.nan]

def read_cross_along_jitter_params(fdir,par_name):

        fn_par = os.path.join(fdir,par_name)

        if os.path.exists(fn_par):
            df = pd.read_csv(fn_par,sep=' ',header=None,index_col=None)
            # print(df.iloc[:,0].tolist())
            return df.iloc[:,0].tolist()
        else:
            return np.array([np.nan])

# #for rmse/count
outfile = '/data/icesat/travail_en_cours/romain/rmse_corrections.csv'

# list_stats_bef = [read_rmse(os.path.dirname(aster_dem),'coreg') for aster_dem in list_aster_dem]
# list_stats_aft = [read_rmse(os.path.dirname(aster_dem),'re-coreg') for aster_dem in list_aster_dem]
#
# zip_bef = list(zip(*list_stats_bef))
# zip_aft = list(zip(*list_stats_aft))
#
# df = pd.DataFrame()
# df = df.assign(rmse_bef=np.array(zip_bef[0]),count_bef=np.array(zip_bef[1])
#                ,rmse_aft=np.array(zip_aft[0]),count_aft=np.array(zip_aft[1]))
# df.to_csv(outfile)

#for params

params = ['params_AlongTrack_Jitter.txt', 'params_AlongTrack_SumofSines.txt', 'params_CrossTrack_Polynomial.txt']

list_stats_cross = [read_cross_along_jitter_params(os.path.dirname(aster_dem),params[2]) for aster_dem in list_aster_dem]
list_stats_along = [read_cross_along_jitter_params(os.path.dirname(aster_dem),params[1]) for aster_dem in list_aster_dem]
list_stats_jitter = [read_cross_along_jitter_params(os.path.dirname(aster_dem),params[0]) for aster_dem in list_aster_dem]

outfile_x = '/data/icesat/travail_en_cours/romain/cross_params.csv'
outfile_a = '/data/icesat/travail_en_cours/romain/along_params.csv'
outfile_j = '/data/icesat/travail_en_cours/romain/jitter_params.csv'

def write_list_to_csv(fn,list):
    with open(fn, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        for sublist in list:
            writer.writerow(sublist)

write_list_to_csv(outfile_x,list_stats_cross)
write_list_to_csv(outfile_a,list_stats_along)
write_list_to_csv(outfile_j,list_stats_jitter)
# np.savetxt(outfile_x,np.array([list_stats_cross],dtype=float),delimiter=',')
# np.savetxt(outfile_a,np.array(list_stats_along,dtype=float),delimiter=',')
# np.savetxt(outfile_j,np.array(list_stats_jitter,dtype=float),delimiter=',')
