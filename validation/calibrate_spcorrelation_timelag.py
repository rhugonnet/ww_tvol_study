"""
@author: hugonnet
sample spatial correlations from differences to ICESat with varying time lags
"""

import os
import numpy as np
import pandas as pd
from glob import glob
import pyddem.spstats_tools as spt
from pybob.ddem_tools import nmad
import scipy.optimize

# dir_valid = '/home/atom/ongoing/work_worldwide/validation/icesat/'
dir_valid = '/data/icesat/travail_en_cours/romain/results/validation3/'

list_fn_valid = glob(os.path.join(dir_valid,'*.csv'),recursive=True)
list_fn_valid = sorted(list_fn_valid)

# outfile = '/home/atom/ongoing/work_worldwide/validation/tinterp_corr.csv'
outfile = '/data/icesat/travail_en_cours/romain/results/tinterp_corr_deseas.csv'

df = pd.DataFrame()
for fn_valid in list_fn_valid:
    tmp_df = pd.read_csv(fn_valid)
    tmp_df = tmp_df.assign(reg=int(os.path.basename(fn_valid).split('_')[2]))
    df = df.append(tmp_df)

df = df[df.pos==2]

# df = df[df.reg==1]

def bin_dh_zsc_by_vals(dh,zsc,bins,bins_val):

    med_dh = []
    nmad_dh = []
    med_zsc = []
    nmad_zsc = []
    mid_bin = []
    for i in range(len(bins)-1):
        ind = np.logical_and(bins_val >= bins[i],bins_val < bins[i+1])
        if len(ind)>100:
            med_dh.append(np.nanmedian(dh[ind]))
            nmad_dh.append(nmad(dh[ind]))
            # nmad_dh.append(np.nanstd(dh[ind]))
            med_zsc.append(np.nanmedian(zsc[ind]))
            nmad_zsc.append(nmad(zsc[ind]))
            # nmad_zsc.append(np.nanstd(zsc[ind]))

            mid_bin.append(bins[i] + 0.5*(bins[i+1]-bins[i]))

    return [np.array(mid_bin), np.array(med_dh), np.array(nmad_dh), np.array(med_zsc), np.array(nmad_zsc)]


def bin_dh_zsc_by_season(dh,zsc,date):

    season_month_bins = np.arange(1,13,1)

    mon = pd.DatetimeIndex(date).month.values

    med_dh = []
    nmad_dh = []
    med_zsc = []
    nmad_zsc = []
    mid_bin = []
    for i in range(len(season_month_bins)):
        ind = (mon == season_month_bins[i])
        if np.count_nonzero(ind)>0:
        # ind = np.logical_and(mon >= season_month_bins[i], mon < season_month_bins[i + 1])
            med_dh.append(np.nanmedian(dh[ind]))
            nmad_dh.append(nmad(dh[ind]))
            med_zsc.append(np.nanmedian(zsc[ind]))
            nmad_zsc.append(nmad(zsc[ind]))

            mid_bin.append(season_month_bins[i])

    return [np.array(mid_bin), np.array(med_dh), np.array(nmad_dh), np.array(med_zsc), np.array(nmad_zsc)]


for reg in list(set(list(df.reg))):
    df_reg = df[df.reg == reg]
    t0_n = bin_dh_zsc_by_season(df_reg.dh, df_reg.dh_ref, df_reg.t)
    coefs1, _ = scipy.optimize.curve_fit(lambda t, a, b, c: a ** 2 * np.sin(t * 2 * np.pi / 12 + c) + b, t0_n[0][:-1],
                                     t0_n[3][:-1])

    season_month_bins = np.arange(1, 13, 1)
    mon = pd.DatetimeIndex(df.t).month.values
    for i in range(len(season_month_bins)):
        ind = np.logical_and(mon == season_month_bins[i],df.reg==reg)
        df.dh[ind]-=coefs1[0] ** 2 * np.sin(season_month_bins[i] * 2 * np.pi / 12 + coefs1[2]) + coefs1[1]


sl_s = bin_dh_zsc_by_vals(df.zsc, df.dh,np.arange(0,60,5),df.slp)
nmad_zsc = nmad(df.zsc)
# nmad_dh = nmad(df.dh)
df_corr = df.copy()
vec_slope = np.arange(0,60,5)
for i in np.arange(len(vec_slope)-1):
    ind = np.logical_and(df_corr.slp>=vec_slope[i],df_corr.slp<vec_slope[i+1])
    df_corr[ind].zsc = df_corr[ind].zsc/sl_s[2][i]
    # df_corr.dh[ind] = df_corr.dh[ind]/sl_s[4][i] * nmad_dh

df = df[np.abs(df_corr.zsc) < 3*nmad_zsc]

nmad_dh = nmad(df.dh)
df = df[np.abs(df.dh) < 3*nmad_dh]

nproc=64

spt.get_tinterpcorr(df,outfile,nproc=nproc,nmax=5000)
spt.aggregate_tinterpcorr(outfile)


