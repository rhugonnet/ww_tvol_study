
import os, sys
import pandas as pd
import numpy as np

fn_stable = '/home/atom/ongoing/work_worldwide/validation/compiled/valid_compil_stable/valid_ICS_IB_seas_corr_final_weight.csv'
fn_glacier = '/home/atom/ongoing/work_worldwide/validation/compiled/seas_corr/valid_ICS_IB_seas_corr_final_weight.csv'


df_sta = pd.read_csv(fn_stable)
df_gla = pd.read_csv(fn_glacier)

df_sta = df_sta[df_sta['var']=='dh']
df_gla = df_gla[df_gla['var']=='dh']

df_sta = df_sta[df_sta.seas_dec==1]
df_gla = df_gla[df_gla.seas_dec==1]


sum_dh = []
for r in sorted(list(set(list(df_sta.reg)))):
    df_reg = df_sta[df_sta.reg==r]
    if r>15:
        summer_dh = df_reg.amp.values[0] * np.sin( (2.5+df_reg.phase.values[0])* 2 * np.pi / 12) + df_reg.h_shift.values[0]
    else:
        summer_dh = df_reg.amp.values[0] * np.sin( (8.5+df_reg.phase.values[0])* 2 * np.pi / 12) + df_reg.h_shift.values[0]
    sum_dh.append(summer_dh)
df_sta.med_dh = sum_dh

sum_dh = []
for r in sorted(list(set(list(df_gla.reg)))):
    df_reg = df_gla[df_gla.reg==r]
    if r>15:
        summer_dh = df_reg.amp.values[0] * np.sin( (2.5+df_reg.phase.values[0])* 2 * np.pi / 12) + df_reg.h_shift.values[0]
    else:
        summer_dh = df_reg.amp.values[0] * np.sin( (8.5+df_reg.phase.values[0])* 2 * np.pi / 12) + df_reg.h_shift.values[0]
    sum_dh.append(summer_dh)
df_gla.med_dh = sum_dh


fn_stable_2 = '/home/atom/ongoing/work_worldwide/validation/compiled/valid_compil_stable/valid_ICS_IB_all_bins_final_weight.csv'
fn_glacier_2 = '/home/atom/ongoing/work_worldwide/validation/compiled/seas_corr/valid_ICS_IB_all_bins_final_weight.csv'

df_gla_2 = pd.read_csv(fn_glacier_2)
df_sta_2 = pd.read_csv(fn_stable_2)

df_gla_2 = df_gla_2[np.logical_and(df_gla_2.type=='reg',df_gla_2.seas_corr==1)]
df_sta_2 = df_sta_2[np.logical_and(df_sta_2.type=='reg',df_sta_2.seas_corr==1)]

column_amp = []
column_phase = []
column_summer_bias = []
column_nb_is = []
column_nb_ib = []
column_residual = []
column_zsc_std = []
column_region = []

for r in sorted(list(set(list(df_gla.reg)))):
    df_g = df_gla[df_gla.reg==r]
    df_s = df_sta[df_sta.reg==r]
    df_g2 = df_gla_2[df_gla_2.mid_bin==str(r+0.5)]
    df_s2 = df_sta_2[df_sta_2.mid_bin==str(r+0.5)]

    column_region.append(str(r).zfill(2))
    column_amp.append('{:.1f}'.format(df_g.amp.values[0])+'\n('+'{:.1f}'.format(df_s.amp.values[0])+')')
    column_phase.append('{:.1f}'.format(df_g.phase.values[0])+'\n('+'{:.1f}'.format(df_s.phase.values[0])+')')
    column_summer_bias.append('{:.1f}'.format(df_g.med_dh.values[0])+'\n('+'{:.1f}'.format(df_s.med_dh.values[0])+')')
    column_nb_is.append('{:,.0f}'.format(df_g2.ns_ics.values[0])+'\n(''{:,.0f}'.format(df_s2.ns_ics.values[0])+')')
    column_nb_ib.append('{:,.0f}'.format(df_g2.ns_ib.values[0])+'\n('+'{:,.0f}'.format(df_s2.ns_ib.values[0])+')')
    column_residual.append('{:.3f}'.format(df_g2.dzsc_dt.values[0]*df_g2.nmad_dh.values[0])+'±'+'{:.3f}'.format(df_g2.dzsc_dt_2std.values[0]*df_g2.nmad_dh.values[0])+'\n('+'{:.3f}'.format(df_s2.dzsc_dt.values[0]*df_s2.nmad_dh.values[0])+'±'+'{:.3f}'.format(df_s2.dzsc_dt_2std.values[0]*df_s2.nmad_dh.values[0])+')')
    column_zsc_std.append('{:.2f}'.format(df_g2.nmad_zsc.values[0])+'\n('+'{:.2f}'.format(df_s2.nmad_zsc.values[0])+')')

df_out = pd.DataFrame()
df_out['Region'] = column_region
df_out['ICESat points'] = column_nb_is
df_out['IceBridge points'] = column_nb_ib
df_out['Amplitude (m)'] = column_amp
df_out['Phase (decimal month)'] = column_phase
df_out['Summer vertical bias (m)'] = column_summer_bias
df_out['Elevation change bias (m yr-1)'] = column_residual
df_out['Standardized uncertainty'] = column_zsc_std

df_out.to_csv('/home/atom/ongoing/work_worldwide/tables/Table_SuppValidation.csv')
