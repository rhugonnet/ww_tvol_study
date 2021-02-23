
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyddem.tdem_tools as tt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib.legend_handler import HandlerErrorbar, HandlerPatch, HandlerBase
import matplotlib.patches as mpatches
from glob import glob

in_ext = '/home/atom/ongoing/work_worldwide/tables/table_man_gard_zemp_wout.csv'
df_ext = pd.read_csv(in_ext)
fn_tarea = '/home/atom/data/inventory_products/RGI/tarea_zemp.csv'

reg_dir = '/home/atom/ongoing/work_worldwide/vol/reg'
list_fn_reg= [os.path.join(reg_dir,'dh_'+str(i).zfill(2)+'_rgi60_int_base_reg.csv') for i in [1,2,3,4,5,6,7,8,9,10,11,12,16,17,18,19]] + [os.path.join(reg_dir,'dh_13_14_15_rgi60_int_base_reg.csv')] + [os.path.join(reg_dir,'dh_01_02_rgi60_int_base_reg.csv')]

tlim_zemp = [np.datetime64('2006-01-01'),np.datetime64('2016-01-01')]
tlim_wouters = [np.datetime64('2002-01-01'),np.datetime64('2016-01-01')]
tlim_cira = [np.datetime64('2002-01-01'),np.datetime64('2020-01-01')]
tlim_gardner = [np.datetime64('2003-01-01'),np.datetime64('2009-01-01')]
# tlim_shean = [np.datetime64('2000-01-01'),np.datetime64('2018-01-01')]
# tlim_braun = [np.datetime64('2000-01-01'),np.datetime64('2013-01-01')]

list_tlim = [tlim_zemp,tlim_wouters,tlim_cira,tlim_gardner]
list_tag = ['zemp','wout','cira','gard']

list_df = []
for fn_reg in list_fn_reg:
    df_reg = pd.read_csv(fn_reg)
    df_agg = tt.aggregate_all_to_period(df_reg,list_tlim=list_tlim,fn_tarea=fn_tarea,frac_area=1,list_tag=list_tag)
    list_df.append(df_agg)

df = pd.concat(list_df)

fig = plt.figure(figsize=(17.5,7.75))
grid = plt.GridSpec(20, 17, wspace=0.25, hspace=0)

nb = 4 + 1

x_axis = np.arange(0,17*nb)

r_list = [1,2,3,4,5,6,7,8,9,10,11,12,21,16,17,18,19]

ax = fig.add_subplot(grid[:-2, :10])

shift_x=0
for i in range(len(r_list)):

    df_g = df[np.logical_and(df.tag=='gard',df.reg==r_list[i])]

    ax.fill_between([shift_x+x_axis[nb*i]-0.5,shift_x+x_axis[nb*i]+0.5],[df_ext.gar[i]+df_ext.gar_err[i]]*2,[df_ext.gar[i]-df_ext.gar_err[i]]*2,color='tab:red',alpha=0.45)
    ax.plot([shift_x+x_axis[nb*i]-0.5,shift_x+x_axis[nb*i]+0.5],[df_ext.gar[i]]*2,color='tab:red',lw=2)
    ax.errorbar(shift_x+x_axis[nb*i],df_g.dmdtda.values[0],2*df_g.err_dmdtda.values[0],fmt='o',color=plt.cm.Greys(0.8),capsize=3,zorder=30)

    df_z = df[np.logical_and(df.tag=='zemp',df.reg==r_list[i])]

    ax.fill_between([shift_x+x_axis[nb*i+3]-0.5,shift_x+x_axis[nb*i+3]+0.5],[df_ext.zemp[i]+df_ext.zemp_err[i]]*2,[df_ext.zemp[i]-df_ext.zemp_err[i]]*2,color='tab:orange',alpha=0.45)
    ax.plot([shift_x+x_axis[nb*i+3]-0.5,shift_x+x_axis[nb*i+3]+0.5],[df_ext.zemp[i]]*2,color='tab:orange',lw=2)
    ax.errorbar(shift_x+x_axis[nb*i+3],df_z.dmdtda.values[0],2*df_z.err_dmdtda.values[0],fmt='o',color=plt.cm.Greys(0.8),capsize=3,zorder=30)

    df_w = df[np.logical_and(df.tag=='wout',df.reg==r_list[i])]

    ax.fill_between([shift_x+x_axis[nb*i+1]-0.5,shift_x+x_axis[nb*i+1]+0.5],[df_ext.wout[i]+df_ext.wout_err[i]]*2,[df_ext.wout[i]-df_ext.wout_err[i]]*2,color='tab:cyan',alpha=0.45)
    ax.plot([shift_x+x_axis[nb*i+1]-0.5,shift_x+x_axis[nb*i+1]+0.5],[df_ext.wout[i]]*2,color='tab:cyan',lw=2)
    if ~np.isnan(df_ext.wout[i]):
        ax.errorbar(shift_x+x_axis[nb*i+1], df_w.dmdtda.values[0],2*df_w.err_dmdtda.values[0],fmt='o',color=plt.cm.Greys(0.8),capsize=3,zorder=30)

    df_c = df[np.logical_and(df.tag == 'cira', df.reg == r_list[i])]

    ax.fill_between([shift_x + x_axis[nb * i + 2] - 0.5, shift_x + x_axis[nb * i + 2] + 0.5],
                    [df_ext.cira[i] + df_ext.cira_err[i]] * 2, [df_ext.cira[i] - df_ext.cira_err[i]] * 2,
                    color='tab:purple', alpha=0.45)
    ax.plot([shift_x + x_axis[nb * i + 2] - 0.5, shift_x + x_axis[nb * i +2] + 0.5], [df_ext.cira[i]] * 2,
            color='tab:purple', lw=2)
    if ~np.isnan(df_ext.cira[i]):
        ax.errorbar(shift_x + x_axis[nb * i + 2], df_c.dmdtda.values[0], 2 * df_c.err_dmdtda.values[0], fmt='o',
                    color=plt.cm.Greys(0.8), capsize=3, zorder=30)

    # if r_list[i] == 21 or r_list[i] == 17 or r_list[i] == 16:
    #
    #     if r_list[i]==21:
    #         df_t = df[np.logical_and(df.tag=='shean',df.reg==r_list[i])]
    #         col = 'tab:purple'
    #         dmdtda = -0.19
    #         err_dmdtda= 0.03
    #     elif r_list[i]==17:
    #         df_t = df[np.logical_and(df.tag=='braun',df.reg==r_list[i])]
    #         col = 'tab:green'
    #         dmdtda = -0.64
    #         err_dmdtda=0.02
    #     elif r_list[i]==16:
    #         df_t = df[np.logical_and(df.tag=='braun',df.reg==r_list[i])]
    #         col = 'tab:green'
    #         dmdtda = -0.23
    #         err_dmdtda=0.04
    #
    #     ax.fill_between([shift_x + x_axis[4 * i + 3] - 0.5,shift_x+ x_axis[4 * i + 3] + 0.5],
    #                     [dmdtda + 2*err_dmdtda] * 2, [dmdtda -2*err_dmdtda] * 2,
    #                     color=col, alpha=0.45)
    #     ax.plot([shift_x + x_axis[4 * i + 3] - 0.5, shift_x+x_axis[4 * i + 3] + 0.5], [dmdtda] * 2, color=col,lw=2)
    #     ax.errorbar(shift_x + x_axis[4 * i + 3], df_t.dmdtda.values[0], 2 * df_t.err_dmdtda.values[0], fmt='o',
    #                      color=plt.cm.Greys(0.8), capsize=3, zorder=30)
    #     shift_x += 1



ax.vlines(np.arange(-1,17*5,5),-3,3,colors='grey',alpha=0.7,linewidth=0.75,linestyles='dashed')


ax.hlines(0,-5,17*5+5,linestyles='dashed',colors=plt.cm.Greys(0.9))

ticks = ['Alaska (01)','Western Canada\nand USA (02)','Arctic Canada\nNorth (03)','Arctic Canada\nSouth (04)','Greenland\nPeriphery (05)', 'Iceland (06)','Svalbard and\nJan Mayen (07)', 'Scandinavia (08)','Russian\nArctic (09)','North Asia (10)','Central\nEurope (11)','Caucasus and\nMiddle East (12)','High Mountain\nAsia (13-15)','Low\nLatitudes (16)','Southern\nAndes (17)','New\nZealand (18)','Antarctic and\nSubantarctic (19)']
ax.set_xticks(np.arange(1.5,17*5,5))
ax.set_xticklabels(ticks,rotation=90)
ax.set_ylabel('Specific mass change rate (m w.e yr$^{-1}$)')
ax.set_xlim([-1,17*5-1])
ax.set_ylim([-1.5,0.4])
ax.text(0.025, 0.95, 'a', transform=ax.transAxes, ha='left', va='top',fontweight='bold',fontsize=14)

p3 = ax.plot([], [], color='tab:red', linewidth=2)
p4 = ax.fill([], [], color='tab:red', alpha=0.45)
p5 = ax.plot([], [], color='tab:cyan', linewidth=2)
p6 = ax.fill([], [], color='tab:cyan', alpha=0.45)
p1 = ax.plot([], [], color='tab:orange', linewidth=2)
p2 = ax.fill([], [], color='tab:orange', alpha=0.45)
p7 = ax.plot([], [], color='tab:purple', linewidth=2)
p8 = ax.fill([], [], color='tab:purple', alpha=0.45)
# p9 = ax.plot([], [], color='tab:green', linewidth=2)
# p10 = ax.fill([], [], color='tab:green', alpha=0.45)
p0 = ax.errorbar([], [], [], fmt='o',
            color='black', capsize=3)

# ax.legend()

def make_legend_polygon(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    a=1.5*height
    p = mpatches.Polygon(np.array([[0,-a/2],[width,-a/2],[width,height+a/2],[0,height+a/2],[0,-a/2]]))
    return p


hm = {p0: HandlerErrorbar(xerr_size=0.9), p4[0]: HandlerPatch(patch_func=make_legend_polygon), p2[0]: HandlerPatch(patch_func=make_legend_polygon), p6[0]: HandlerPatch(patch_func=make_legend_polygon),p8[0]: HandlerPatch(patch_func=make_legend_polygon)}
ax.legend([(p3[0],p4[0]),(p5[0],p6[0]),(p7[0], p8[0]),(p1[0], p2[0]),p0], ['Gardner et al.'+'$^{5}$'+ ' (2003-2009):\ngravi., in-situ, elev. change','Wouters et al. '+'$^{19}$'+'(2002-2016):\ngravi.','Ciracì et al.'+'$^{20}$'+' (2002-2020):\ngravi.','Zemp et al.'+'$^{21}$'+' (2006-2016):\nin-situ, elev. change','This study (corresp. period):\nelev. change'],ncol=3,handlelength=1,framealpha=1,loc='upper right',labelspacing=0.5,handler_map=hm,borderpad=0.4)
# ax.yaxis.grid(True,linestyle='--')

reg_dir = '/home/atom/ongoing/work_worldwide/vol/reg'
list_fn_reg_multann = [os.path.join(reg_dir,'dh_'+str(i).zfill(2)+'_rgi60_int_base_reg_subperiods.csv') for i in np.arange(1,20)]
df_all = pd.DataFrame()
for fn_reg_multann in list_fn_reg_multann:
    df_all= df_all.append(pd.read_csv(fn_reg_multann))

tlims = [np.datetime64('20'+str(i).zfill(2)+'-01-01') for i in range(21)]

list_df_glob = []
list_df_per = []
for i in range(len(tlims)-1):
    period = str(tlims[i])+'_'+str(tlims[i+1])
    df_p = df_all[df_all.period==period]
    df_global = tt.aggregate_indep_regions(df_p)
    df_global['period']=period
    df_noperiph = tt.aggregate_indep_regions(df_p[~df_p.reg.isin([5,19])])
    df_noperiph['period']=period

    list_df_glob.append(df_global)
    list_df_per.append(df_noperiph)
df_glob = pd.concat(list_df_glob)
df_per = pd.concat(list_df_per)


df_g = df[df.tag=='gard']
df_g_glo = tt.aggregate_indep_regions(df_g)
df_g_per = tt.aggregate_indep_regions(df_g[~df_g.reg.isin([5, 19])])

df_z = df[df.tag=='zemp']
df_z_glo = tt.aggregate_indep_regions(df_z)
df_z_per = tt.aggregate_indep_regions(df_z[~df_z.reg.isin([5, 19])])

dmdtda_gard = -0.35
err_gard = 0.04

dmdtda_zemp = -0.48
err_zemp = 0.2

ax1 = fig.add_subplot(grid[:-2, 14:])

tlims = [np.datetime64('20'+str(i).zfill(2)+'-01-01') for i in range(21)]

list_df_glob = []
list_df_per = []
for i in range(len(tlims)-1):
    period = str(tlims[i])+'_'+str(tlims[i+1])
    df_p = df_all[df_all.period==period]
    df_global = tt.aggregate_indep_regions(df_p)
    df_global['period']=period
    df_noperiph = tt.aggregate_indep_regions(df_p[~df_p.reg.isin([5,19])])
    df_noperiph['period']=period

    list_df_glob.append(df_global)
    list_df_per.append(df_noperiph)
df_glob = pd.concat(list_df_glob)
df_per = pd.concat(list_df_per)
for i in range(len(tlims)-1):
    df_p = df_glob[df_glob.period==str(tlims[i])+'_'+str(tlims[i+1])]
    # ax1.errorbar(tlims[i]+(tlims[i+1]-tlims[i])/2,df_p.dmdtda.values[0],2*df_p.err_dmdtda.values[0],fmt='x',color=plt.cm.Blues(0.9),capsize=3,zorder=3,lw=0.5)
    ax1.fill_between([tlims[i],tlims[i+1]],[df_p.dmdtda.values[0]+2*df_p.err_dmdtda.values[0]]*2,[df_p.dmdtda.values[0]-2*df_p.err_dmdtda.values[0]]*2,color=plt.cm.Greys(0.9),alpha=0.3)
    ax1.plot([tlims[i],tlims[i+1]],[df_p.dmdtda.values[0]]*2,color=plt.cm.Greys(0.9),lw=1)

    # df_p = df_per[df_per.period==str(tlims[i])+'_'+str(tlims[i+1])]
    # ax1.errorbar(tlims[i]+(tlims[i+1]-tlims[i])/2,df_p.dmdtda.values[0],2*df_p.err_dmdtda.values[0],fmt='<',color=plt.cm.Purples(0.9),capsize=5,zorder=4)
    # ax1.fill_between([tlims[i],tlims[i+1]],[df_p.dmdtda.values[0]+2*df_p.err_dmdtda.values[0]]*2,[df_p.dmdtda.values[0]-2*df_p.err_dmdtda.values[0]]*2,color=plt.cm.Purples(0.9),alpha=0.5)
    # ax1.plot([tlims[i],tlims[i+1]],[df_p.dmdtda.values[0]]*2,color=plt.cm.Purples(0.9),lw=2,linestyle='dotted')

tlims = [np.datetime64('20'+str(5*i).zfill(2)+'-01-01') for i in range(5)]

list_df_glob = []
list_df_per = []
for i in range(len(tlims)-1):
    period = str(tlims[i])+'_'+str(tlims[i+1])
    df_p = df_all[df_all.period==period]
    df_global = tt.aggregate_indep_regions(df_p)
    df_global['period']=period
    df_noperiph = tt.aggregate_indep_regions(df_p[~df_p.reg.isin([5,19])])
    df_noperiph['period']=period

    list_df_glob.append(df_global)
    list_df_per.append(df_noperiph)
df_glob = pd.concat(list_df_glob)
df_per = pd.concat(list_df_per)
for i in range(len(tlims)-1):
    df_p = df_glob[df_glob.period==str(tlims[i])+'_'+str(tlims[i+1])]
    ax1.fill_between([tlims[i],tlims[i+1]],[df_p.dmdtda.values[0]+2*df_p.err_dmdtda.values[0]]*2,[df_p.dmdtda.values[0]-2*df_p.err_dmdtda.values[0]]*2,color=plt.cm.Greys(0.9),alpha=0.5)
    ax1.plot([tlims[i],tlims[i+1]],[df_p.dmdtda.values[0]]*2,color=plt.cm.Greys(0.9),lw=2)


ax1.fill_between(tlim_zemp,[dmdtda_zemp+err_zemp]*2,[dmdtda_zemp-err_zemp]*2,color='tab:orange',alpha=0.45)
ax1.plot(tlim_zemp, [dmdtda_zemp] * 2, color='tab:orange')
# ax1.errorbar(tlim_zemp[0]+(tlim_zemp[1]-tlim_zemp[0])/2,df_z_glo.dmdtda.values[0],2*df_z_glo.err_dmdtda.values[0],fmt='o',color=plt.cm.Greys(0.8),capsize=3,zorder=30)
ax1.fill_between(tlim_gardner,[dmdtda_gard+err_gard]*2,[dmdtda_gard-err_gard]*2,color='tab:red',alpha=0.45)
ax1.plot(tlim_gardner, [dmdtda_gard] * 2, color='tab:red')
# ax1.errorbar(tlim_gardner[0]+(tlim_gardner[1]-tlim_gardner[0])/2,df_g_glo.dmdtda.values[0],2*df_g_glo.err_dmdtda.values[0],fmt='o',color=plt.cm.Greys(0.8),capsize=3,zorder=30)


ax1.plot([np.datetime64('2000-01-01'),np.datetime64('2020-01-01')],[0,0],linestyle='dashed',color=plt.cm.Greys(0.9))

ax1.set_xlim((np.datetime64('2000-01-01'),np.datetime64('2020-01-01')))
ax1.set_xticks([np.datetime64('20'+str(5*i).zfill(2)+'-01-01') for i in range(5)])
ax1.set_xticklabels(['2020/00  ','2005','2010','2015','2020'])
ax1.set_ylim([-0.75,0.01])
ax1.set_yticks([])
ax.set_ylabel('Specific mass change rate (m w.e yr$^{-1}$)')
ax1.set_xlabel('Year\n(Global total)')


ax1 = fig.add_subplot(grid[:-2, 11:14])

dmdtda_gard = -0.42
err_gard = 0.05

dmdtda_zemp = -0.56
err_zemp = 0.04

dmdtda_wouters = -0.41
err_wouters = 0.07

dmdtda_cira = -0.58
err_cira = 0.12

tlims = [np.datetime64('20'+str(i).zfill(2)+'-01-01') for i in range(21)]

list_df_glob = []
list_df_per = []
for i in range(len(tlims)-1):
    period = str(tlims[i])+'_'+str(tlims[i+1])
    df_p = df_all[df_all.period==period]
    df_global = tt.aggregate_indep_regions(df_p)
    df_global['period']=period
    df_noperiph = tt.aggregate_indep_regions(df_p[~df_p.reg.isin([5,19])])
    df_noperiph['period']=period

    list_df_glob.append(df_global)
    list_df_per.append(df_noperiph)
df_glob = pd.concat(list_df_glob)
df_per = pd.concat(list_df_per)
for i in range(len(tlims)-1):
    df_p = df_per[df_per.period==str(tlims[i])+'_'+str(tlims[i+1])]
    ax1.fill_between([tlims[i],tlims[i+1]],[df_p.dmdtda.values[0]+2*df_p.err_dmdtda.values[0]]*2,[df_p.dmdtda.values[0]-2*df_p.err_dmdtda.values[0]]*2,color=plt.cm.Greys(0.9),alpha=0.3)
    ax1.plot([tlims[i],tlims[i+1]],[df_p.dmdtda.values[0]]*2,color=plt.cm.Greys(0.9),lw=1)

tlims = [np.datetime64('20'+str(5*i).zfill(2)+'-01-01') for i in range(5)]

list_df_glob = []
list_df_per = []
for i in range(len(tlims)-1):
    period = str(tlims[i])+'_'+str(tlims[i+1])
    df_p = df_all[df_all.period==period]
    df_global = tt.aggregate_indep_regions(df_p)
    df_global['period']=period
    df_noperiph = tt.aggregate_indep_regions(df_p[~df_p.reg.isin([5,19])])
    df_noperiph['period']=period

    list_df_glob.append(df_global)
    list_df_per.append(df_noperiph)
df_glob = pd.concat(list_df_glob)
df_per = pd.concat(list_df_per)
for i in range(len(tlims)-1):
    df_p = df_per[df_per.period==str(tlims[i])+'_'+str(tlims[i+1])]
    ax1.fill_between([tlims[i],tlims[i+1]],[df_p.dmdtda.values[0]+2*df_p.err_dmdtda.values[0]]*2,[df_p.dmdtda.values[0]-2*df_p.err_dmdtda.values[0]]*2,color=plt.cm.Greys(0.9),alpha=0.5)
    ax1.plot([tlims[i],tlims[i+1]],[df_p.dmdtda.values[0]]*2,color=plt.cm.Greys(0.9),lw=2)


ax1.fill_between(tlim_cira,[dmdtda_cira+err_cira]*2,[dmdtda_cira-err_cira]*2,color='tab:purple',alpha=0.45,zorder=3)
ax1.plot(tlim_cira, [dmdtda_cira] * 2, color='tab:purple',zorder=3)

ax1.fill_between(tlim_wouters,[dmdtda_wouters+err_wouters]*2,[dmdtda_wouters-err_wouters]*2,color='tab:cyan',alpha=0.45,zorder=2)
ax1.plot(tlim_wouters, [dmdtda_wouters] * 2, color='tab:cyan',zorder=2)
# ax1.errorbar(tlim_zemp[0]+(tlim_zemp[1]-tlim_zemp[0])/2,df_z_glo.dmdtda.values[0],2*df_z_glo.err_dmdtda.values[0],fmt='o',color=plt.cm.Greys(0.8),capsize=3,zorder=30)
ax1.fill_between(tlim_gardner,[dmdtda_gard+err_gard]*2,[dmdtda_gard-err_gard]*2,color='tab:red',alpha=0.45,zorder=4)
ax1.plot(tlim_gardner, [dmdtda_gard] * 2, color='tab:red',zorder=4)
# ax1.errorbar(tlim_gardner[0]+(tlim_gardner[1]-tlim_gardner[0])/2,df_g_glo.dmdtda.values[0],2*df_g_glo.err_dmdtda.values[0],fmt='o',color=plt.cm.Greys(0.8),capsize=3,zorder=30)
ax1.fill_between(tlim_zemp,[dmdtda_zemp+err_zemp]*2,[dmdtda_zemp-err_zemp]*2,color='tab:orange',alpha=0.45,zorder=5)
ax1.plot(tlim_zemp, [dmdtda_zemp] * 2, color='tab:orange',zorder=5)

ax1.plot([np.datetime64('2000-01-01'),np.datetime64('2020-01-01')],[0,0],linestyle='dashed',color=plt.cm.Greys(0.9))

ax1.set_xlim((np.datetime64('2000-01-01'),np.datetime64('2020-01-01')))
ax1.set_ylim([-0.75,0.01])
ax1.set_xticks([np.datetime64('20'+str(5*i).zfill(2)+'-01-01') for i in range(5)])
ax1.set_xticklabels(['2000','2005','2010','2015',''])
# ax.set_ylabel('Specific mass change (m w.e yr$^{-1}$)')
ax1.set_xlabel('Year\n(Global excl. Greenland Periphery\n and Antarctic and Subantarctic)')
ax1.set_ylabel('Specific mass change rate (m w.e yr$^{-1}$)')
ax1.text(0.075, 0.95, 'b', transform=ax1.transAxes, ha='left', va='top',fontweight='bold',fontsize=14)

p10 = ax.plot([], [], color=plt.cm.Greys(0.9), linewidth=1)
p11 = ax.fill([], [], color=plt.cm.Greys(0.9), alpha=0.3)

p12= ax.plot([], [], color=plt.cm.Greys(0.9), linewidth=2)
p13 = ax.fill([], [], color=plt.cm.Greys(0.9), alpha=0.5)

hm = {p0: HandlerErrorbar(xerr_size=0.9), p4[0]: HandlerPatch(patch_func=make_legend_polygon), p2[0]: HandlerPatch(patch_func=make_legend_polygon), p6[0]: HandlerPatch(patch_func=make_legend_polygon),p8[0]: HandlerPatch(patch_func=make_legend_polygon), p13[0]: HandlerPatch(patch_func=make_legend_polygon), p11[0]: HandlerPatch(patch_func=make_legend_polygon)}
ax1.legend([(p3[0],p4[0]),(p5[0],p6[0]),(p7[0], p8[0]),(p1[0], p2[0]),(p10[0],p11[0]),(p12[0],p13[0])], ['Gardner et al.'+'$^{5}$','Wouters et al.'+'$^{19}$','Ciracì et al.'+'$^{20}$','Zemp et al.'+'$^{21}$','This study (yearly)','This study (5-year)'],handlelength=1,framealpha=1,loc=(0.35,0.815),ncol=2,labelspacing=1,handler_map=hm,borderpad=0.6)


plt.savefig('/home/atom/ongoing/work_worldwide/figures/revised/Figure_3_rev2.png',dpi=400)