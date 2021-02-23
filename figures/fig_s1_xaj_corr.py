
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
from pybob.ddem_tools import nmad
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

fn_x = '/home/atom/ongoing/work_worldwide/coverage/cross_params.csv'
fn_a = '/home/atom/ongoing/work_worldwide/coverage/along_params.csv'
fn_j = '/home/atom/ongoing/work_worldwide/coverage/jitter_params.csv'

df = pd.read_csv(fn_x,header=None)

fig = plt.figure(figsize=(12,9))

grid = plt.GridSpec(5, 3, wspace=0.3, hspace=0.8)

ax = fig.add_subplot(grid[:2,:1])

xx = np.arange(0,61000,1000)

med_pars = np.nanmedian(df.iloc[:,:],axis=0)

# #cross-track
# xx_list = []
# for i in np.arange(len(df)):
#
#     params = df.iloc[i,:]
#     if ~np.isnan(params[0]):
#         params[np.isnan(params)] = 0
#         xx_poly = sum([p * (np.divide(xx, 1000) ** i) for i, p in enumerate(params)])
#         xx_poly = xx_poly - np.nanmedian(xx_poly)
#         xx_list.append(xx_poly)
#
# xx_arr = np.array(xx_list)
# xx_med = np.nanmedian(xx_arr,axis=0)
# plt.plot(xx,xx_med)
# # xx_nmad = np.array(np.shape(xx))
# for i in np.arange(len(xx)):
#     nmad_tmp = nmad(xx_arr[:,i])
#     xx_arr[:,i][np.abs(xx_arr[:,i]-xx_med[i])>5*nmad_tmp]=np.nan
# plt.fill_between(xx,np.nanpercentile(xx_arr,16,axis=0),np.nanpercentile(xx_arr,84,axis=0),alpha=0.5)
xx_poly = sum([p * (np.divide(xx, 1000) ** i) for i, p in enumerate(med_pars)])
ax.plot(xx,xx_poly,label='Polynomial of degree 6',color='black')
ax.set_xlabel('Cross track distance (m)')
ax.set_ylabel('Elevation (m)')
ax.legend()
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_locator(MultipleLocator(20000))
ax.text(0.066, 0.95, 'a', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')
ax.set_ylim([-6,9])
ax.grid()

xa = np.arange(0,180200,200)
# xb = np.flip(xa)
dfa = pd.read_csv(fn_a,header=None)

med_pars = np.nanmean(dfa.iloc[:,:],axis=0)

#
# xa_list = []
# for i in np.arange(100):
#
#     params = dfa.iloc[i,:]
#
#     if ~np.isnan(params[0]):
#
#         p = np.squeeze(np.asarray(med_pars))
#         aix = np.arange(0, p.size, 6)
#         bix = np.arange(1, p.size, 6)
#         cix = np.arange(2, p.size, 6)
#         dix = np.arange(3, p.size, 6)
#         eix = np.arange(4, p.size, 6)
#         fix = np.arange(5, p.size, 6)
#
#         xa_poly = np.sum(p[aix] * np.sin(np.divide(2 * np.pi, p[bix]) *
#                                        np.divide(xa[:, np.newaxis], 1000) +
#                                        p[cix]) + p[dix] * np.sin(np.divide(2 * np.pi, p[eix]) *
#                                                                  np.divide(xa[:, np.newaxis], 1000) + p[fix]), axis=1)
#         # xa_poly = xa_poly - np.nanmedian(xa_poly)
#         # xa_list.append(xa_poly)
#
# xa_arr = np.array(xa_list)
# xa_med = np.nanmedian(xa_arr,axis=0)

p = np.squeeze(np.asarray(med_pars))
aix = np.arange(0, p.size, 6)
bix = np.arange(1, p.size, 6)
cix = np.arange(2, p.size, 6)
dix = np.arange(3, p.size, 6)
eix = np.arange(4, p.size, 6)
fix = np.arange(5, p.size, 6)

freqs = p[aix] * np.sin(np.divide(2 * np.pi, p[bix]) *
                                       np.divide(xa[:, np.newaxis], 1000) +
                                       p[cix]) + p[dix] * np.sin(np.divide(2 * np.pi, p[eix]) *
                                                                 np.divide(xa[:, np.newaxis], 1000) + p[fix])

ax2 = fig.add_subplot(grid[:2,1:])

ax2.plot(xa, freqs[:,0],label='Long-range undulation: 1st freq.',color='tab:cyan')
# plt.plot(xa, freqs[:,1])
ax2.plot(xa, freqs[:,2],label='Long-range undulation: 2nd freq.',color='tab:green')
# plt.plot(xa, freqs[:,3])


# xx_nmad = np.array(np.shape(xx))
# for i in np.arange(len(xa)):
#
#     nmad_tmp = nmad(xa_arr[:,i])
#     xa_arr[:,i][np.abs(xa_arr[:,i]-xa_med[i])>10*nmad_tmp]=np.nan
#
# plt.fill_between(xa,np.nanpercentile(xa_arr,16,axis=0),np.nanpercentile(xa_arr,84,axis=0),alpha=0.5)


xa = np.arange(0,180200,200)
dfj = pd.read_csv(fn_j,header=None)

med_pars = np.nanmean(dfj.iloc[:,:],axis=0)

p = np.squeeze(np.asarray(med_pars))
aix = np.arange(0, p.size, 6)
bix = np.arange(1, p.size, 6)
cix = np.arange(2, p.size, 6)
dix = np.arange(3, p.size, 6)
eix = np.arange(4, p.size, 6)
fix = np.arange(5, p.size, 6)

freqs = p[aix] * np.sin(np.divide(2 * np.pi, p[bix]) *
                                       np.divide(xa[:, np.newaxis], 1000) +
                                       p[cix]) + p[dix] * np.sin(np.divide(2 * np.pi, p[eix]) *
                                                                 np.divide(xa[:, np.newaxis], 1000) + p[fix])

ax2.plot(xa, freqs[:,0],label='Jitter: 1st freq.',color='tab:blue')
# plt.plot(xa, freqs[:,1])
ax2.plot(xa, freqs[:,2],label='Jitter: 2nd freq.',color='tab:orange')
# plt.plot(xa, freqs[:,3])
ax2.set_xlabel('Along track distance (m)')
ax2.set_ylabel('Elevation (m)')
ax2.legend(ncol=2,loc='lower right')
ax2.text(0.033, 0.95, 'b', transform=ax2.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')
ax2.yaxis.set_major_locator(MultipleLocator(5))
ax2.xaxis.set_major_locator(MultipleLocator(30000))
ax2.grid()


in_csv = '/home/atom/ongoing/work_worldwide/coverage/rmse_corrections.csv'

df = pd.read_csv(in_csv)

df = df[np.logical_and(df.rmse_bef<25,df.rmse_aft<25)]
#
# fig = plt.figure(figsize=(16,9))
#
# plt.hist(df.rmse_aft-df.rmse_bef,100,alpha=0.8,label='before')
# # plt.xlim([0,20])
# plt.legend()

bin_count = [100,1000,10000,100000,1000000,10000000]

ax3 = fig.add_subplot(grid[2:,:])

cols = ['r','b','k','y','g','m','r','b']

# function for setting the colors of the box plots pairs
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

data_bef = []
data_aft = []
nb = []
for i in np.arange(len(bin_count)-1):

    df_tmp = df[np.logical_and(df.count_aft>bin_count[i],df.count_aft<=bin_count[i+1])]

    data_bef.append(df_tmp.rmse_bef)
    data_aft.append(df_tmp.rmse_aft)
    nb.append(len(df_tmp))
    # if i == 1:
    #     plt.scatter(np.median(df_tmp.rmse_bef),bin_count[i],marker='o',color='black',label='Before corrections')
    #     plt.scatter(np.median(df_tmp.rmse_aft),bin_count[i],marker='s',color='black',label='After corrections')
    # else:
    #     plt.scatter(np.median(df_tmp.rmse_bef),bin_count[i],marker='o',color='black')
    #     plt.scatter(np.median(df_tmp.rmse_aft),bin_count[i],marker='s',color='black')
    plt.text(7,2*i+2.5,'RMSE improvement: '+str(int((np.median(df_tmp.rmse_bef)-np.median(df_tmp.rmse_aft))*10)/10.)+' m ; Number of ASTER DEMs: '+str(len(df_tmp)))
bp_bef= ax3.boxplot(data_bef,positions=np.array(range(len(data_bef)))*2.0+1.8,sym='',widths=0.2,vert=False,whis=[10,90])
bp_aft = ax3.boxplot(data_aft,positions=np.array(range(len(data_aft)))*2.0+2.2,sym='',widths=0.2,vert=False,whis=[10,90])
set_box_color(bp_bef,'tab:blue')
set_box_color(bp_aft,'tab:orange')

ax3.plot([], c='tab:blue', label='Before corrections')
ax3.plot([], c='tab:orange', label='After corrections')
ax3.text(0.025, 0.95, 'c', transform=ax3.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')
# ticks = [str(bin_count[i])+'-'+str(bin_count[i+1]) for i in range(len(bin_count)-1)]
ticks = ['10-10$^2$','10$^2$-10$^3$','10$^3$-10$^4$','10$^4$-10$^5$','10$^5$-10$^6$']
ax3.set_ylim(1,11)
ax3.set_yticklabels(ticks)
ax3.set_xlabel('RMSE of elevation differences on stable terrain (m)')
ax3.set_ylabel('Number of valid points for corrections')
# plt.grid()
ax3.xaxis.grid(True,linestyle='--')
ax3.legend()

plt.savefig('/home/atom/ongoing/work_worldwide/figures/revised/Figure_S1.png',dpi=400)