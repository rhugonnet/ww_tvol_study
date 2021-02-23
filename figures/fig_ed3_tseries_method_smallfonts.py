
import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 5})
plt.rcParams.update({'lines.linewidth':0.35})
plt.rcParams.update({'axes.linewidth':0.35})
plt.rcParams.update({'lines.markersize':2.5})
plt.rcParams.update({'axes.labelpad':1.5})

fig = plt.figure(figsize=(7.2,6))

grid = plt.GridSpec(18, 10, wspace=4, hspace=15)

ax = fig.add_subplot(grid[:9, :5])
ax.text(0.025, 0.966, 'a', transform=ax.transAxes,
        fontsize=8, fontweight='bold', va='top', ha='left')

var_dir = '/home/atom/ongoing/work_worldwide/variance'
region_list = os.listdir(var_dir)

region_nmad = []
region_nsamp = []
for region in region_list:

    list_fn_csv = [os.path.join(var_dir,region,f) for f in os.listdir(os.path.join(var_dir,region))]

    list_nmad = []
    list_nsamp = []
    for fn_csv in list_fn_csv:

        df = pd.read_csv(fn_csv)

        list_nmad.append(df.nmad.values)
        list_nsamp.append(df.nsamp.values)

    nmad_all = np.stack(list_nmad,axis=1)
    nsamp_all = np.stack(list_nsamp,axis=1)

    nan_mask = np.all(np.logical_or(np.isnan(nmad_all),nmad_all==0),axis=1)
    nmad_final = np.nansum(nmad_all * nsamp_all,axis=1) / np.nansum(nsamp_all,axis=1)
    nsamp_final = np.nansum(nsamp_all,axis=1)

    nmad_final[nan_mask] = np.nan
    nsamp_final[nan_mask] = 0

    region_nmad.append(nmad_final)
    region_nsamp.append(nsamp_final)


# ax.figure(figsize=(16,9))

slope = df.bin_slope.values
corr = df.bin_corr.values
bin_slope = sorted(list(set(list(slope))))
bin_corr = sorted(list(set(list(corr))))
nb_slope = len(bin_slope)
nb_corr = len(bin_corr)

color_list = ['tab:orange','tab:blue','tab:olive','tab:cyan','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive']
ls_list = ['solid','dashed','dotted']

# model_var = np.sqrt(3**2 + (20 * np.tan(np.array(5) * np.pi / 180))**2) + (((100-np.array(bin_corr))/100)*20)**1.25
#
# for i in range(len(region_nmad)):
#     i = 0
#     for j in range(nb_slope-2):
#
#         nmad = region_nmad[i]
#
#         ax.plot(corr[1:nb_corr],nmad[j*nb_corr+1:j*nb_corr+nb_corr],label='Slope category: '+str(bin_slope[j]-5)+'-'+str(bin_slope[j]+5)+' degrees',color=color_list[j],linestyle=ls_list[i])
#
#
# # ax.plot(bin_corr,model_var,label='model',linewidth=2)
#
# ax.xlabel('Correlation (percent)')
# ax.ylabel('Stable terrain NMAD (m)')
# ax.ylim([0,50])
# ax.legend()
#

x_slope = np.arange(5,45,0.1)

model_var = np.sqrt(3**2 + (40 * np.tan(np.array(x_slope) * np.pi / 180))**2.5 + (((100-np.array(50))/100)*20)**2)
i=0
# for i in range(len(region_nmad)-1):

u=0
for j in np.arange(1,nb_corr,2):

    nmad = region_nmad[i]

    # ax.plot(bin_slope,nmad[np.arange(j,len(slope),nb_corr)],label='region: '+region_list[i]+', corr: '+str(bin_corr[j]),color=color_list[j],linestyle=ls_list[i])
    ax.plot(bin_slope[:-2],nmad[np.arange(j,len(slope)-2*nb_corr,nb_corr)]**2,label='Empirical variance: $q$='+str(int(bin_corr[j]-5))+'-'+str(int(bin_corr[j]+5))+' %',color=color_list[u],linestyle=ls_list[i],marker='o',lw=0.5)
    u+=1



model_var = np.sqrt(3**2 + ((20+(((100-np.array(100))/100)*20)) * np.tan(np.array(x_slope) * np.pi / 180))**2 + (((100-np.array(95))/100)*15)**2.5)
ax.plot(x_slope,model_var**2,label='Modelled: center of above\ncategories',linestyle='dashed',color='black',lw=0.5)
model_var = np.sqrt(3**2 + ((20+(((100-np.array(80))/100)*20)) * np.tan(np.array(x_slope) * np.pi / 180))**2 + (((100-np.array(75))/100)*15)**2.5)
ax.plot(x_slope,model_var**2,linestyle='dashed',color='black',lw=0.5)
model_var = np.sqrt(3**2 + ((20+(((100-np.array(60))/100)*20)) * np.tan(np.array(x_slope) * np.pi / 180))**2 + (((100-np.array(55))/100)*15)**2.5)
ax.plot(x_slope,model_var**2,linestyle='dashed',color='black',lw=0.5)
model_var = np.sqrt(3**2 + ((20+(((100-np.array(40))/100)*20)) * np.tan(np.array(x_slope) * np.pi / 180))**2 + (((100-np.array(35))/100)*15)**2.5)
ax.plot(x_slope,model_var**2,linestyle='dashed',color='black',lw=0.5)
model_var = np.sqrt(3**2 + ((20+(((100-np.array(20))/100)*20)) * np.tan(np.array(x_slope) * np.pi / 180))**2 + (((100-np.array(15))/100)*15)**2.5)
ax.plot(x_slope,model_var**2,linestyle='dashed',color='black',lw=0.5)


ax.set_xlabel('Slope $\\alpha$ (degrees)')
ax.set_ylabel('Variance of elevation differences (m$^{2}$)')
ax.set_ylim([-100,2450])
ax.legend(loc='upper center',bbox_to_anchor=(0, 0., 0.75, 1),title='Elevation measurement error\nwith slope $\\alpha$ and quality\n of stereo-correlation $q$',title_fontsize=6)
ax.grid(linewidth=0.25)
ax.tick_params(width=0.35,length=2.5)

ax = fig.add_subplot(grid[9:, :5])
ax.text(0.025, 0.966, 'b', transform=ax.transAxes,
        fontsize=8, fontweight='bold', va='top', ha='left')

csv_dir = '/home/atom/ongoing/work_worldwide/tvar/06_rgi60'
list_in_csv = glob(os.path.join(csv_dir,'**/*dh_tvar.csv'),recursive=True)
# in_csv = '/home/atom/ongoing/work_worldwide/tvar/06_rgi60/N64W018_dh_tvar.csv'
for in_csv in list_in_csv:

    df = pd.read_csv(in_csv)
    df = df[df['bin_val']==-15]

    lags=df['lags']
    vmean=df['vmean']
    vstd=df['vstd']
    ax.plot(lags, vmean, lw=0.75, label='Empirical variogram',color='tab:blue')


sublag = np.linspace(0,20,200)
white = np.ones(np.shape(sublag)) * 65
local = 50 * (1 - np.exp(-1/2*4*sublag**2))
period = 40 * (1 - np.exp(-2 * np.sin(np.pi/1*sublag)**2))
linear = 1.2 * sublag **2

rq = 150*(1 - (1+ sublag**2/(2*100000*3))**(-100000))

# ft.plot_vgm(df['lags'],df['vmean'],df['vstd'])
ax.plot(sublag,white+local+period+linear, color='black', lw=0.75, linestyle='dashed',label = 'Modelled variogram: \nsum of individual models')

ax.plot(sublag,white, color='tab:cyan', label = 'Model: white noise',linestyle='dashed',lw=0.5)
ax.plot(sublag,local, color='tab:red', label = 'Model: local',linestyle='dashed',lw=0.5)
ax.plot(sublag,period, color='tab:gray', label = 'Model: periodic',linestyle='dashed',lw=0.5)
ax.plot(sublag,linear, color='tab:orange', label = 'Model: linear',linestyle='dashed',lw=0.5)
# ax.plot(sublag,rq,color='magenta',label='local-linear kernel')

# ax.fill_between(lags, vmean + vstd, vmean - vstd, facecolor='blue', alpha=0.5)
# ax.set_title('Variogram: ')
ax.set_xlabel('Temporal lag (years)')
ax.set_ylabel('Variance of elevation differences (m$^{2}$)')
ax.legend(loc='upper center',bbox_to_anchor=(0, 0., 0.75, 1),title='Temporal covariance\nof glacier elevation',title_fontsize=6)
ax.set_xlim([0,12])
ax.grid(linewidth=0.25)
ax.set_ylim([0,400])
ax.tick_params(width=0.35,length=2.5)
# ax.set_xlim([0,10])

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, ExpSineSquared as ESS, PairwiseKernel, RationalQuadratic as RQ, WhiteKernel as WK
import pyddem.fit_tools as ft
import xarray as xr
import matplotlib.pyplot as plt

# fn_stack = '/home/atom/ongoing/N64W017.nc'
# fn_stack = '/home/atom/ongoing/N77E016.nc'
# Upsala
fn_stack='/home/atom/ongoing/work_worldwide/figures/data_for_figs/S50W074.nc'
ds=xr.open_dataset(fn_stack)
ds.load()

ref_dem_date=np.datetime64('2013-01-01')
gla_mask = '/home/atom/data/inventory_products/RGI/rgi60_all.shp'
# gla_mask = None
nproc=10
clobber=True
tstep=0.1
opt_gpr=False
kernel=None
filt_ref='both'
filt_ls=False
conf_filt_ls=0.99
#specify the exact temporal extent needed to be able to merge neighbouring stacks properly
tlim=[np.datetime64('2000-01-01'),np.datetime64('2019-01-01')]

#pixel

# x=418930
# y=7107460

# x= 439900
# y=7099000

# x=530000
# y=8593000

# x=515300
# y=8601200

# x=510740
# y=8584920

# x=544674
# y=8580970

#Upsala
x=623000
y=4471000

#filtering temporal values
keep_vals = ds.uncert.values < 20
ds = ds.isel(time=keep_vals)

t_vals = ds.time.values
# dates_rm_dupli = sorted(list(set(list(t_vals))))
# ind_firstdate = []
# for i, date in enumerate(dates_rm_dupli):
#     ind_firstdate.append(list(t_vals).index(date))
# ds_filt = ds.isel(time=np.array(ind_firstdate))
# for i in range(len(dates_rm_dupli)):
#     t_ind = (t_vals == dates_rm_dupli[i])
#     if np.count_nonzero(t_ind) > 1:
#         print('Here: '+str(i))
#         print(ds.time.values[t_ind])
#         # careful, np.nansum gives back zero for an axis full of NaNs
#         mask_nan = np.all(np.isnan(ds.z.values[t_ind, :]), axis=0)
#         ds_filt.z.values[i, :] = np.nansum(ds.z.values[t_ind, :] * 1. / ds.uncert.values[t_ind, None, None] ** 2,
#                                            axis=0) / np.nansum(1. / ds.uncert.values[t_ind, None, None] ** 2, axis=0)
#         # ds_filt.z.values[i, : ] = np.nanmean(ds.z.values[t_ind,:],axis=0)
#         ds_filt.z.values[i, mask_nan] = np.nan
#         # ds_filt.uncert.values[i] = np.nanmean(ds.uncert.values[t_ind])
#         ds_filt.uncert.values[i] = np.nansum(ds.uncert.values[t_ind] * 1. / ds.uncert.values[t_ind] ** 2) / np.nansum(
#             1. / ds.uncert.values[t_ind] ** 2)

# ds = ds_filt

#starting
t_vals = ds['time'].values
ds_pixel = ds.sel(x=x,y=y,method='pad')
elev = ds_pixel.z.values
filt = np.logical_and.reduce((np.isfinite(elev),elev>-420,elev<8900))
t_vals=t_vals[filt]
elev=elev[filt]
med_slope = 10.
corr = ds_pixel.corr.values[filt]
uncert = np.sqrt(ds.uncert.values[filt]**2  + (20 * np.tan(med_slope * np.pi / 180)) ** 2 + (((100-corr)/100)*20)**2.5)

# fig, ax = plt.subplots(figsize=(16,9))
# ax.errorbar(t_vals, elev, uncert,fmt='o',color='black')
# ax.set_title('Raw data')
# ax.set_xlabel('Year after 2000')
# ax.set_ylabel('Elevation (m)')
# ax.legend(loc='lower left')
# ax.grid()
# plt.savefig('elev_raw.png',dpi=360)
# plt.close()

ref_arr = ds.ref_z.values
ref_vals = ds_pixel.ref_z.values

#spatial filtering
cutoff_kern_size=500
cutoff_thr=400.
res = 30
rad = int(np.floor(cutoff_kern_size / res))
max_arr, min_arr = ft.maxmin_disk_filter((ref_arr, rad))
ind_x = np.where(ds.x.values==ds_pixel.x.values)[0]
ind_y = np.where(ds.y.values==ds_pixel.y.values)[0]
max_elev = max_arr[ind_y,ind_x][0]
min_elev = min_arr[ind_y,ind_x][0]

ind = np.logical_or(elev > (max_elev + cutoff_thr), elev < (min_elev - cutoff_thr))

elev2 = np.copy(elev)
elev2[ind] = np.nan

#temporal filtering
base_thresh=150.
thresh=[-50,50]
delta_t = (ref_dem_date - t_vals).astype('timedelta64[D]').astype(float) / 365.24
dh = ref_vals - elev2
dt_arr = np.ones(dh.shape)
for i, d in enumerate(delta_t):
    dt_arr[i] = dt_arr[i] * d
d_data = dh / dt_arr
ind2 = np.logical_or(np.logical_and(dt_arr < 0, np.logical_or(dh < - base_thresh + dt_arr*thresh[1], dh > base_thresh + dt_arr*thresh[0])),
                             np.logical_and(dt_arr > 0, np.logical_or(dh > base_thresh + dt_arr*thresh[1], dh < - base_thresh + dt_arr*thresh[0])))

vect1 = np.arange(np.datetime64('2000-01-01'),ref_dem_date,np.timedelta64(10))
dt1 = (ref_dem_date - vect1).astype('timedelta64[D]').astype(float)/365.24
vect2 = np.arange(ref_dem_date,np.datetime64('2020-01-01'),np.timedelta64(10))
dt2 = (ref_dem_date - vect2).astype('timedelta64[D]').astype(float)/365.24
vect = np.arange(np.datetime64('2000-01-01'),np.datetime64('2020-01-01'),np.timedelta64(10))

ax = fig.add_subplot(grid[:6,5:])

ax.text(0.025, 0.95, 'c', transform=ax.transAxes,
        fontsize=8, fontweight='bold', va='top', ha='left')

ax.scatter(t_vals[~ind], elev[~ind],color='black')
ax.scatter(t_vals[ind2], elev2[ind2],color='tab:red')
ax.scatter(t_vals[ind], elev[ind],color='tab:red',label='Elevation outliers')
ax.scatter(t_vals[~ind2], elev2[~ind2],color='black',label='Elevations considered valid')
ax.plot(vect,np.ones(np.shape(vect))*(max_elev+cutoff_thr),lw=0.5,color='tab:orange')
ax.plot(vect,np.ones(np.shape(vect))*(min_elev-cutoff_thr),lw=0.5,color='tab:orange',label='Spatial filtering')
# ax.text(np.datetime64('2004-01-01'),max_elev+cutoff_thr+200,'Max. reference in disk + threshold',color='green',fontweight='bold')
# ax.text(np.datetime64('2004-01-01'),min_elev-cutoff_thr-200,'Min. reference in disk + threshold',color='green',fontweight='bold')
# ax.text(np.datetime64('2009-01-01'),min_elev-cutoff_thr+100,'Max. positive trend + threshold',color='blue',fontweight='bold')
# ax.text(np.datetime64('2009-01-01'),max_elev+cutoff_thr-200,'Max. negative trend + threshold',color='orange',fontweight='bold')
ax.plot(vect1, ref_vals - base_thresh - dt1*thresh[1],lw=0.5,color='tab:blue',label='Temporal filtering')
ax.plot(vect2, ref_vals + base_thresh - dt2*thresh[1],lw=0.5,color='tab:blue')
ax.plot(vect1, ref_vals + base_thresh - dt1*thresh[0],lw=0.5,color='tab:blue')
ax.plot(vect2, ref_vals - base_thresh - dt2*thresh[0],lw=0.5,color='tab:blue')

ax.fill_between(vect1,np.max(np.stack((ref_vals - base_thresh - dt1*thresh[1],np.ones(np.shape(vect1))*(min_elev-cutoff_thr))),axis=0),
                np.min(np.stack((ref_vals + base_thresh - dt1*thresh[0],np.ones(np.shape(vect1))*(max_elev+cutoff_thr))),axis=0),color='grey',alpha=0.5)
ax.fill_between(vect2,np.min(np.stack((ref_vals + base_thresh - dt2*thresh[1],np.ones(np.shape(vect2))*(max_elev+cutoff_thr))),axis=0),
                np.max(np.stack((ref_vals - base_thresh - dt2*thresh[0],np.ones(np.shape(vect2))*(min_elev-cutoff_thr))),axis=0),color='grey',alpha=0.5,label='Interval of valid elevations')
ax.scatter(ref_dem_date,ref_vals,5,color='tab:pink',label='Reference elevation (TanDEM-X)')
# ax.add_patch(patches.Rectangle((np.datetime64('2000-01-01'),120),np.timedelta64(int(365.24*20)),460,color='lightgrey',linestyle='dashed',fill=False,lw=1))
# ax.text(np.datetime64('2002-01-01'),200,'panel B',fontcolor='grey')
ax.set_xlabel('Year')
ax.set_ylabel('Elevation (m)')
ax.set_ylim([-500,3500])
ax.legend(loc='upper right',title='Filtering with reference TanDEM-X',title_fontsize=6,ncol=2,columnspacing=0.5)
ax.grid(linewidth=0.25)
ax.set_axisbelow(True)
ax.tick_params(width=0.35,length=2.5)

# plt.savefig('/home/atom/ongoing/work_worldwide/figures/Figure_S3.png',dpi=360)
# plt.close()

elev[ind] = np.nan

#gpr filtering
data = elev

tstep=0.1
y0 = 2000
y1 = 2020
t_pred = np.arange(y0, y1+tstep, tstep) - y0

ftime = t_vals[np.isfinite(data)]
total_delta = np.datetime64('{}-01-01'.format(int(y1))) - np.datetime64('{}-01-01'.format(int(y0)))
ftime_delta = np.array([t - np.datetime64('{}-01-01'.format(int(y0))) for t in ftime])
time_vals = (ftime_delta / total_delta) * (int(y1) - y0)

data_vals = data[np.isfinite(data)]
err_vals = uncert[np.isfinite(data)]**2

# beta1, beta0, incert_slope, Yl, Yu = ft.wls_matrix(time_vals,data_vals,1./err_vals,conf_slope=0.99)
#
# reg = ft.detrend(time_vals[~np.isnan(data_vals)],data_vals[~np.isnan(data_vals)],err_vals[~np.isnan(data_vals)])

# tmp_time = np.arange(0,20,0.1)
# fig, ax = plt.subplots(figsize=(16,9))
# ax.errorbar(time_vals, data_vals, np.sqrt(err_vals),fmt='o',color='black')
# ax.plot(tmp_time,tmp_time*beta1+beta0,color='blue')
# ax.fill_between(time_vals[~np.isnan(Yl)],Yu[~np.isnan(Yl)],Yl[~np.isnan(Yl)],color='blue',alpha=0.5)
# ax.set_title('Weighted least squares')
# ax.set_xlabel('Year after 2000')
# ax.set_ylabel('Elevation (m)')
# ax.legend(loc='lower left')
# ax.grid()

optimizer = None
n_restarts_optimizer = 0
n_out = 1
niter = 0
tag_detr = 1
final_fit = 0

num_finite = data_vals.size
good_vals = np.isfinite(data_vals)
max_z_score = [20, 12, 9, 6, 4]
perc_nonlin = [0,0,25,50,100]
# coef_var = [5,4,3,2,1]
# new_time_vals = np.unique(time_vals)
# new_data_vals = np.zeros(np.shape(new_time_vals))
# new_err_vals = np.zeros(np.shape(new_time_vals))
# for i, time_val in enumerate(new_time_vals):
#     ind = time_vals == time_val
#     new_data_vals[i] = np.nanmean(data_vals[ind])
#     new_err_vals[i] = np.nanmean(err_vals[ind])
# data_vals = new_data_vals
# time_vals = new_time_vals
# err_vals = new_err_vals
# good_vals = np.isfinite(data_vals)

# here we need to change the 0 for the x axis, in case we are using a linear kernel

ax = fig.add_subplot(grid[6:12,5:])

ax.text(0.025, 0.95, 'd', transform=ax.transAxes,
        fontsize=8, fontweight='bold', va='top', ha='left')

while (n_out > 0 or final_fit==0) and num_finite >= 2:

    beta1, beta0, incert_slope, _, _ = ft.wls_matrix(time_vals[good_vals], data_vals[good_vals],
                                                     1. / err_vals[good_vals], conf_slope=0.99)

    # standardized dispersion from linearity
    res_stdized = np.sqrt(np.mean(
        (data_vals[good_vals] - (beta0 + beta1 * time_vals[good_vals])) ** 2 / err_vals[good_vals]))
    res = np.sqrt(np.mean((data_vals[good_vals] - (beta0 + beta1 * time_vals[good_vals])) ** 2))
    if perc_nonlin[min(niter, len(perc_nonlin) - 1)] == 0:
        opt_var = 0
    else:
        opt_var = (res / res_stdized ** 2) ** 2 * 100. / (5 * perc_nonlin[min(niter, len(perc_nonlin) - 1)])

    k1 = PairwiseKernel(1, metric='linear') + PairwiseKernel(1, metric='linear') * C(opt_var) * RQ(10, 3)  # linear kernel
    k2 = C(30) * ESS(length_scale=1, periodicity=1)  # periodic kernel
    k3 = C(50) * RBF(0.75)
    kernel = k1 + k2 + k3

    mu_x = np.nanmean(time_vals[good_vals])
    detr_t_pred = t_pred - mu_x
    detr_time_vals = time_vals - mu_x
    mu_y = np.nanmean(data_vals)
    detr_data_vals = data_vals - mu_y

    # if we remove a linear trend, normalize_y should be false...
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer,
                                  alpha=err_vals[good_vals], normalize_y=False)
    gp.fit(detr_time_vals[good_vals].reshape(-1, 1), detr_data_vals[good_vals].reshape(-1, 1))
    y_pred, sigma = gp.predict(detr_t_pred.reshape(-1, 1), return_std=True)
    y_, s_ = ft.interp_data(detr_t_pred, y_pred.squeeze(), sigma.squeeze(), detr_time_vals)
    z_score = np.abs(detr_data_vals - y_) / s_

    isin = np.logical_or(z_score < 4, ~np.isfinite(z_score))
    # we continue the loop if there is a least one value outside 4 stds
    n_out = np.count_nonzero(~isin)

    # good elevation values can also be outside 4stds because of bias in the first fits
    # thus, if needed, we remove outliers packet by packet, starting with the largest ones
    isout = np.logical_and(z_score > max_z_score[min(niter, len(max_z_score) - 1)], np.isfinite(z_score))

    if niter == 4:
        ax.plot(y0+detr_t_pred + mu_x, y_pred + mu_y, label='Successive GP\nregression fits', color=plt.cm.Greys(0.8),linewidth=0.25)
        ax.errorbar(y0+detr_time_vals[isout] +mu_x, data_vals[isout], np.sqrt(err_vals[isout]), fmt='o', color='tab:red',label='Elevation outliers\n(1$\sigma$ measurement error)')
        ax.fill_between(y0+detr_t_pred +mu_x, mu_y + y_pred.squeeze() + max_z_score[min(niter, len(max_z_score) - 1)] * sigma.squeeze(),
                        mu_y + y_pred.squeeze() - max_z_score[min(niter, len(max_z_score) - 1)] * sigma.squeeze(), facecolor=plt.cm.Blues(0.2+niter/10), alpha=0.8,label='Successive GP CIs\nof 20, 12, 9, 6 and 4-$\sigma$')
    else:
        ax.plot(y0+detr_t_pred + mu_x, y_pred + mu_y, color=plt.cm.Greys(0.8),linewidth=0.75)
        ax.errorbar(y0+detr_time_vals[isout] + mu_x, data_vals[isout], np.sqrt(err_vals[isout]), fmt='o', color='tab:red')
        ax.fill_between(y0+detr_t_pred + mu_x,
                        mu_y + y_pred.squeeze() + max_z_score[min(niter, len(max_z_score) - 1)] * sigma.squeeze(),
                        mu_y + y_pred.squeeze() - max_z_score[min(niter, len(max_z_score) - 1)] * sigma.squeeze(),
                        facecolor=plt.cm.Blues(0.2+niter/10), alpha=0.8)


    data_vals[isout] = np.nan

    good_vals = np.isfinite(data_vals)
    num_finite = np.count_nonzero(good_vals)

    niter += 1

    # if we have no outliers outside 4 std, initialize back values to jump directly to the final fitting step
    if n_out == 0 and final_fit == 0 and niter < len(max_z_score) - 1:
        n_out = 1
        final_fit = 1
        niter = len(max_z_score) - 1
    elif niter >= len(max_z_score) - 1:
        final_fit = 1

ax.errorbar(y0+detr_time_vals[~isout] + mu_x, data_vals[~isout], np.sqrt(err_vals[~isout]), fmt='o', color='black',zorder=30,label='Elevations considered valid\n(1$\sigma$ measurement error)')
# ax.set_title('GPR: ')
ax.set_xlabel('Year')
ax.set_ylabel('Elevation (m)')
ax.legend(loc='upper right',title='Filtering with iterative GP regression',title_fontsize=6,ncol=2,columnspacing=0.5)
ax.grid(linewidth=0.25)
ax.set_ylim([140,750])
ax.set_xticks(np.arange(2000,2024,4))
ax.tick_params(width=0.35,length=2.5)

ax = fig.add_subplot(grid[12:,5:])

ax.text(0.025, 0.95, 'e', transform=ax.transAxes,
        fontsize=8, fontweight='bold', va='top', ha='left')
ax.plot(y0+detr_t_pred + mu_x, y_pred + mu_y, lw=0.75, label='GP regression time series', color='blue')
ax.errorbar(y0+detr_time_vals[~isout] + mu_x, data_vals[~isout], np.sqrt(err_vals[~isout]), fmt='o', color='black', label='Elevations considered valid\n(1$\sigma$ measurement error)')
ax.fill_between(y0+detr_t_pred +mu_x, mu_y + y_pred.squeeze() + sigma.squeeze(),
                        mu_y + y_pred.squeeze() - sigma.squeeze(), facecolor='blue', alpha=0.5, label='1$\sigma$ GP credible interval')
ax.set_xlabel('Year')
ax.set_ylabel('Elevation (m)')
ax.legend(loc='upper right',title='Final time series\nwith GP regression',title_fontsize=6,columnspacing=0.5)
ax.set_ylim([200,520])
ax.set_xticks(np.arange(2000, 2024, 4))
ax.grid(linewidth=0.25)
ax.tick_params(width=0.35,length=2.5)

plt.savefig('/home/atom/ongoing/work_worldwide/figures/final/ED_Figure_3.jpg',dpi=500,bbox_inches='tight')