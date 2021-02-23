"""
@author: hugonnet
display filtering + fitting operations performed for 1 pixel of a stack (for reasons of space, cannot be done with the entire stack)
#TODO: add filtering routines with the surrounding pixels; right now is only demonstrated the rough first filtering + GP
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, ExpSineSquared as ESS, PairwiseKernel, RationalQuadratic as RQ, WhiteKernel as WK
import pyddem.fit_tools as ft
import xarray as xr
import matplotlib.pyplot as plt

fn_stack = '/home/atom/ongoing/work_worldwide/N60W141.nc'
# Upsala
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
tlim=[np.datetime64('2000-01-01'),np.datetime64('2020-01-01')]

#pixel

#Upsala
# x=505300
# y=3915430

x=501000
y=6655000

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

fig, ax = plt.subplots(figsize=(16,9))
ax.errorbar(t_vals, elev, uncert,fmt='o',color='black')
ax.set_title('Raw data')
ax.set_xlabel('Year after 2000')
ax.set_ylabel('Elevation (m)')
ax.legend(loc='lower left')
ax.grid()
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

fig, ax = plt.subplots(figsize=(16,9))
ax.errorbar(t_vals[~ind], elev[~ind], uncert[~ind],fmt='o',color='black')
ax.errorbar(t_vals[ind], elev[ind], uncert[ind],fmt='o',color='red')
ax.plot(t_vals,np.ones(np.shape(t_vals))*(max_elev+cutoff_thr),lw=2,color='green')
ax.plot(t_vals,np.ones(np.shape(t_vals))*(min_elev-cutoff_thr),lw=2,color='green')
ax.set_title('Reference spatial filtering: ')
ax.set_xlabel('Year after 2000')
ax.set_ylabel('Elevation (m)')
ax.legend(loc='lower left')
ax.grid()
# plt.savefig('elev_spfilt.png',dpi=360)
# plt.close()

elev[ind] = np.nan

#temporal filtering
base_thresh=100.
thresh=[-15,15]
delta_t = (ref_dem_date - t_vals).astype('timedelta64[D]').astype(float) / 365.24
dh = ref_vals - elev
dt_arr = np.ones(dh.shape)
for i, d in enumerate(delta_t):
    dt_arr[i] = dt_arr[i] * d
d_data = dh / dt_arr

ind = np.logical_or(np.logical_and(dt_arr < 0, np.logical_or(dh < - base_thresh + dt_arr*thresh[1], dh > base_thresh + dt_arr*thresh[0])),
                             np.logical_and(dt_arr > 0, np.logical_or(dh > base_thresh + dt_arr*thresh[1], dh < - base_thresh + dt_arr*thresh[0])))

fig, ax = plt.subplots(figsize=(16,9))
ax.errorbar(t_vals[~ind], elev[~ind], uncert[~ind],fmt='o',color='black')
ax.errorbar(t_vals[ind], elev[ind], uncert[ind],fmt='o',color='red')
ax.scatter(ref_dem_date,ref_vals,color='orange')
ax.plot(t_vals[dt_arr<=0], ref_vals + base_thresh - dt_arr[dt_arr<=0]*thresh[1],color='blue')
ax.plot(t_vals[dt_arr>=0], ref_vals - base_thresh - dt_arr[dt_arr>=0]*thresh[1], color='blue')
ax.plot(t_vals[dt_arr<=0], ref_vals - base_thresh - dt_arr[dt_arr<=0]*thresh[0],color='orange')
ax.plot(t_vals[dt_arr>=0], ref_vals + base_thresh - dt_arr[dt_arr>=0]*thresh[0], color='orange')
ax.set_title('Reference temporal filtering: ')
ax.set_xlabel('Year after 2000')
ax.set_ylabel('Elevation (m)')
ax.legend(loc='lower left')
ax.grid()
# plt.savefig('elev_tfilt.png',dpi=360)
# plt.close()

elev[ind] = np.nan

#gpr filtering
data = elev

tstep=0.1
y0 = t_vals[0].astype('datetime64[D]').astype(object).year
y0 = 2000
y1 = t_vals[-1].astype('datetime64[D]').astype(object).year + 1.1
t_pred = np.arange(y0, y1, tstep) - y0

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
perc_nonlin = [0,0,0,0,0]
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


while (n_out > 0 or final_fit == 0) and num_finite >= 2:

    # we want a local linear kernel only with variance significantly different from 0 at 99% confidence
    beta1, beta0, incert_slope, _, _ = ft.wls_matrix(time_vals[good_vals], data_vals[good_vals],
                                                  1. / err_vals[good_vals], conf_slope=0.99)

    # standardized dispersion from linearity
    res_stdized = np.sqrt(np.mean(
        (data_vals[good_vals] - (beta0 + beta1 * time_vals[good_vals])) ** 2 / err_vals[good_vals]))
    res = np.sqrt(np.mean((data_vals[good_vals] - (beta0 + beta1 * time_vals[good_vals])) ** 2))
    if final_fit==0:
        nonlin_var = 0
        period_nonlinear = 20
        ind_first = np.logical_and(good_vals, time_vals < 10.)
        ind_last = np.logical_and(good_vals, time_vals >= 10.)
        #first time, filter out very large outliers
        if niter ==0:
            base_var = 150.
        elif np.count_nonzero(ind_first) >= 5 and np.count_nonzero(ind_last) >= 5:
            diff = np.abs(np.mean(data_vals[ind_first]) - np.mean(data_vals[ind_last]))
            diff_std = np.sqrt(np.var(data_vals[ind_first]) + np.var(data_vals[ind_last]))
            if diff - diff_std >0:
                base_var = 50 + (diff-diff_std)**2/2
            else:
                base_var = 50.
        else:
            base_var = 50.
    else:
        nonlin_var = (res / res_stdized) ** 2
        # nonlin_var = 1
        period_nonlinear = 100. / res_stdized ** 2
        # period_nonlinear = 10.
        base_var=50

    print(base_var)

    k1 = PairwiseKernel(1, metric='linear') + PairwiseKernel(1, metric='linear') * C(nonlin_var) * RQ(10, period_nonlinear)  # linear kernel
    k2 = C(30) * ESS(length_scale=1, periodicity=1)  # periodic kernel
    k3 = C(base_var) * RBF(1.5)
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

    fig, ax = plt.subplots(figsize=(16,9))
    ax.plot(detr_t_pred +mu_x, y_pred + mu_y, lw=2, label='mean', color='blue')
    ax.errorbar(detr_time_vals[~isout] +mu_x, data_vals[~isout], np.sqrt(err_vals[~isout]), fmt='o', color='black')
    ax.errorbar(detr_time_vals[isout] +mu_x, data_vals[isout], np.sqrt(err_vals[isout]), fmt='o', color='red')
    ax.fill_between(detr_t_pred +mu_x, mu_y + y_pred.squeeze() + max_z_score[min(niter, len(max_z_score) - 1)] * sigma.squeeze(),
                    mu_y + y_pred.squeeze() - max_z_score[min(niter, len(max_z_score) - 1)] * sigma.squeeze(), facecolor='blue', alpha=0.5)
    ax.set_title('GPR: ')
    ax.set_xlabel('Year after 2000')
    ax.set_ylabel('Elevation (m)')
    ax.legend(loc='lower left')
    ax.grid()
    # plt.savefig('elev_gpr_zsc+'+str(max_z_score[min(niter, len(max_z_score) - 1)])+'.png', dpi=360)
    # plt.close()

    data_vals[isout] = np.nan

    good_vals = np.isfinite(data_vals)
    num_finite = np.count_nonzero(good_vals)

    niter += 1

    # if we have no outliers outside 4 std, initialize back values to jump directly to the final fitting step
    if final_fit == 1:
        n_out = 0
    if n_out == 0 and final_fit == 0:
        n_out = 1
        final_fit = 1
        niter = len(max_z_score) - 1

fig, ax = plt.subplots(figsize=(16,9))
ax.plot(detr_t_pred +mu_x, y_pred + mu_y, lw=2, label='mean', color='blue')
ax.errorbar(detr_time_vals[~isout] +mu_x, data_vals[~isout], np.sqrt(err_vals[~isout]), fmt='o', color='black')
ax.errorbar(detr_time_vals[isout] +mu_x, data_vals[isout], np.sqrt(err_vals[isout]), fmt='o', color='red')
ax.fill_between(detr_t_pred +mu_x, mu_y + y_pred.squeeze() +  sigma.squeeze(),
                mu_y + y_pred.squeeze() - sigma.squeeze(), facecolor='blue', alpha=0.5)
ax.set_title('GPR: ')
ax.set_xlabel('Year after 2000')
ax.set_ylabel('Elevation (m)')
ax.legend(loc='lower left')
ax.set_xticks(np.arange(0.75,19.75,1))
ax.grid()

plt.savefig('elev_gpr_final.png', dpi=360)
plt.close()


