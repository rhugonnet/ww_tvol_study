"""
@author: hugonnet
compile elevation variances derived for each tile
#TODO: this was done in parallel of previous fitting results, need to dissociate the two processes in "fit_tools" to allow to do this sequentially
"""
from __future__ import print_function
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

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


plt.figure(figsize=(16,9))

slope = df.bin_slope.values
corr = df.bin_corr.values
bin_slope = sorted(list(set(list(slope))))
bin_corr = sorted(list(set(list(corr))))
nb_slope = len(bin_slope)
nb_corr = len(bin_corr)

color_list = ['r','b','k','m','y','c','g','r','b','k']
ls_list = ['solid','dashed','dotted']

model_var = np.sqrt(3**2 + (20 * np.tan(np.array(5) * np.pi / 180))**2) + (((100-np.array(bin_corr))/100)*20)**1.25

for i in range(len(region_nmad)):
    i = 0
    for j in range(nb_slope-2):

        nmad = region_nmad[i]

        plt.plot(corr[1:nb_corr],nmad[j*nb_corr+1:j*nb_corr+nb_corr],label='Slope category: '+str(bin_slope[j]-5)+'-'+str(bin_slope[j]+5)+' degrees',color=color_list[j],linestyle=ls_list[i])


# plt.plot(bin_corr,model_var,label='model',linewidth=2)

plt.xlabel('Correlation (percent)')
plt.ylabel('Stable terrain NMAD (m)')
plt.ylim([0,50])
plt.legend()


plt.figure(figsize=(16,9))

model_var = np.sqrt(3**2 + (20 * np.tan(np.array(bin_slope) * np.pi / 180))**2 + (((100-np.array(20))/100)*20)**2.5)
i=0
for i in range(len(region_nmad)-1):
    for j in range(nb_corr):

        nmad = region_nmad[i]

        plt.plot(bin_slope,nmad[np.arange(j,len(slope),nb_corr)],label='region: '+region_list[i]+', corr: '+str(bin_corr[j]),color=color_list[j],linestyle=ls_list[i])

plt.plot(bin_slope,model_var,label='model',linewidth=2)

plt.xlabel('Slope (degrees)')
plt.ylabel('Stable terrain NMAD (m)')
plt.ylim([0,100])
plt.legend()



