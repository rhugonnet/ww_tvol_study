from __future__ import print_function
import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,8))

dir_reg = '/home/atom/ongoing/work_worldwide/vol/reg'
list_reg = np.arange(1,20)

colors = ['black','tab:blue','tab:orange','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']
names = ['ALA (01)','WNA (02)','ACN (03)','ACS (04)','GRL (05)','ISL (06)','SJM (07)','SCA (08)','RUA (09)','ASN (10)','CEU (11)','CAU (12)','ASC (13)','ASW (14)','ASE (15)','TRP (16)','SAN (17)','NZL (18)','ANT (19)']
ax = fig.add_subplot(1, 2, 1)

ax.text(0.05,0.975,'a',transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')

for reg in list_reg:

    df_reg = pd.read_csv(os.path.join(dir_reg,'dh_'+str(reg).zfill(2)+'_rgi60_int_base_reg_subperiods.csv'))

    vec = np.arange(2000,2021,1)

    cum_dm = np.array([0]+list(np.cumsum(df_reg.dmdt.values[0:20])))

    ind = list(list_reg).index(reg) % 10
    if list(list_reg).index(reg) >= 10:
        ls='dashed'
    else:
        ls='-'
    ax.plot(vec,cum_dm,color=colors[ind],linestyle=ls,lw=2,label=names[list(list_reg).index(reg)])

ax.set_xlabel('Year')
ax.set_ylabel('Cumulative mass change (Gt)')
ax.set_ylim((-1500,150))
ax.set_xticks(np.arange(2000,2021,4))
ax.legend(loc='lower left',framealpha=1,ncol=2)
ax.grid()

ax = fig.add_subplot(1, 2, 2)

ax.text(0.05,0.975,'b',transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')

for reg in list_reg:

    df_reg = pd.read_csv(os.path.join(dir_reg,'dh_'+str(reg).zfill(2)+'_rgi60_int_base_reg_subperiods.csv'))

    vec = np.arange(2000,2021,1)

    cum_dm = np.array([0]+list(np.cumsum(df_reg.dmdt.values[0:20]/df_reg.tarea.values[0:20]*10**9)))
    ind = list(list_reg).index(reg) % 10
    if list(list_reg).index(reg) >= 10:
        ls='dashed'
    else:
        ls='-'
    ax.plot(vec,cum_dm,color=colors[ind],linestyle=ls,lw=2)

ax.set_xlabel('Year')
ax.set_ylabel('Cumulative specific mass change (m w.e.)')
ax.set_ylim((-20,2))
ax.set_xticks(np.arange(2000,2021,4))
ax.grid()

plt.savefig('/home/atom/ongoing/work_worldwide/figures/Figure_S15.png',dpi=400)