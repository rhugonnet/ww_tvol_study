
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize

all_csv = '/home/atom/ongoing/work_worldwide/validation/tcorr/tinterp_corr_deseas_agg_all.csv'
# all_csv = '/home/atom/ongoing/work_worldwide/validation/tinterp_corr_agg_all.csv'

df = pd.read_csv(all_csv)

# df = df[df.reg==5]

cutoffs = list(set(list(df.cutoff)))
dts = sorted(list(set(list(df.nb_dt))))

col = ['tab:orange','tab:blue','tab:green','tab:red','tab:cyan','tab:brown','tab:gray','tab:pink','tab:purple']

#plot covar by lag
# for dt in dts:
#
#     df_dt = df[df.nb_dt == dt]
#
#     for cutoff in cutoffs:
#         df_c = df_dt[df_dt.cutoff == cutoff]
#
#         if cutoff == 10000:
#             plt.scatter(df_c.bins.values[1],df_c.exp.values[1],color=col[dts.index(dt)],label=str(dt))
#             plt.scatter(df_c.bins.values[20],df_c.exp.values[20],color=col[dts.index(dt)])
#             plt.scatter(df_c.bins.values[50],df_c.exp.values[50],color=col[dts.index(dt)])
#         elif cutoff == 100000:
#             plt.scatter(df_c.bins.values[20],df_c.exp.values[20],color=col[dts.index(dt)])
#             plt.scatter(df_c.bins.values[50],df_c.exp.values[50],color=col[dts.index(dt)])
#         else:
#             plt.scatter(df_c.bins.values[20],df_c.exp.values[20],color=col[dts.index(dt)])
#             plt.scatter(df_c.bins.values[50],df_c.exp.values[50],color=col[dts.index(dt)])
#
# plt.ylim([0,50])
# plt.xscale('log')
# plt.legend()


#plot covar by dt
dts = sorted(dts)
dts.remove(540.)
dts.remove(900.)
dts.remove(1750.)
dts.remove(2250.)
arr_res = np.zeros((len(dts),7))


arr_count = np.zeros((len(dts),7))

for dt in dts:

    df_dt = df[df.nb_dt == dt]

    for cutoff in cutoffs:
        df_c = df_dt[df_dt.cutoff == cutoff]

        if cutoff == 10000:
            arr_res[dts.index(dt),0]=np.nanmean(df_c.exp.values[1:2])
            arr_count[dts.index(dt),0]=np.nanmean(df_c['count'].values[1:2])
            arr_res[dts.index(dt), 1] = np.nanmean(df_c.exp.values[20 - 10:20 + 10])
            arr_count[dts.index(dt), 1] = np.nanmean(df_c['count'].values[20 - 10:20 + 10])
            arr_res[dts.index(dt), 2] = np.nanmean(df_c.exp.values[50 - 10:50 + 10])
            arr_count[dts.index(dt), 2] = np.nanmean(df_c['count'].values[50 - 10:50 + 10])
        elif cutoff == 100000:
            arr_res[dts.index(dt),3]=np.nanmean(df_c.exp.values[20-5:20+20])
            arr_count[dts.index(dt),3]=np.nanmean(df_c['count'].values[20-10:20+10])
            arr_res[dts.index(dt),4]=np.nanmean(df_c.exp.values[50-10:50+10])
            arr_count[dts.index(dt),4]=np.nanmean(df_c['count'].values[50-10:50+10])
        elif cutoff == 1000000:
            arr_res[dts.index(dt),5]=np.nanmean(df_c.exp.values[20-10:20+30])
            arr_count[dts.index(dt),5]=np.nanmean(df_c['count'].values[20-10:20+30])
            arr_res[dts.index(dt),6]=np.nanmean(df_c.exp.values[50-40:50+40])
            arr_count[dts.index(dt),6]=np.nanmean(df_c['count'].values[50-40:50+40])

arr_res[arr_count<100]=np.nan
# for dt in dts:
#
#     df_dt = df[df.nb_dt == dt]
#
#     for cutoff in cutoffs:
#         df_c = df_dt[df_dt.cutoff == cutoff]
#
#         if cutoff == 10000:
#             plt.scatter(dt,df_c.exp.values[1],color=col[0])
#             plt.scatter(dt,np.nanmean(df_c.exp.values[20-10:20+10]),color=col[1])
#             plt.scatter(dt,np.nanmean(df_c.exp.values[50-10:50+10]),color=col[2])
#         elif cutoff == 100000:
#             plt.scatter(dt,np.nanmean(df_c.exp.values[20-10:20+10]),color=col[3])
#             plt.scatter(dt,np.nanmean(df_c.exp.values[50-10:50+10]),color=col[4])
#         else:
#             plt.scatter(dt,np.nanmean(df_c.exp.values[20-10:20+10]),color=col[5])
#             plt.scatter(dt,np.nanmean(df_c.exp.values[50-10:50+10]),color=col[6])


fig = plt.figure(figsize=(16,16))

# plt.subplots_adjust(hspace=0.3)
grid = plt.GridSpec(6, 13, wspace=0.05, hspace=0.5)

ax = fig.add_subplot(grid[:3,:2])

# ax = fig.add_subplot(2, 1, 1)

vario = df[df.nb_dt == 720.]

vec_bins = []
vec_exp = []

vgm1 = vario[vario.cutoff == 10000]
vgm1 = vgm1[vgm1.bins<3000]

for i in range(6):
    vec_bins += [np.nanmean(vgm1.bins.values[0+i*5:5+i*5])]
    vec_exp += [np.nanmean(vgm1.exp.values[0+i*5:5+i*5])]
# vec_bins += vgm1.bins.tolist()
# vec_exp += vgm1.exp.tolist()

vgm1 = vario[vario.cutoff == 100000]
vgm1 = vgm1[np.logical_and(vgm1.bins>3000,vgm1.bins<30000)]
vec_bins += vgm1.bins.tolist()
vec_exp += vgm1.exp.tolist()

vgm1 = vario[vario.cutoff == 1000000]
vgm1 = vgm1[vgm1.bins>30000]

for i in range(18):
    vec_bins += [np.nanmean(vgm1.bins.values[0+i*5:5+i*5])]
    vec_exp += [np.nanmean(vgm1.exp.values[0+i*5:5+i*5])]

vec_bins = np.array(vec_bins)
vec_exp=np.array(vec_exp)

def sph_var(c0,c1,a1,h):
    if h < a1:
        vgm = c0 + c1 * (3 / 2 * h / a1-1 / 2 * (h / a1) ** 3)
    else:
        vgm = c0 + c1
    return vgm

vect = np.array(list(np.arange(0,3000,1)) + list(np.arange(3000,30000,10)) + list(np.arange(30000,3000000,100)))

mod = []
c1s = [0] + list(arr_res[dts.index(720.),:])
a1s = [0.2,2,5,20,50,200]

#find unbiased sills
list_c = []
for j in range(len(a1s)):
    print('Range:' + str(a1s[-1 - j]))
    c = c1s[-2 - j] - c1s[-3 - j]
    print(c)
    for k in range(j):
        # c -= sph_var(0, list_c[k], a1s[-1 - k] * 1000, a1s[-1 - j] * 1000)
        if j>5:
            c -= (sph_var(0, list_c[k], a1s[-1 - k] * 1000, a1s[-1 - j] * 1000) - sph_var(0,list_c[k], a1s[-1-k]*1000,a1s[-2-j]*1000))
        elif j==5:
            c -= sph_var(0, list_c[k], a1s[-1 - k] * 1000, a1s[-1 - j] * 1000)
    c = max(0, c)
    list_c.append(c)

list_c.reverse()

#compute variogram
for i in range(len(vect)):
    val = 0
    for j in range(len(a1s)):
        val += sph_var(0,list_c[j],a1s[j]*1000,vect[i])
    mod.append(val)

mod = np.array(mod)

ax.scatter(vec_bins/1000,vec_exp,color='black',marker='x')
ax.set_xlim((0,3))
ax.set_ylim((0,55))
ax.set_xticks([0,1,2])

ax.text(0.075, 0.975, 'a', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')
ax.vlines(0.15,0,60,color=col[0])
ax.text(0.4,c1s[1]-5,'$s_0$',color=col[0],ha='left',va='bottom',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7))
ax.vlines(2,0,60,color=col[1])
ax.text(2.2,c1s[2]-5,'$s_1$',color=col[1],ha='left',va='bottom',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7))

ax.plot(vect/1000,mod,color='dimgrey',linestyle='dashed')
# ax.hlines(25,0,500,colors='black',linestyles='dotted')
ax.set_ylabel('Variance of elevation differences (m$^2$)')


ax = fig.add_subplot(grid[:3,4:6])

ax.scatter(vec_bins/1000,vec_exp,color='black',marker='x')
ax.set_xlim((0,550))
ax.set_ylim((0,55))
ax.set_xticks([0,100,200,300,400,500])

# ax.text(0.075, 0.975, 'C', transform=ax.transAxes,
#         fontsize=14, fontweight='bold', va='top', ha='left')
ax.vlines(50,0,60,colors=[col[4]])
ax.text(70,c1s[5]-5,'$s_4$',color=col[4],ha='left',va='bottom',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7))
ax.vlines(200,0,60,colors=[col[5]])
ax.text(220,c1s[6]-7,'$s_5$',color=col[5],ha='left',va='bottom',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7))
ax.vlines(500,0,60,colors=[col[6]])
ax.text(480,c1s[6]-7,'$s_6$',color=col[6],ha='right',va='bottom',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7))
ax.plot(vect/1000,mod,color='dimgrey',linestyle='dashed')
# ax.hlines(25,0,500,colors='black',linestyles='dotted')

ax.set_yticks([])


ax = fig.add_subplot(grid[:3,2:4])

ax.scatter(vec_bins/1000,vec_exp,color='black',marker='x')
ax.set_xlim((0,30))
ax.set_ylim((0,55))
ax.set_xticks([0,10,20])

# ax.text(0.075, 0.975, 'B', transform=ax.transAxes,
#         fontsize=14, fontweight='bold', va='top', ha='left')
ax.vlines(5,0,60,color=col[2])
ax.text(6,c1s[3]-5,'$s_2$',color=col[2],ha='left',va='bottom',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7))
ax.vlines(20,0,60,color=col[3])
ax.text(21,c1s[4]-5,'$s_3$',color=col[3],ha='left',va='bottom',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7))
ax.plot(vect/1000,mod,color='dimgrey',linestyle='dashed')
# ax.hlines(25,0,500,colors='black',linestyles='dotted',label='Global mean variance')

ax.set_yticks([])
ax.set_xlabel('Spatial lag (km)')
ax.plot([],[],color='grey',linestyle='dashed',label='Sum of spherical models')
ax.scatter([],[],color='black',marker='x',label='Empirical variance')
ax.vlines([],[],[],color=col[0],label='0.15 km')
ax.vlines([],[],[],color=col[1],label='2 km')
ax.vlines([],[],[],color=col[2],label='5 km')
ax.vlines([],[],[],color=col[3],label='20 km')
ax.vlines([],[],[],color=col[4],label='50 km')
ax.vlines([],[],[],color=col[5],label='200 km')
ax.vlines([],[],[],color=col[6],label='500 km')
ax.legend(loc='lower center',ncol=3,title='Spatial correlations of GP elevation at $\Delta t$ = 720 days',title_fontsize=12)


ax = fig.add_subplot(grid[3:,:6])

coefs_list = []
y = None
# arr_res[0:1,4]=25
# arr_res[arr_res>25] = 25.
# arr_res[4,2]=np.nan
# arr_res[3:,3]=np.nan
# arr_res[0,3]=25.
# arr_res[0,3:] = np.nan
for i in [0,1,2,3,4,5,6]:

# i=0
# arr_res[-1,0]=np.nan
    coefs , _ = scipy.optimize.curve_fit(lambda t,a,b:a*t+b,  np.array(dts)[~np.isnan(arr_res[:,i])],  np.sqrt(arr_res[:,i][~np.isnan(arr_res[:,i])]))
    coefs_list.append(coefs)

    x = np.arange(0, 3000, 1)
    if y is not None:
        y0 = y
    else:
        y0 = x*0

    y = coefs[0]*x+coefs[1] #- 2*np.sin(x/365.2224*np.pi)**2
    # y[y>25]=25.
    # y[y<y0]=y0[y<y0]

    y = y
    ax.plot(x,y**2 -2*np.sin(x/365.2224*2*np.pi)**2,color=col[i])

    ax.fill_between(x,y0**2 -2*np.sin(x/365.2224*2*np.pi)**2,y**2 -2*np.sin(x/365.2224*2*np.pi)**2,color = col[i],alpha=0.2)

# ax.fill_between(x,40*np.ones(len(x)),y,color='tab:gray')

# arr_res[0,3:]=25.
for i in [0,1,2,3,4,5,6]:
    ax.scatter(dts,arr_res[:,i],color=col[i])


# ax.hlines(25,0,3000,linestyles='dashed',color='tab:gray')

ax.scatter([],[],color='black',label='Empirical variance')
ax.plot([],[],color='black',label='Quadratic + Sin fit')
ax.fill_between([],[],color=col[0],label='0.15 km')
ax.fill_between([],[],color=col[1],label='2 km')
ax.fill_between([],[],color=col[2],label='5 km')
ax.fill_between([],[],color=col[3],label='20 km')
ax.fill_between([],[],color=col[4],label='50 km')
ax.fill_between([],[],color=col[5],label='200 km')
ax.fill_between([],[],color=col[6],label='500 km')
ax.set_xlim([0,1500])
ax.set_ylim([0,60])
ax.set_ylabel('Variance of elevation differences (m$^{2}$)')
ax.set_xlabel('Days to closest observation $\Delta t$')
ax.vlines(720,0,100,colors='black',linestyles='dashed')
ax.text(800,5,'$\overline{s_{0}(\Delta t)}$: correlated until 0.15 km',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7),color='tab:orange')
ax.text(950,24,'$s_{1}(\Delta t)$: correlated until 2 km',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7),color='tab:blue')
ax.text(1250,38,'$s_{3}(\Delta t)$',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7),color='tab:red')
ax.text(1370,48,'$s_{5}(\Delta t)$',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7),color='tab:brown')

# ax.text(1000,22,'Fully correlated = Systematic',bbox= dict(boxstyle='round', facecolor='white', alpha=0.5),color='dimgrey')
# plt.xscale('log')
ax.legend(loc=(0.025,0.475),ncol=1,title='Spatial correlations of\nGP elevation with\ntime lag to observation',title_fontsize=12)
ax.text(0.025, 0.975, 'b', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')
ax.text(740,45,'panel (a)',fontweight='bold',va='bottom',ha='left')
# plt.savefig('/home/atom/ongoing/work_worldwide/figures/Figure_S12.png',dpi=360)

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pybob.ddem_tools import nmad

df_gp = pd.read_csv('/home/atom/data/other/Hugonnet_2020/dhdt_int_GP.csv')
df_hr = pd.read_csv('/home/atom/data/other/Hugonnet_2020/dhdt_int_HR.csv')

# ind = np.logical_and(df_hr.perc_meas>0.70,df_hr.category.values=='matthias')
# ind = np.logical_and(df_hr.perc_meas>0.70,df_hr.area.values<1000000.)
ind = df_hr.perc_meas>0.70

list_rgiid = list(df_hr[ind].rgiid)

list_area = list(df_hr[df_hr.rgiid.isin(list_rgiid)].area)

list_rgiid = [rgiid for _, rgiid in sorted(zip(list_area,list_rgiid),reverse=True)]
list_area = sorted(list_area,reverse=True)

ax = fig.add_subplot(grid[:2, 7:])

kval = 3.5

# sites=np.unique(data['Site'])
# colors=['b','g','r','c','m','y','k','grey']
colors = ['tab:blue','tab:orange','tab:red','tab:grey']
# sites=sites.tolist()
ax.plot([-2.5, 0.5], [-2.5, 0.5], color='k', linestyle='-', linewidth=2)

label_list=[]
diff2 = []
list_area2 = []
for rgiid in list_rgiid:

    df_gp_rgiid = df_gp[df_gp.rgiid==rgiid]
    df_hr_rgiid = df_hr[df_hr.rgiid==rgiid]

    if df_hr_rgiid.category.values[0]=='matthias':
        col = colors[0]
    elif df_hr_rgiid.category.values[0]=='brian':
        col = colors[1]
    else:
        if df_hr_rgiid.site.values[0] in ['Chhota','Gangotri','Abramov','Mera']:
            col = colors[2]
        elif df_hr_rgiid.site.values[0] == 'Yukon':
            col=colors[3]
        elif df_hr_rgiid.site.values[0] == 'MontBlanc':
            col=colors[0]

    ax.errorbar(df_hr_rgiid.dhdt.values[0], df_gp_rgiid.dhdt.values[0],
            xerr=df_hr_rgiid.err_dhdt.values[0],
            yerr=df_gp_rgiid.err_dhdt.values[0],marker='o',mec='k',
            ms=kval*(df_hr_rgiid.area.values[0]/1000000)**0.5, mew=0.25,elinewidth=0.5,ecolor=col,mfc=col,alpha=0.9)
                 #,ecolor=colors[sites.index(data['Site'][value])]mfc=colors[sites.index(data['Site'][value])],alpha=0.5)
    diff2.append(df_hr_rgiid.dhdt.values[0]-df_gp_rgiid.dhdt.values[0])
    list_area2.append(df_hr_rgiid.area.values[0])

ax.text(-1.5,0,'Mean bias:\n'+str(np.round(np.nanmean(diff2),2))+'$\pm$'+str(np.round(2*nmad(diff2)/np.sqrt(len(diff2)),2))+' m yr$^{-1}$',ha='center',va='center',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7))

print(np.nanmean(diff2))
print(np.nansum(np.array(diff2)*np.array(list_area2))/np.nansum(np.array(list_area2)))

ax.set_ylabel('GP mean elevation change (m yr$^{-1}$)')
ax.set_xlabel('High-resolution DEMs mean elevation change (m yr$^{-1}$)')

#plt.legend(loc='upper left')
ax.set_xlim([-2.75, 0.5])
ax.set_ylim([-2.75, 0.5])

#mask = ~np.isnan(b_dot_anomaly) & ~np.isnan(dP)
# slope, intercept, r_value, p_value, std_err = stats.linregress(data['MB GEOD'], data['MB ASTER'])
# print(slope)
# print("r-squared:", r_value**2)
# print('std err:', std_err)

# plt.text(-320, -1250, 'Slope:' + str(np.round(slope, 2)))
# plt.text(-320, -1300, 'r$^{2}$:' + str(np.round(r_value**2, 2)))


## add symbols to show relative size of glaciers
ax.errorbar(-2150/1000,-150/1000,ms = kval*(5.0**0.5), xerr=0.0001, yerr=0.0001, color='k',marker='o')
ax.errorbar(-2150/1000,-500/1000,ms = kval*(50.0**0.5), xerr=0.0001, yerr=0.0001,color='k',marker='o')
ax.errorbar(-2150/1000,-1250/1000,ms = kval*(500.0**0.5), xerr=0.0001, yerr=0.0001, color='k', marker='o')

ax.text(-2150/1000, -220/1000,'5 km$^2$',va='top',ha='center')
ax.text(-2150/1000, -650/1000,'50 km$^2$',va='top',ha='center')
ax.text(-2150/1000, -1730/1000,'500 km$^2$',va='top',ha='center')

ax.text(0.025,0.966,'c',transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')

ax.plot([],[],color=colors[0],label='Alps',lw=4)
ax.plot([],[],color=colors[1],label='Western North America',lw=4)
ax.plot([],[],color=colors[2],label='High Mountain Asia',lw=4)
ax.plot([],[],color=colors[3],label='Alaska',lw=4)
ax.plot([],[],color='k',label='1:1 line',lw=2)

ax.legend(loc='lower right',title='Validation of GP estimates\nwith high-resolution DEMs',title_fontsize=12)

ax = fig.add_subplot(grid[4:, 7:])
ax.text(0.025,0.966,'e',transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')

vec_err_dhdt=[0.1,0.2,0.4,0.6,0.8,1,1.5,2]

list_err_emp = []
list_err_the = []
bin_err = []
nb_95ci = []
nb_gla = []
for i in range(len(vec_err_dhdt)-1):

    ind = np.logical_and(df_gp.err_dhdt < vec_err_dhdt[i+1],df_gp.err_dhdt>=vec_err_dhdt[i])
    list_rgiid = list(df_gp[ind].rgiid)

    diff_dhdt = []
    err_dhdt = []
    ci_size = []
    for rgiid in list_rgiid:
        diff = df_hr[df_hr.rgiid==rgiid].dhdt.values[0] - df_gp[df_gp.rgiid==rgiid].dhdt.values[0]
        err = np.sqrt(df_hr[df_hr.rgiid==rgiid].err_dhdt.values[0]**2+df_gp[df_gp.rgiid==rgiid].err_dhdt.values[0]**2)
        err_dhdt.append(err)
        diff_dhdt.append(diff)
        if np.abs(diff) - 2 * np.abs(df_hr[df_hr.rgiid == rgiid].err_dhdt.values[0]) - 2 * np.abs(
                df_gp[df_gp.rgiid == rgiid].err_dhdt.values[0]) > 0:
            ci_too_small = 0
        elif ~np.isnan(diff):
            ci_too_small = 1
        else:
            ci_too_small = np.nan
        ci_size.append(ci_too_small)
    list_err_emp.append(nmad(diff_dhdt))
    list_err_the.append(np.nanmedian(err_dhdt))
    bin_err.append(np.mean((vec_err_dhdt[i+1],vec_err_dhdt[i])))
    nb_95ci.append(np.nansum(ci_size)/np.count_nonzero(~np.isnan(ci_size)))
    nb_gla.append(np.count_nonzero(~np.isnan(ci_size)))

    if i < 2:
        va_text = 'bottom'
        y_off = 0.1
        if i == 0:
            x_off = -0.05
        else:
            x_off = 0
    else:
        va_text = 'top'
        y_off = -0.1
    ax.text(bin_err[i]+x_off, list_err_emp[i] + y_off, str(nb_gla[i]) + ' gla.\n' + str(np.round(nb_95ci[i] * 100, 0)) + '%',
            va=va_text, ha='center')

ax.plot([0,2],[0,2],color='k',label='1:1 line',lw=2)
ax.plot(bin_err,list_err_emp,color='tab:blue',label='Error (1$\sigma$) comparison\nto HR elevation differences\n(printed: glacier number and percent of intersecting 95% CIs)',linestyle='dashed',marker='x')

ax.set_xlabel('Theoretical elevation change error (m yr$^{-1}$)')
ax.set_ylabel('Empirical elevation\nchange error (m yr$^{-1}$)')
ax.set_ylim((0,1.4))
ax.legend(loc='upper right',title='Validation of errors with varying error size',title_fontsize=12)


ax = fig.add_subplot(grid[2:4, 7:])
ax.text(0.025,0.966,'d',transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')

vec_area=[0.01,0.05,0.2,1,5,20,200,1500]

list_err_emp = []
list_err_the = []
bin_err = []
nb_95ci = []
nb_gla = []
for i in range(len(vec_area)-1):

    ind = np.logical_and(df_gp.area.values/1000000 < vec_area[i+1],df_gp.area.values/1000000>=vec_area[i])
    list_rgiid = list(df_gp[ind].rgiid)

    diff_dhdt = []
    err_dhdt = []
    ci_size = []
    for rgiid in list_rgiid:
        diff = df_hr[df_hr.rgiid==rgiid].dhdt.values[0] - df_gp[df_gp.rgiid==rgiid].dhdt.values[0]
        err = np.sqrt(df_hr[df_hr.rgiid==rgiid].err_dhdt.values[0]**2+df_gp[df_gp.rgiid==rgiid].err_dhdt.values[0]**2)
        diff_dhdt.append(diff)
        err_dhdt.append(err)
        if np.abs(diff) - 2 * np.abs(df_hr[df_hr.rgiid == rgiid].err_dhdt.values[0]) - 2 * np.abs(
                df_gp[df_gp.rgiid == rgiid].err_dhdt.values[0]) > 0:
            ci_too_small = 0
        elif ~np.isnan(diff):
            ci_too_small = 1
        else:
            ci_too_small = np.nan
        ci_size.append(ci_too_small)
    list_err_emp.append(nmad(diff_dhdt))
    list_err_the.append(np.nanmedian(err_dhdt))
    bin_err.append(np.mean((vec_area[i+1],vec_area[i])))
    nb_95ci.append(np.nansum(ci_size)/np.count_nonzero(~np.isnan(ci_size)))
    nb_gla.append(np.count_nonzero(~np.isnan(ci_size)))

    if i <2:
        va_text = 'top'
        y_off = -0.1
    else:
        va_text = 'bottom'
        y_off = 0.1
    ax.text(bin_err[i],list_err_emp[i]+y_off,str(nb_gla[i])+' gla.\n'+str(np.round(nb_95ci[i]*100,0))+'%',va=va_text,ha='center')

ax.plot(bin_err,list_err_the,color='black',label='Theoretical error (1$\sigma$):\nspatially integrated variograms',marker='x')
ax.plot(bin_err,list_err_emp,color='tab:blue',label='Empirical error (1$\sigma$):\ncomparison to HR elevation differences\n(printed: glacier number and percent of intersecting 95% CIs)',linestyle='dashed',marker='x')

ax.set_xscale('log')
ax.set_xlabel('Area (km$^{2}$)')
ax.set_ylabel('Elevation\nchange error (m yr$^{-1}$)')
ax.set_ylim([0,1.4])
ax.legend(loc='upper right',title='Validation of errors with varying glaciers area',title_fontsize=12)

plt.savefig('/home/atom/ongoing/work_worldwide/figures/revised/ED_Figure_5_2.png',dpi=400)
