
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
from pyddem.volint_tools import neff_circ, std_err
import functools
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid.inset_locator import inset_axes

plt.rcParams.update({'font.size': 5})
plt.rcParams.update({'lines.linewidth':0.35})
plt.rcParams.update({'axes.linewidth':0.35})
plt.rcParams.update({'lines.markersize':2.5})
plt.rcParams.update({'axes.labelpad':1.5})

all_csv = '/home/atom/ongoing/work_worldwide/validation/tcorr/tinterp_corr_deseas_agg_all.csv'
# all_csv = '/home/atom/ongoing/work_worldwide/validation/tinterp_corr_agg_all.csv'

df = pd.read_csv(all_csv)

# df = df[df.reg==5]

cutoffs = list(set(list(df.cutoff)))
dts = sorted(list(set(list(df.nb_dt))))

col = ['tab:orange','tab:blue','tab:olive','tab:red','tab:cyan','tab:brown','tab:gray','tab:pink','tab:purple']

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


fig = plt.figure(figsize=(7.2,9.3))

# plt.subplots_adjust(hspace=0.3)
grid = plt.GridSpec(8, 13, wspace=0.05, hspace=0.5)

ax = fig.add_subplot(grid[:2,:2])

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
ax.set_ylim((0,50))
ax.set_xticks([0,1,2])

ax.text(0.075, 0.975, 'a', transform=ax.transAxes,
        fontsize=8, fontweight='bold', va='top', ha='left')
ax.vlines(0.15,0,60,color=col[0],linewidth=0.5)
ax.text(0.4,c1s[1]-5,'$s_0$',color=col[0],ha='left',va='bottom',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7,linewidth=0.35))
ax.vlines(2,0,60,color=col[1],linewidth=0.5)
ax.text(2.2,c1s[2]-5,'$s_1$',color=col[1],ha='left',va='bottom',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7,linewidth=0.35))

ax.plot(vect/1000,mod,color='dimgrey',linestyle='dashed')
# ax.hlines(25,0,500,colors='black',linestyles='dotted')
ax.set_ylabel('Variance of elevation differences (m$^2$)')
ax.tick_params(width=0.35,length=2.5)

ax = fig.add_subplot(grid[:2,2:4])

ax.scatter(vec_bins/1000,vec_exp,color='black',marker='x')
ax.set_xlim((0,30))
ax.set_ylim((0,50))
ax.set_xticks([0,10,20])

# ax.text(0.075, 0.975, 'B', transform=ax.transAxes,
#         fontsize=14, fontweight='bold', va='top', ha='left')
ax.vlines(5,0,60,color=col[2],linewidth=0.5)
ax.text(6,c1s[3]-5,'$s_2$',color=col[2],ha='left',va='bottom',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7,linewidth=0.35))
ax.vlines(20,0,60,color=col[3],linewidth=0.5)
ax.text(21,c1s[4]-5,'$s_3$',color=col[3],ha='left',va='bottom',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7,linewidth=0.35))
ax.plot(vect/1000,mod,color='dimgrey',linestyle='dashed')
# ax.hlines(25,0,500,colors='black',linestyles='dotted',label='Global mean variance')

ax.set_yticks([])
ax.set_xlabel('Spatial lag (km)')
ax.tick_params(width=0.35,length=2.5)


ax = fig.add_subplot(grid[:2,4:6])

ax.scatter(vec_bins/1000,vec_exp,color='black',marker='x')
ax.set_xlim((0,550))
ax.set_ylim((0,50))
ax.set_xticks([0,100,200,300,400,500])

# ax.text(0.075, 0.975, 'C', transform=ax.transAxes,
#         fontsize=14, fontweight='bold', va='top', ha='left')
ax.vlines(50,0,60,colors=[col[4]],linewidth=0.5)
ax.text(70,c1s[5]-5,'$s_4$',color=col[4],ha='left',va='bottom',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7,linewidth=0.35))
ax.vlines(200,0,60,colors=[col[5]],linewidth=0.5)
ax.text(220,c1s[6]-7,'$s_5$',color=col[5],ha='left',va='bottom',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7,linewidth=0.35))
ax.vlines(500,0,60,colors=[col[6]],linewidth=0.5)
ax.text(480,c1s[6]-7,'$s_6$',color=col[6],ha='right',va='bottom',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7,linewidth=0.35))
ax.plot(vect/1000,mod,color='dimgrey',linestyle='dashed')
# ax.hlines(25,0,500,colors='black',linestyles='dotted')
ax.tick_params(width=0.35,length=2.5)
ax.plot([],[],color='grey',linestyle='dashed',label='Sum of spherical models')
ax.scatter([],[],color='black',marker='x',label='Empirical variance')
ax.vlines([],[],[],color=col[0],label='0.15 km',linewidth=0.5)
ax.vlines([],[],[],color=col[1],label='2 km',linewidth=0.5)
ax.vlines([],[],[],color=col[2],label='5 km',linewidth=0.5)
ax.vlines([],[],[],color=col[3],label='20 km',linewidth=0.5)
ax.vlines([],[],[],color=col[4],label='50 km',linewidth=0.5)
ax.vlines([],[],[],color=col[5],label='200 km',linewidth=0.5)
ax.vlines([],[],[],color=col[6],label='500 km',linewidth=0.5)
ax.legend(loc='lower right',ncol=3,title='Spatial correlations of GP elevation at $\Delta t$ = 720 days',title_fontsize=6,columnspacing=0.5)
ax.set_yticks([])



ax = fig.add_subplot(grid[2:4,:6])

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

ax.plot([],[],color='black',label='Model fit')
ax.fill_between([],[],color=col[0],label='0.15 km')
ax.fill_between([],[],color=col[1],label='2 km')
ax.fill_between([],[],color=col[2],label='5 km')
ax.fill_between([],[],color=col[3],label='20 km')
ax.scatter([],[],color='black',label='Empirical\nvariance')
ax.fill_between([],[],color=col[4],label='50 km')
ax.fill_between([],[],color=col[5],label='200 km')
ax.fill_between([],[],color=col[6],label='500 km')
ax.set_xlim([0,1370])
ax.set_ylim([0,78])
ax.set_ylabel('Variance of elevation differences (m$^{2}$)')
ax.set_xlabel('Days to closest observation $\Delta t$')
ax.vlines(720,0,100,colors='black',linestyles='dashed')
ax.text(740,5,'$\overline{s_{0}(\Delta t)}$: correlated until 0.15 km',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7,linewidth=0.35),color='tab:orange')
ax.text(800,22,'$s_{1}(\Delta t)$: correlated until 2 km',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7,linewidth=0.35),color='tab:blue')
ax.text(1150,35,'$s_{3}(\Delta t)$',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7,linewidth=0.35),color='tab:red')
ax.text(1250,48,'$s_{5}(\Delta t)$',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7,linewidth=0.35),color='tab:brown')

# ax.text(1000,22,'Fully correlated = Systematic',bbox= dict(boxstyle='round', facecolor='white', alpha=0.5),color='dimgrey')
# plt.xscale('log')
ax.legend(loc='upper left',bbox_to_anchor=(0.0625,0,0.9375,1),title='Spatial correlations of\nGP elevation with\ntime lag to observation',title_fontsize=6,ncol=2,columnspacing=0.5)
ax.text(0.025, 0.975, 'b', transform=ax.transAxes,
        fontsize=8, fontweight='bold', va='top', ha='left')
ax.text(740,45,'panel (a)',fontweight='bold',va='bottom',ha='left')
# plt.savefig('/home/atom/ongoing/work_worldwide/figures/Figure_S12.png',dpi=360)
ax.tick_params(width=0.35,length=2.5)


ax = fig.add_subplot(grid[4:6,:6])

corr_ranges = [150, 2000, 5000, 20000, 50000]
coefs = [np.array([1.26694247e-03, 3.03486839e+00]),
         np.array([1.35708936e-03, 4.05065698e+00]),
         np.array([1.42572733e-03, 4.20851582e+00]),
         np.array([1.82537137e-03, 4.28515920e+00]),
         np.array([1.87250755e-03, 4.31311254e+00]),
         np.array([2.06249620e-03, 4.33582812e+00])]
thresh = [0, 0, 0, 180, 180]
ind = [1, 1, 1, 2, 1]

def sill_frac(t, a, b, c, d):
    if t >= c:
        return (coefs[-1][0] * t + coefs[-1][1]) ** 2 - (a * t + b) ** 2 - (
                    (coefs[-1][1] + c * coefs[-1][0]) ** 2 - (coefs[-1 - d][1] + c * coefs[-1 - d][0]) ** 2)
    else:
        return 0

corr_std_dt = [functools.partial(sill_frac,a=coefs[i][0],b=coefs[i][1],c=thresh[i],d=ind[i]) for i in range(len(corr_ranges))]

list_areas = [100*2**i for i in np.arange(3,31)]

list_df=[]
for area in list_areas:

    dt = [180,540,900,1260]
    perc_area = [0.5,0.2,0.2,0.1]
    dx=100.

    nsamp_dt = np.zeros(len(dt)) * np.nan
    err_corr = np.zeros((len(dt), len(corr_ranges) + 1)) * np.nan

    for j in np.arange(len(dt)):

        final_num_err_dt = 10.
        nsamp_dt[j] = perc_area[j]*area

        sum_var = 0
        for k in range(len(corr_ranges)+1):

            if k != len(corr_ranges):
                err_corr[j,k] = np.sqrt(max(0,corr_std_dt[len(corr_ranges)-1-k](dt[j]) - sum_var))
                sum_var += err_corr[j,k] ** 2
            else:
                err_corr[j, k]=np.sqrt(max(0,final_num_err_dt**2-sum_var))


    final_num_err_corr, int_err_corr = (np.zeros( len(corr_ranges) + 1) * np.nan for i in range(2))
    for k in range(len(corr_ranges) + 1):
        final_num_err_corr[k] = np.sqrt(np.nansum(err_corr[:, k] * nsamp_dt) / np.nansum(nsamp_dt))

        if k == 0:
            tmp_length = 200000
        else:
            tmp_length = corr_ranges[len(corr_ranges) - k]

        if final_num_err_corr[k] == 0:
            int_err_corr[k] = 0
        else:
            int_err_corr[k] = std_err(final_num_err_corr[k],
                                         neff_circ(area, [(tmp_length, 'Sph', final_num_err_corr[k] ** 2)]))

    df_int = pd.DataFrame()
    for i in range(len(corr_ranges)):
        df_int['err_corr_'+str(corr_ranges[i])] =[int_err_corr[len(corr_ranges)-i]]
    df_int['err_corr_200000'] =[int_err_corr[0]]
    df_int['area']=area

    list_df.append(df_int)

df = pd.concat(list_df)

#First panel: sources for volume change
col = ['tab:orange','tab:blue','tab:olive','tab:red','tab:cyan','tab:brown','tab:gray','tab:pink','tab:purple']


tmp_y = np.zeros(len(list_areas))
tmp_y_next = np.zeros(len(list_areas))

for i in range(6):
    tmp_y = tmp_y_next
    tmp_y_next = tmp_y + (2*df.iloc[:len(list_areas),i])**2

    ax.fill_between(x=np.array(list_areas)/1000000,y1=tmp_y,y2=tmp_y_next,interpolate=True,color=col[i],alpha=0.5,edgecolor=None)
    if i == 0:
        ax.plot(np.array(list_areas)/1000000,tmp_y_next,color='black',linestyle='--')

ax.fill_between([],[],color=col[0],label='0.15 km',alpha=0.5)
ax.fill_between([],[],color=col[1],label='2 km',alpha=0.5)
ax.fill_between([],[],color=col[2],label='5 km',alpha=0.5)
ax.fill_between([],[],color=col[3],label='20 km',alpha=0.5)
ax.fill_between([],[],color=col[4],label='50 km',alpha=0.5)
ax.fill_between([],[],color=col[5],label='200 km',alpha=0.5)
ax.plot([],[],color='black',linestyle='--',label='Limit GP/spatial\ncorrelation sources')
ax.set_xscale('log')
ax.set_xlabel('Glacier area (km²)')
ax.set_ylabel('Squared uncertainties of\nspecific volume change (m²)')
ax.set_ylim((0,30))
ax.set_xlim((0.005,7.5*10**10/1000000))

handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
print(labels[0:2])
ax.legend(handles[0:2]+(handles[-1],)+handles[2:-1], labels[0:2]+(labels[-1],)+labels[2:-1],title='Uncertainty sources for specific volume change\n(i.e. mean elevation change)',title_fontsize=6,ncol=3,columnspacing=0.5)

ax.text(0.023,4*1.2,'Uncertainty \nsources from\npixel-wise\nGP regression\n(0.15 km)',color=plt.cm.Greys(0.8),va='center',ha='center')
ax.text(5,4*2,'Uncertainty sources from \nshort- to long-\nrange correlations\n(2 km - 200 km)',color=plt.cm.Greys(0.8),va='center',ha='center')
ax.text(0.025, 0.95, 'c', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top', ha='left')
ax.tick_params(width=0.35,length=2.5)




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
ax.plot([-3, 0.5], [-3, 0.5], color='k', linestyle='-', linewidth=0.75)

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
            ms=kval*(df_hr_rgiid.area.values[0]/1000000)**0.5/3, mew=0.25,elinewidth=0.25,ecolor=col,mfc=col,alpha=0.9)
                 #,ecolor=colors[sites.index(data['Site'][value])]mfc=colors[sites.index(data['Site'][value])],alpha=0.5)
    diff2.append(df_hr_rgiid.dhdt.values[0]-df_gp_rgiid.dhdt.values[0])
    list_area2.append(df_hr_rgiid.area.values[0])

ax.text(-1.9,0,'Mean bias:\n'+str(np.round(np.nanmean(diff2),2))+'$\pm$'+str(np.round(2*nmad(diff2)/np.sqrt(len(diff2)),2))+' m yr$^{-1}$',ha='center',va='center',bbox= dict(boxstyle='round', facecolor='white', alpha=0.7,linewidth=0.35))

print(np.nanmean(diff2))
print(np.nansum(np.array(diff2)*np.array(list_area2))/np.nansum(np.array(list_area2)))

ax.set_ylabel('Specific volume change (m yr$^{-1}$)')
ax.set_xlabel('High-resolution specific volume change (m yr$^{-1}$)')

#plt.legend(loc='upper left')
ax.set_xlim([-2.95, 0.5])
ax.set_ylim([-2.95, 0.5])

#mask = ~np.isnan(b_dot_anomaly) & ~np.isnan(dP)
# slope, intercept, r_value, p_value, std_err = stats.linregress(data['MB GEOD'], data['MB ASTER'])
# print(slope)
# print("r-squared:", r_value**2)
# print('std err:', std_err)

# plt.text(-320, -1250, 'Slope:' + str(np.round(slope, 2)))
# plt.text(-320, -1300, 'r$^{2}$:' + str(np.round(r_value**2, 2)))


## add symbols to show relative size of glaciers
ax.errorbar(-2500/1000,-150/1000,ms = kval*(5.0**0.5)/3, xerr=0.0001, yerr=0.0001, color='k',marker='o')
ax.errorbar(-2500/1000,-500/1000,ms = kval*(50.0**0.5)/3, xerr=0.0001, yerr=0.0001,color='k',marker='o')
ax.errorbar(-2500/1000,-1250/1000,ms = kval*(500.0**0.5)/3, xerr=0.0001, yerr=0.0001, color='k', marker='o')

ax.text(-2500/1000, -220/1000,'5 km$^2$',va='top',ha='center')
ax.text(-2500/1000, -650/1000,'50 km$^2$',va='top',ha='center')
ax.text(-2500/1000, -1730/1000,'500 km$^2$',va='top',ha='center')

ax.text(0.025,0.966,'d',transform=ax.transAxes,
        fontsize=8, fontweight='bold', va='top', ha='left')

ax.plot([],[],color=colors[0],label='Alps',lw=1)
ax.plot([],[],color=colors[1],label='Western NA',lw=1)
ax.plot([],[],color=colors[2],label='High Mountain Asia',lw=1)
ax.plot([],[],color=colors[3],label='Alaska',lw=1)
ax.plot([],[],color='k',label='1:1 line',lw=0.5)

ax.legend(loc='lower right',title='Validation of volume changes with high-resolution DEMs',title_fontsize=6,ncol=3)
ax.tick_params(width=0.35,length=2.5)

ax = fig.add_subplot(grid[4:6, 7:])
ax.text(0.025,0.966,'f',transform=ax.transAxes,
        fontsize=8, fontweight='bold', va='top', ha='left')

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

ax.plot([0,2],[0,2],color='k',label='1:1 line',lw=0.5)
ax.plot(bin_err,list_err_emp,color='tab:blue',label='Error (1$\sigma$) comparison to HR elevation differences\n(printed: glacier number and $\%$ of intersecting 95% CIs)',linestyle='dashed',marker='x')

ax.set_xlabel('Theoretical specific volume change uncertainty (m yr$^{-1}$)')
ax.set_ylabel('Empirical specific volume\nchange uncertainty (m yr$^{-1}$)')
ax.set_ylim((0,1.4))
ax.legend(loc='upper right',title='Validation of volume change uncertainties\nwith varying uncertainty size',title_fontsize=6)
ax.tick_params(width=0.35,length=2.5)


ax = fig.add_subplot(grid[2:4, 7:])
ax.text(0.025,0.966,'e',transform=ax.transAxes,
        fontsize=8, fontweight='bold', va='top', ha='left')

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

ax.plot(bin_err,list_err_the,color='black',label='Theoretical uncertainty (1$\sigma$):\nspatially integrated variograms',marker='x')
ax.plot(bin_err,list_err_emp,color='tab:blue',label='Empirical uncertainty (1$\sigma$):\ncomparison to HR elevation differences\n(printed: glacier number and\n$\%$ of intersecting 95% CIs)',linestyle='dashed',marker='x')

ax.set_xscale('log')
ax.set_xlabel('Glacier area (km$^{2}$)')
ax.set_ylabel('Specific volume\nchange uncertainty (m yr$^{-1}$)')
ax.set_ylim([0,1.4])
ax.legend(loc='upper right',title='Validation of volume change uncertainties\nwith varying glaciers area',title_fontsize=6)
ax.tick_params(width=0.35,length=2.5)

ax2 = fig.add_subplot(grid[6:,:])

reg_dir = '/home/atom/ongoing/work_worldwide/vol/final'
list_fn_reg = [os.path.join(reg_dir,'dh_'+str(i).zfill(2)+'_rgi60_int_base_reg.csv') for i in np.arange(1,20)]

list_df_out = []
for fn_reg in list_fn_reg:

    df = pd.read_csv(fn_reg)

    mult_ann = 20

    area = df.area.values[0]
    dvol = (df[df.time == '2000-01-01'].dvol.values - df[df.time == '2020-01-01'].dvol.values)[0]
    dh = dvol / area


    err_dh = np.sqrt(
        df[df.time == '2000-01-01'].err_dh.values[0] ** 2 +
        df[df.time == '2020-01-01'].err_dh.values[0] ** 2)
    err_dvol = np.sqrt((err_dh * area) ** 2 + (dh * df.perc_err_cont.values[0] / 100. * area) ** 2)

    dvoldt = dvol / mult_ann
    err_dvoldt = err_dvol / mult_ann

    dmdt = dvol * 0.85 / 10 ** 9 / mult_ann
    err_dmdt = np.sqrt((err_dvol * 0.85 / 10 ** 9) ** 2 + (
            dvol * 0.06 / 10 ** 9) ** 2) / mult_ann

    sq_err_dmdt_fromdh = (err_dh*area)**2 * (0.85 / mult_ann)**2 /area**2
    sq_err_dmdt_fromarea = (dh * df.perc_err_cont.values[0] / 100. * area) ** 2 * (0.85 / mult_ann)**2 /area**2
    sq_err_dmdt_fromdensity = (dvol * 0.06) ** 2 / mult_ann**2 / area**2

    dmdtda = dmdt/area*10**9

    df_out = pd.DataFrame()
    df_out['region']=[df.reg.values[0]]
    df_out['dmdtda'] = [dmdtda]
    df_out['sq_err_fromdh'] = [sq_err_dmdt_fromdh]
    df_out['sq_err_fromarea'] = [sq_err_dmdt_fromarea]
    df_out['sq_err_fromdensity'] = [sq_err_dmdt_fromdensity]
    df_out['area'] = [area]

    list_df_out.append(df_out)

df_all = pd.concat(list_df_out)

df_g = pd.DataFrame()
df_g['region']=[21]
df_g['dmdtda'] = [np.nansum(df_all.dmdtda.values*df_all.area.values)/np.nansum(df_all.area.values)]
df_g['sq_err_fromdh'] = [np.nansum(df_all.sq_err_fromdh.values * df_all.area.values **2)/np.nansum(df_all.area.values)**2]
df_g['sq_err_fromarea'] = [np.nansum(df_all.sq_err_fromarea.values * df_all.area.values **2)/np.nansum(df_all.area.values)**2]
df_g['sq_err_fromdensity'] = [np.nansum(df_all.sq_err_fromdensity.values * df_all.area.values **2)/np.nansum(df_all.area.values)**2]
df_g['area'] = [np.nansum(df_all.area.values)]

df_noper = pd.DataFrame()
ind = ~df_all.region.isin([5,19])
df_noper['region']=[20]
df_noper['dmdtda'] = [np.nansum(df_all[ind].dmdtda.values*df_all[ind].area.values)/np.nansum(df_all[ind].area.values)]
df_noper['sq_err_fromdh'] = np.nansum(df_all[ind].sq_err_fromdh.values * df_all[ind].area.values **2)/np.nansum(df_all[ind].area.values)**2
df_noper['sq_err_fromarea'] = np.nansum(df_all[ind].sq_err_fromarea.values * df_all[ind].area.values **2)/np.nansum(df_all[ind].area.values)**2
df_noper['sq_err_fromdensity'] = np.nansum(df_all[ind].sq_err_fromdensity.values * df_all[ind].area.values **2)/np.nansum(df_all[ind].area.values)**2
df_noper['area'] = [np.nansum(df_all[ind].area.values)]

df_all = pd.concat([df_all,df_noper,df_g])

ticks = ['Alaska (01)','Western Canada\nand USA (02)','Arctic Canada\nNorth (03)','Arctic Canada\nSouth (04)','Greenland\nPeriphery (05)', 'Iceland (06)','Svalbard and\nJan Mayen (07)', 'Scandinavia (08)','Russian\nArctic (09)','North Asia (10)','Central\nEurope (11)','Caucasus and\nMiddle East (12)','Central\nAsia (13)','South Asia\nWest (14)','South Asia\nEast (15)','Low\nLatitudes (16)','Southern\nAndes (17)','New\nZealand (18)','Antarctic and\nSubantarctic (19)','Global excl.\n 05 and 19','Global']

x_shift = 0

for i in np.arange(1,22):

    if i==20:
        x_shift+=2

    df_tmp = df_all[df_all.region==i]

    y1 = 4*df_tmp.sq_err_fromdh.values[0]
    y2 = y1 + 4*df_tmp.sq_err_fromarea.values[0]
    y3 = y2 + 4*df_tmp.sq_err_fromdensity.values[0]

    ax2.fill_between(x_shift+np.array((i,i+1)),(0,0),(y1,y1),color='tab:red',edgecolor='white')
    ax2.fill_between(x_shift+np.array((i,i+1)),(y1,y1),(y2,y2),color='tab:blue',edgecolor='white')
    ax2.fill_between(x_shift+np.array((i,i+1)),(y2,y2),(y3,y3),color='tab:pink',edgecolor='white')

ax2.fill_between([],[],color='tab:red',label='Elevation change')
ax2.fill_between([],[],color='tab:blue',label='Glacier outlines')
ax2.fill_between([],[],color='tab:pink',label='Density conversion')
ax2.text(0.025, 0.95, 'g', transform=ax2.transAxes, fontsize=8, fontweight='bold', va='top', ha='left')
ax2.set_ylabel('Squared uncertainties of\nspecific mass change rate (m² w.e. yr$^{-2}$)')
ax2.set_xlabel('RGI region')
ax2.legend(title='Uncertainty sources for\nspecific mass change\nduring 2000-2019',loc='upper right',bbox_to_anchor=(0.3,1),title_fontsize=6)

ax2.set_xticks(list(np.arange(1.5,20.5))+[22.5,23.5])
ax2.set_xticklabels(ticks,rotation=90)
ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
ax2.fill_between((22,24),(-0.00001,-0.00001),(4*0.000275,4*0.000275),facecolor='None',edgecolor='black')
ax2.text(23,4*0.0005,'panel (h)',fontweight='bold',va='bottom',ha='center')
ax2.tick_params(width=0.35,length=2.5)

ax3 = inset_axes(ax2,width="15%",height='50%',loc='upper right')
x_shift=0
for i in np.arange(20,22):

    if i==20:
        x_shift+=2

    df_tmp = df_all[df_all.region==i]

    y1 = 4*df_tmp.sq_err_fromdh.values[0]
    y2 = y1 + 4*df_tmp.sq_err_fromarea.values[0]
    y3 = y2 + 4*df_tmp.sq_err_fromdensity.values[0]

    ax3.fill_between(x_shift+np.array((i,i+1)),(0,0),(y1,y1),color='tab:red',edgecolor='white')
    ax3.fill_between(x_shift+np.array((i,i+1)),(y1,y1),(y2,y2),color='tab:blue',edgecolor='white')
    ax3.fill_between(x_shift+np.array((i,i+1)),(y2,y2),(y3,y3),color='tab:pink',edgecolor='white')
    ax3.set_xlim((22,24))
    ax3.set_xticks([22.5,23.5])
    ax3.set_ylim((-0.00001,4*0.000275))
    ax3.set_xticklabels(ticks[-2:],rotation=90)
    ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    ax3.text(0.9, 0.95, 'h', transform=ax3.transAxes, fontsize=8, fontweight='bold', va='top', ha='right')
    ax3.tick_params(width=0.35,length=2.5)


plt.savefig('/home/atom/ongoing/work_worldwide/figures/final/ED_Figure_5.jpg',dpi=500,bbox_inches='tight')
