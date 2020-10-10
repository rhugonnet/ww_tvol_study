from __future__ import print_function
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fn_csv = '/home/atom/ongoing/work_Rounce/df_pergla_global_10yr_20yr.csv'
df=pd.read_csv(fn_csv)

def bin_var(var,var_bin,bin_vals,area_vals):

    mid_bin, median_bin, nmad_bin, ns_bin, area_bin = ([] for i in range(5))
    for i in range(len(bin_vals) - 1):
        ind = np.logical_and(var_bin >= bin_vals[i], var_bin < bin_vals[i + 1])
        if i == len(bin_vals) - 2:
            ind = np.logical_and(var_bin >= bin_vals[i], var_bin <= bin_vals[i + 1])
        binned_var = var[ind]

        mid_bin.append(bin_vals[i]+0.5*(bin_vals[i+1]-bin_vals[i]))
        median_bin.append(np.nanmedian(binned_var))
        nmad_bin.append(np.nanstd(binned_var))
        ns_bin.append(len(binned_var))
        area_bin.append(np.nansum(area_vals[ind]))

    df = pd.DataFrame()
    df = df.assign(mid_bin=mid_bin, med_bin = median_bin, nmad_bin = nmad_bin, ns_bin=ns_bin, area_bin=area_bin)

    return df

def plot_binned_with_histogram_on_top(df,bin_vals,xlab,ylab,cornlab,log=False,fn_out=None):

    #example; histogram written by hand, sufficient
    fig = plt.figure(figsize=(16,9))
    grid = plt.GridSpec(20, 15, wspace=0, hspace=0.8)

    ax0 = fig.add_subplot(grid[:4, :])
    # ax1=ax0.twinx()
    tot_count = np.sum(df.ns_bin.values)
    for i in range(len(bin_vals)-1):
        area = df.ns_bin.values[i]/tot_count*100
        ax0.fill_between([bin_vals[i],bin_vals[i+1]],[0]*2,[area]*2,facecolor=plt.cm.Blues(0.75),alpha=1,edgecolor='white')
        ax0.text(0.5*(bin_vals[i+1]+bin_vals[i]),area+2,str(np.round(area,2))+'%',ha='center',va='bottom')
    ax0.set_ylabel('Glacier count (%)')
    # ax0.text(0.05,0.95,'Nb gla:\n'+str(tot_count),transform=ax0.transAxes,ha='left',va='top',fontweight='bold')
    ax0.set_ylim(0,100)
    if log:
        ax0.set_xscale('log')
    ax0.vlines(bin_vals, 0, 100, colors='grey',
               alpha=0.7, linewidth=0.75, linestyles='dashed')
    ax0.set_xlim((bin_vals[0],bin_vals[-1]))
    # ax0.set_xticks([])

    ax1 = fig.add_subplot(grid[5:9, :])

    # ax1=ax0.twinx()
    tot_area = np.sum(df.area_bin.values)
    for i in range(len(bin_vals) - 1):
        area = df.area_bin.values[i] / tot_area * 100
        ax1.fill_between([bin_vals[i], bin_vals[i + 1]], [0] * 2, [area] * 2, facecolor=plt.cm.Reds(0.75), alpha=1,
                         edgecolor='white')
        ax1.text(0.5 * (bin_vals[i + 1] + bin_vals[i]), area + 2, str(np.round(area, 2)) + '%', ha='center',
                 va='bottom')
    ax1.set_ylabel('Glacier area (%)')
    # ax0.text(0.05,0.95,'Nb gla:\n'+str(tot_count),transform=ax0.transAxes,ha='left',va='top',fontweight='bold')
    ax1.set_ylim(0, 100)
    if log:
        ax1.set_xscale('log')
    ax1.vlines(bin_vals, 0, 100, colors='grey',
               alpha=0.7, linewidth=0.75, linestyles='dashed')
    ax1.set_xlim((bin_vals[0], bin_vals[-1]))
    # ax1.set_xticks([])

    ax = fig.add_subplot(grid[10:-1, :])

    ax.errorbar(df.mid_bin,df.med_bin,yerr=0.25*df.nmad_bin,label='median $\pm$ 0.25 std\n(for better display)')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if log:
        ax.set_xscale('log')
    ax.vlines(bin_vals, np.nanmin(df.med_bin)-2*np.nanmax(df.nmad_bin), np.nanmax(df.med_bin)+2*np.nanmax(df.nmad_bin), colors='grey',
              alpha=0.7, linewidth=0.75, linestyles='dashed')
    ax.set_ylim((np.nanmin(df.med_bin-0.25*df.nmad_bin),np.nanmax(df.med_bin+0.25*df.nmad_bin)))
    ax.set_xlim((bin_vals[0],bin_vals[-1]))
    ax.text(0.025, 0.95, cornlab, transform=ax.transAxes, ha='left', va='top', fontweight='bold', fontsize=12)
    ax.legend()
    if fn_out is not None:
        plt.savefig(fn_out,dpi=300)

# for i in np.arange(1,20):
#
#     #keep full period, region, remove CL2 Greenland glaciers with NaN area
#     df_reg = df[np.logical_and.reduce((df.period=='2000-01-01_2020-01-01',df.reg==i,~np.isnan(df.area)))]
#
#     # print('Region '+str(i))
#     # print(len(df_reg))
#     # print(np.count_nonzero(np.isnan(df_reg.perc_area_meas)))
#     ind_base = ~np.isnan(df_reg.perc_area_meas)
#     ind_filt1 = np.logical_and(df_reg.perc_area_meas>0.8,df_reg.err_dmdtda<0.5)
#     ind_filt2 = np.logical_and(df_reg.perc_area_meas>0.5,df_reg.err_dmdtda<0.75)
#     ind_filt3 = np.logical_and(df_reg.perc_area_meas>0.3,df_reg.err_dmdtda<1)
#
#     reg_tag = 'REGION '+str(i)+': \n'+"{:,}".format(len(df_reg))+\
#       ' glaciers\n'+"{:,}".format(np.count_nonzero(np.isnan(df_reg.perc_area_meas)))+' no-data glaciers\n'+\
#       'Base coverage: '+str(np.round(np.count_nonzero(ind_base)/len(ind_base)*100,2))+ '% glaciers, '+str(np.round(np.nansum(df_reg[ind_base].area)/np.nansum(df_reg.area)*100,2))+' % area\n'+\
#       'Filter >80%, 2$\sigma$ < 1 m w.e.: '+str(np.round(np.count_nonzero(ind_filt1)/len(ind_filt1)*100,2))+ '% glaciers, '+str(np.round(np.nansum(df_reg[ind_filt1].area)/np.nansum(df_reg.area)*100,2))+' % area\n'+\
#       'Filter >50%, 2$\sigma$ < 1.5 m w.e.: '+str(np.round(np.count_nonzero(ind_filt2)/len(ind_filt2)*100,2))+ '% glaciers, '+str(np.round(np.nansum(df_reg[ind_filt2].area)/np.nansum(df_reg.area)*100,2))+' % area\n'+\
#       'Filter >30%, 2$\sigma$ < 2 m w.e.: '+str(np.round(np.count_nonzero(ind_filt3)/len(ind_filt3)*100,2))+ '% glaciers, '+str(np.round(np.nansum(df_reg[ind_filt3].area)/np.nansum(df_reg.area)*100,2))+' % area'
#
#     bins_cov = np.arange(0, 105, 5)
#     df_cov = bin_var(df_reg.err_dmdtda, df_reg.perc_area_meas * 100, bins_cov, df_reg.area)
#     plot_binned_with_histogram_on_top(df_cov, bins_cov, 'Area measured (%)',
#                                       '1$\sigma$ mass change rate error (m w.e. yr$^{-1}$)',reg_tag,
#                                       fn_out=os.path.join(os.path.dirname(fn_csv),'reg_'+str(i)+'_cov.png'))
#
#     bins_err = [0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.75,1,2,5,10]
#     df_err = bin_var(df_reg.perc_area_meas*100, df_reg.err_dmdtda, bins_err, df_reg.area)
#     plot_binned_with_histogram_on_top(df_err, bins_err,
#                                       '1$\sigma$ mass change rate error (m w.e. yr$^{-1}$)','Area measured (%)', reg_tag,
#                                       fn_out=os.path.join(os.path.dirname(fn_csv), 'reg_' + str(i) + '_err.png'),log=True)
#
#     bins_area = [0, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000]
#     df_area = bin_var(df_reg.err_dmdtda, df_reg.area, bins_area,df_reg.area)
#     plot_binned_with_histogram_on_top(df_area, bins_area, 'Glacier size (km²)',
#                                       '1$\sigma$ mass change rate error (m w.e. yr$^{-1}$)',reg_tag,log=True,
#                                       fn_out=os.path.join(os.path.dirname(fn_csv),'reg_'+str(i)+'_area.png'))
#
#     bins_dmdtda = [-20,-5,-2.5,-1,-0.5,0,0.5,1,2.5,5,20]
#     df_dmdtda = bin_var(df_reg.err_dmdtda, df_reg.dmdtda, bins_dmdtda,df_reg.area)
#     plot_binned_with_histogram_on_top(df_dmdtda, bins_dmdtda, 'Mass change rate (m w.e. yr$^{-1}$)',
#                                       '1$\sigma$ mass change rate error (m w.e. yr$^{-1}$)',reg_tag,
#                                       fn_out=os.path.join(os.path.dirname(fn_csv),'reg_'+str(i)+'_mb.png'))



df = df[np.logical_and.reduce((df.period=='2000-01-01_2020-01-01',~np.isnan(df.area)))]
# ind_base = ~np.isnan(df.perc_area_meas)
# ind_filt1 = np.logical_and(df.perc_area_meas>0.8,df.err_dmdtda<0.5)
# ind_filt2 = np.logical_and(df.perc_area_meas>0.5,df.err_dmdtda<0.75)
# ind_filt3 = np.logical_and(df.perc_area_meas>0.3,df.err_dmdtda<1)
# df_cov = bin_var(df.err_dmdtda, df.perc_area_meas * 100, bins_cov,df.area)
#
# glob_tag = 'GLOBAL: \n'+"{:,}".format(len(df))+' glaciers\n'+"{:,}".format(np.count_nonzero(np.isnan(df.perc_area_meas)))+' no-data glaciers\n'+\
#     'Base coverage: '+str(np.round(np.count_nonzero(ind_base)/len(ind_base)*100,2))+ '% glaciers, '+str(np.round(np.nansum(df[ind_base].area)/np.nansum(df.area)*100,2))+' % area\n'+\
#     'Filter >80%, 2$\sigma$ < 1 m w.e.: '+str(np.round(np.count_nonzero(ind_filt1)/len(ind_filt1)*100,2))+ '% glaciers, '+str(np.round(np.nansum(df[ind_filt1].area)/np.nansum(df.area)*100,2))+' % area\n'+\
#     'Filter >50%, 2$\sigma$ < 1.5 m w.e.: '+str(np.round(np.count_nonzero(ind_filt2)/len(ind_filt2)*100,2))+ '% glaciers, '+str(np.round(np.nansum(df[ind_filt2].area)/np.nansum(df.area)*100,2))+' % area\n'+\
#     'Filter >30%, 2$\sigma$ < 2 m w.e.: '+str(np.round(np.count_nonzero(ind_filt3)/len(ind_filt3)*100,2))+ '% glaciers, '+str(np.round(np.nansum(df[ind_filt3].area)/np.nansum(df.area)*100,2))+' % area'
#
# plot_binned_with_histogram_on_top(df_cov, bins_cov, 'Area measured (%)',
#                                   '1$\sigma$ mass change rate error (m w.e. yr$^{-1}$)',glob_tag,
#                                   fn_out=os.path.join(os.path.dirname(fn_csv),'global_cov.png'))
#
# df_err = bin_var(df.perc_area_meas * 100, df.err_dmdtda, bins_err, df.area)
# plot_binned_with_histogram_on_top(df_err, bins_err,
#                                   '1$\sigma$ mass change rate error (m w.e. yr$^{-1}$)', 'Area measured (%)',glob_tag,
#                                   fn_out=os.path.join(os.path.dirname(fn_csv), 'global_err.png'),log=True)
#
# df_area = bin_var(df.err_dmdtda, df.area, bins_area,df.area)
# plot_binned_with_histogram_on_top(df_area, bins_area, 'Glacier size (km²)',
#                                   '1$\sigma$ mass change rate error (m w.e. yr$^{-1}$)',glob_tag,log=True,
#                                   fn_out=os.path.join(os.path.dirname(fn_csv),'global_area.png'))
#
# df_dmdtda = bin_var(df.err_dmdtda, df.dmdtda, bins_dmdtda,df.area)
# plot_binned_with_histogram_on_top(df_dmdtda, bins_dmdtda, 'Mass change rate (m w.e. yr$^{-1}$)',
#                                   '1$\sigma$ mass change rate error (m w.e. yr$^{-1}$)',glob_tag,
#                                   fn_out=os.path.join(os.path.dirname(fn_csv),'global_mb.png'))

df_area = bin_var(df.dmdtda,df.area,[0.005, 0.01,0.05,0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000],df.area)

plot_binned_with_histogram_on_top(df_area,[0.005, 0.01,0.05,0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000],'Area (km²)','Mass change (m w.e. yr$^{-1}$)','',log=True)
