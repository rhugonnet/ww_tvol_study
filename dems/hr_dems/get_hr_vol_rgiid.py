"""
@author: hugonnet
derive volume changes from all high-res differences of DEMs
"""

import os, sys
from rastlib import list_valid_feat_intersect
from glob import glob
import pyddem.vector_tools as ot
import gdal
from pybob.GeoImg import GeoImg
import numpy as np
import pandas as pd
from pybob.ddem_tools import nmad

def ddem_hypso(ddem,dem,mask,gsd,bin_type='fixed',bin_val=100.,filt='5NMAD'):

    final_mask = np.logical_and(np.logical_and(np.isfinite(ddem), np.isfinite(dem)),mask)

    std_stable = nmad(ddem[~mask])

    area_tot = np.count_nonzero(mask) * gsd ** 2

    if np.count_nonzero(final_mask)==0:
        return np.nan, np.nan, 0, area_tot

    dem_on_mask = dem[final_mask]
    ddem_on_mask = ddem[final_mask]

    min_elev = np.min(dem_on_mask) - (np.min(dem_on_mask) % bin_val)
    max_elev = np.max(dem_on_mask) + 1

    if bin_type == 'fixed':
        bin_final = bin_val
    elif bin_type == 'percentage':
        bin_final = np.ceil(bin_val / 100. * (max_elev - min_elev))
    else:
        sys.exit('Bin type not recognized.')

    bins_on_mask = np.arange(min_elev, max_elev, bin_final)
    nb_bin = len(bins_on_mask)

    elev_bin, nmad_bin, med_bin, mean_bin, std_bin, area_tot_bin, area_meas_bin = (np.zeros(nb_bin)*np.nan for i in range(7))

    for i in np.arange(nb_bin):

        idx_bin = np.array(dem_on_mask >= bins_on_mask[i]) & np.array(
            dem_on_mask < (bins_on_mask[i] + bin_final))
        idx_orig = np.array(dem >= bins_on_mask[i]) & np.array(
            dem < (bins_on_mask[i] + bin_final)) & mask
        area_tot_bin[i] = np.count_nonzero(idx_orig)*gsd**2
        area_meas_bin[i] = np.count_nonzero(idx_bin)*gsd**2
        elev_bin[i] = bins_on_mask[i] + bin_final / 2.
        dh_bin = ddem_on_mask[idx_bin]

        nvalid = len(dh_bin[~np.isnan(dh_bin)])
        if nvalid > 0:

            std_bin[i] = np.nanstd(dh_bin)
            med_bin[i] = np.nanmedian(dh_bin)
            if filt and nvalid > 10:
                median_temp = np.nanmedian(dh_bin)
                MAD_temp = np.nanmedian(np.absolute(dh_bin - median_temp))
                NMAD_temp = 1.4826 * MAD_temp
                nmad_bin[i] = NMAD_temp
                dh_bin[np.absolute(dh_bin - median_temp) > 5 * NMAD_temp] = np.nan
            mean_bin[i] = np.nanmean(dh_bin)


    area_meas = np.nansum(area_meas_bin)
    perc_area_meas = area_meas / area_tot
    idx_nonvoid = area_meas_bin > 0

    final_mean = np.nansum(mean_bin * area_tot_bin) / area_tot

    list_vgm = [(gsd*5, 'Sph', std_stable**2)]

    # list_vgm = [(corr_ranges[0],'Sph',final_num_err_corr[i,3]**2),(corr_ranges[1],'Sph',final_num_err_corr[i,2]**2),
    #             (corr_ranges[2],'Sph',final_num_err_corr[i,1]**2),(500000,'Sph',final_num_err_corr[i,0]**2)]
    neff_num_tot = ot.neff_circ(area_meas, list_vgm)
    final_err = ot.std_err(std_stable, neff_num_tot)

    print(final_mean)

    return final_mean, final_err, perc_area_meas, area_tot


fn_shp = '/home/atom/data/inventory_products/RGI/00_rgi60/rgi60_all.shp'
ds_shp = gdal.OpenEx(fn_shp, gdal.OF_VECTOR)
layer_name = os.path.splitext(os.path.basename(fn_shp))[0]

out_csv = '/home/atom/data/other/Hugonnet_2020/dhdt_int_HR_tmp.csv'

# list_dhdt_dir = ['/home/atom/data/other/Hugonnet_2020/Brian_LiDAR','/home/atom/data/other/Hugonnet_2020/Etienne_Fanny_Pléiades_SPOT','/home/atom/data/other/Hugonnet_2020/Matthias_2000_2020/DEMs_periods/final']
list_dhdt_dir = ['/home/atom/data/other/Hugonnet_2020/Matthias_2000_2020/DEMs_periods/final']
# category = ['brian','etienne','matthias']
category = ['matthias']
list_df_tmp = []
for dhdt_dir in list_dhdt_dir:

    print('Working on dDEMs in: '+dhdt_dir)

    # list_fn_dhdt = glob(os.path.join(dhdt_dir,'*.tif'))
    list_fn_dhdt = ['/home/atom/data/other/Hugonnet_2020/Matthias_2000_2020/DEMs_periods/final/dhdt_gor_AT_2015-08-26_AT_2007-09-13.tif']

    for fn_dhdt in list_fn_dhdt:

        print('Working on dDEM: '+fn_dhdt)

        list_rgiid_valid = list_valid_feat_intersect(fn_dhdt,fn_shp,'RGIId',70.)

        if len(list_rgiid_valid)>0:

            print('Found '+str(len(list_rgiid_valid))+' valid outlines intersecting')

            dhdt = GeoImg(fn_dhdt)

            split_fn = os.path.splitext(os.path.basename(fn_dhdt))[0].split('_')
            sens_early = split_fn[-2]
            sens_late = split_fn[-4]
            date_early = split_fn[-1]
            date_late = split_fn[-3]
            site = split_fn[1]

            for rgiid_valid in list_rgiid_valid:

                print('Working on '+rgiid_valid)

                dhdt.img[np.abs(dhdt.img)>15]=np.nan

                mask = ot.geoimg_mask_on_feat_shp_ds(ds_shp, dhdt, layer_name=layer_name, feat_id='RGIId',
                                                          feat_val=rgiid_valid)

                mean, err, perc, area = ddem_hypso(dhdt.img,np.ones(np.shape(dhdt.img)),mask,dhdt.dx)

                df_tmp = pd.DataFrame()
                df_tmp = df_tmp.assign(rgiid=[rgiid_valid],dhdt=[mean],err_dhdt=[err],area=[area],perc_meas=[perc],
                                       sensor_early=[sens_early],sensor_late=[sens_late],date_early=[date_early],
                                       date_late=[date_late],site=[site],category=[category[list_dhdt_dir.index(dhdt_dir)]])

                list_df_tmp.append(df_tmp)

df_out = pd.concat(list_df_tmp)
df_out.to_csv(out_csv)