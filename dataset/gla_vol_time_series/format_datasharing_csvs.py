import pandas as pd
import os

# script to format final files for data sharing

# PER GLACIER
# per glacier cumulative series
list_fn_cumul_pergla = ['/home/atom/ongoing/work_worldwide/vol/dh_06_rgi60_int_base.csv']

for fn_cumul_pergla in list_fn_cumul_pergla:
    df = pd.read_csv(fn_cumul_pergla,index_col=0)
    df = df.round({'dh':3,'err_dh':3,'dt':1,'std_dt':1,'perc_area_meas':3,'perc_area_res':3,'err_corr_150':3,'err_corr_2000':3,
                   'err_corr_5000':3,'err_corr_20000':3,'err_corr_50000':3,'err_corr_200000':3,'valid_obs':2,'valid_obs_py':2,
                   'area':0,'lon':4,'lat':4,'perc_err_cont':3})
    df = df[['rgiid','time','area','dh','err_dh','perc_area_meas','perc_area_res','valid_obs','valid_obs_py','dt','std_dt'
                ,'err_corr_150','err_corr_2000','err_corr_5000','err_corr_20000','err_corr_50000','err_corr_200000','lat','lon']]
    df.to_csv(os.path.join(os.path.dirname(fn_cumul_pergla),os.path.splitext(os.path.basename(fn_cumul_pergla))[0]+'_fmt.csv'))


# RGI O1 REGIONS WITH TAREA
# RGI O1 regional cumulative series
list_fn_cumul_reg = ['/home/atom/ongoing/work_worldwide/vol/final/dh_01_rgi60_int_base_reg.csv']
for fn_cumul_reg in list_fn_cumul_reg:
    df = pd.read_csv(fn_cumul_reg)
    df = df.drop(columns=['area_valid_obs_py','perc_err_cont'])
    df = df.round({'dh':3,'err_dh':3,'dvol':0,'err_dvol':0,'dm':4,'err_dm':4,'dt':1,'std_dt':1,'perc_area_meas':3,'perc_area_res':3
                   ,'valid_obs':2,'valid_obs_py':2, 'area':0,'area_nodata':0})
    df = df[['reg','time','area','dh','err_dh','dvol','err_dvol','dm','err_dm','perc_area_meas','perc_area_res','valid_obs','valid_obs_py','area_nodata']]
    df.to_csv(os.path.join(os.path.dirname(fn_cumul_reg),os.path.splitext(os.path.basename(fn_cumul_reg))[0]+'_fmt.csv'),index=None)

# RGI O1 regional rates
list_fn_rates_reg = ['/home/atom/ongoing/work_worldwide/vol/final/dh_01_rgi60_int_base_reg_subperiods.csv']
for fn_rates_reg in list_fn_rates_reg:
    df = pd.read_csv(fn_rates_reg,index_col=0)
    df = df.round({'dhdt':3,'err_dhdt':3,'dvoldt':0,'err_dvoldt':0,'dmdt':4,'err_dmdt':4,'dmdtda':3,'err_dmdtda':3,'perc_area_meas':3,'perc_area_res':3,
                   'valid_obs':2,'valid_obs_py':2,'area':0,'area_nodata':0, 'tarea':0})
    df = df[['reg','period','area','tarea','dhdt','err_dhdt','dvoldt','err_dvoldt','dmdt','err_dmdt','perc_area_meas','perc_area_res','valid_obs','valid_obs_py','area_nodata']]
    df.to_csv(os.path.join(os.path.dirname(fn_rates_reg),os.path.splitext(os.path.basename(fn_rates_reg))[0]+'_fmt.csv'),index=None)

# TILES
# tile cumulative series
list_fn_cumul_tile = ['/home/atom/ongoing/work_worldwide/vol/final/dh_world_tiles_2deg.csv']
for fn_cumul_tile in list_fn_cumul_tile:
    df = pd.read_csv(fn_cumul_tile)
    df = df.drop(columns=['area_valid_obs_py','perc_err_cont'])
    df = df.round({'dh':3,'err_dh':3,'dvol':0,'err_dvol':0,'dm':4,'err_dm':4,'dt':1,'std_dt':1,'perc_area_meas':3,'perc_area_res':3
                   ,'valid_obs':2,'valid_obs_py':2, 'area':0,'area_nodata':0,'tile_lonmin':1,'tile_latmin':1,'tile_size':1})
    df = df[['tile_lonmin','tile_latmin','tile_size','time','area','dh','err_dh','dvol','err_dvol','dm','err_dm','perc_area_meas','perc_area_res','valid_obs','valid_obs_py','area_nodata']]
    df.to_csv(os.path.join(os.path.dirname(fn_cumul_tile),os.path.splitext(os.path.basename(fn_cumul_tile))[0]+'_fmt.csv'),index=None)

# tile rates
list_fn_rates_tile = ['/home/atom/ongoing/work_worldwide/vol/final/dh_world_tiles_2deg_subperiods.csv']
for fn_rates_tile in list_fn_rates_tile:
    df = pd.read_csv(fn_rates_tile,index_col=0)
    df = df.drop(columns=['tarea'])
    df = df.round({'dhdt':3,'err_dhdt':3,'dvoldt':0,'err_dvoldt':0,'dmdt':4,'err_dmdt':4,'dmdtda':3,'err_dmdtda':3,'perc_area_meas':3,'perc_area_res':3,
                   'valid_obs':2,'valid_obs_py':2,'area':0,'area_nodata':0,'tile_lonmin':1,'tile_latmin':1,'tile_size':1})
    df = df[['tile_lonmin','tile_latmin','tile_size','period','area','dhdt','err_dhdt','dvoldt','err_dvoldt','dmdt','err_dmdt','perc_area_meas','perc_area_res','valid_obs','valid_obs_py','area_nodata']]
    df.to_csv(os.path.join(os.path.dirname(fn_rates_tile),os.path.splitext(os.path.basename(fn_rates_tile))[0]+'_fmt.csv'),index=None)

#SHP with TW/NTW sorting
# shp cumulative series
list_fn_cumul_shp = ['/home/atom/ongoing/work_worldwide/vol/final/subreg_HIMAP_cumul.csv']
for fn_cumul_shp in list_fn_cumul_shp:
    df = pd.read_csv(fn_cumul_shp)
    df = df.round({'dh':3,'err_dh':3,'dvol':0,'err_dvol':0,'dm':4,'err_dm':4,'dt':1,'std_dt':1,'perc_area_meas':3,'perc_area_res':3
                   ,'valid_obs':2,'valid_obs_py':2, 'area':0,'area_nodata':0})
    df = df[['subreg','time','area','dh','err_dh','dvol','err_dvol','dm','err_dm','perc_area_meas','perc_area_res','valid_obs','valid_obs_py','area_nodata']]
    df.to_csv(os.path.join(os.path.dirname(fn_cumul_shp),os.path.splitext(os.path.basename(fn_cumul_shp))[0]+'_fmt.csv'),index=None)

# shp rates
list_fn_rates_shp = ['/home/atom/ongoing/work_worldwide/vol/final/subreg_HIMAP_rates.csv']
for fn_rates_shp in list_fn_rates_shp:
    df = pd.read_csv(fn_rates_shp,index_col=0)
    df = df.drop(columns=['tarea'])
    df = df.round({'dhdt':3,'err_dhdt':3,'dvoldt':0,'err_dvoldt':0,'dmdt':4,'err_dmdt':4,'dmdtda':3,'err_dmdtda':3,'perc_area_meas':3,'perc_area_res':3,
                   'valid_obs':2,'valid_obs_py':2,'area':0,'area_nodata':0})
    df = df[['subreg','period','area','dhdt','err_dhdt','dvoldt','err_dvoldt','dmdt','err_dmdt','perc_area_meas','perc_area_res','valid_obs','valid_obs_py','area_nodata']]
    df.to_csv(os.path.join(os.path.dirname(fn_rates_shp),os.path.splitext(os.path.basename(fn_rates_shp))[0]+'_fmt.csv'),index=None)

