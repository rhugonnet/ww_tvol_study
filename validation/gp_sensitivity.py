"""
@author: hugonnet
sensivitity of Gaussian Process parameters: running all in regions 6 and 8 with varying sets of parameters + integrating into volumes for all glaciers and region
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=6
import multiprocessing as mp
import pyddem.tdem_tools as tt
from glob import glob
import pyddem.fit_tools as ft
from pyddem.vector_tools import SRTMGL1_naming_to_latlon, latlon_to_UTM, niceextent_utm_latlontile
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic as RQ, ExpSineSquared as ESS, \
    PairwiseKernel
from pybob.bob_tools import mkdir_p

#run manually for region 6 and 8
region = '06_rgi60'
world_calc_dir = '/calcul/santo/hugonnet/worldwide/'
res = 100
y0 = 2000
nproc= 16
nfit = 1

tmp_ref_dir = os.path.join(world_calc_dir,region,'ref')

p_var = 30.
b_var = 50.
nl_var= 100.
nl_length =50.
nl_alpha=10.
b_length=1.

params = [p_var,b_var,nl_var,nl_length,nl_alpha,b_length]
params_name = ('pvar','bvar','nlvar','nllength','nlalpha','blength')

base_flist=[np.ones(6)]
base_nlist = ['base']

for i in np.arange(12):
    if (i - i % 6) >= 6:
        fac = 4
        fac_name = 'quadrice'
    else:
        fac = 0.25
        fac_name = 'quarter'

    j = i % 6
    tmp_flist = np.ones(6)
    tmp_flist[j] *= fac
    base_flist.append(tmp_flist)
    base_nlist.append(params_name[j]+'_'+fac_name)

for i in range(len(base_flist)):

    name = base_nlist[i]
    period_var, base_var, nonlin_var, nonlin_length, nonlin_alpha, base_length = base_flist[i]*params

    print(base_nlist[i])
    print(base_flist[i]*params)

    base_dir = os.path.join(world_calc_dir,region,'stacks')
    out_dir = os.path.join(world_calc_dir,region,'sensitivity',name,'stacks')
    ref_gla_csv = os.path.join('/calcul/santo/hugonnet/list_tiles','list_glacierized_tiles_'+region+'.csv')
    df = pd.read_csv(ref_gla_csv)
    tilelist = df['Tile_name'].tolist()

    def fit_tile_wrapper(arg_dict):

        return fit_tile(**arg_dict)

    def fit_tile(tile,tmp_ref_dir,base_dir,out_dir):

        method = 'gpr'
        subspat = None
        ref_dem_date = np.datetime64('2013-01-01')
        gla_mask = '/calcul/santo/hugonnet/outlines/rgi60_merge.shp'
        inc_mask = '/calcul/santo/hugonnet/outlines/rgi60_buff_10.shp'
        write_filt = True
        clobber = True
        tstep = 1./12.
        time_filt_thresh=[-50,50]
        opt_gpr = False
        filt_ref = 'both'
        filt_ls = False
        conf_filt_ls = 0.99
        # specify the exact temporal extent needed to be able to merge neighbouring stacks properly
        tlim = [np.datetime64('2000-01-01'), np.datetime64('2020-01-01')]


        #for sensitivity test: force final fit only, and change kernel parameters in entry of script
        force_final_fit = True
        k1 = PairwiseKernel(1, metric='linear')  # linear kernel
        k2 = C(period_var) * ESS(length_scale=1, periodicity=1)  # periodic kernel
        k3 = C(base_var * 0.6) * RBF(base_length*0.75) + C(base_var * 0.3) * RBF(base_length*1.5) + C(base_var * 0.1) * RBF(base_length*3)
        k4 = PairwiseKernel(1, metric='linear') * C(nonlin_var) * RQ(nonlin_length, nonlin_alpha)
        kernel = k1 + k2 + k3 + k4

        lat, lon = SRTMGL1_naming_to_latlon(tile)
        epsg, utm = latlon_to_UTM(lat, lon)
        print('Fitting tile: ' + tile + ' in UTM zone ' + utm)

        # reference DEM
        ref_utm_dir = os.path.join(tmp_ref_dir, utm)
        ref_vrt = os.path.join(ref_utm_dir, 'tmp_' + utm + '.vrt')
        infile = os.path.join(base_dir, utm, tile + '.nc')
        outfile = os.path.join(out_dir, utm, tile + '_final.nc')

        fn_filt = os.path.join(base_dir,utm,tile+'_filtered.nc')

        if True:#not os.path.exists(outfile):
            ft.fit_stack(infile, fn_filt=fn_filt, fit_extent=subspat, fn_ref_dem=ref_vrt, ref_dem_date=ref_dem_date, exc_mask=gla_mask,
                         tstep=tstep, tlim=tlim, inc_mask=inc_mask, filt_ref=filt_ref, time_filt_thresh=time_filt_thresh,
                         write_filt=True, outfile=outfile, method=method, filt_ls=filt_ls, conf_filt_ls=conf_filt_ls, nproc=nproc,
                         clobber=True,kernel=kernel,force_final_fit=force_final_fit)

            # write dh/dts for visualisation
            ds = xr.open_dataset(outfile)

            t0 = np.datetime64('2000-01-01')
            t1 = np.datetime64('2020-01-01')

            ft.get_full_dh(ds, os.path.join(os.path.dirname(outfile), os.path.splitext(os.path.basename(outfile))[0]),t0=t0, t1=t1)

        else:
            print('Tile already processed.')


    if nfit == 1:
        for tile in tilelist:
            fit_tile(tile,tmp_ref_dir,base_dir,out_dir)
    else:
        pool = mp.Pool(nfit)
        arg_dict = [{'tile':tile,'tmp_ref_dir':tmp_ref_dir,'base_dir':base_dir,'out_dir':out_dir} for tile in tilelist]
        pool.map(fit_tile_wrapper,arg_dict,chunksize=1)
        pool.close()
        pool.join()

    print('>>>Fin fit for parameters: '+name)

print('>>>FIN ELEVATION TIME SERIES')

list_dir = os.listdir(os.path.join(world_calc_dir,region,'sensitivity'))
print(list_dir)

for i in range(len(list_dir)):

    name = list_dir[i]

    print('Working on parameters:'+list_dir[i])

    feat_id='RGIId'
    tlim = None
    nproc=64
    dir_shp = '/calcul/santo/hugonnet/outlines/rgi60/00_rgi60_neighb_renamed'
    fn_base = '/calcul/santo/hugonnet/outlines/rgi60/base_rgi.csv'
    fn_tarea='/calcul/santo/hugonnet/outlines/rgi60/tarea_zemp.csv'

    region = '06_rgi60'
    print('Working on region: '+region)

    out_dir = os.path.join(world_calc_dir,region,'sensitivity',name,'stacks')
    results_dir = os.path.join(world_calc_dir,region,'sensitivity',name,'vol')

    print(results_dir)

    outfile = os.path.join(results_dir, 'dh_' + region + '_' + name + '.csv')

    if not os.path.exists(outfile):

        mkdir_p(results_dir)

        #integrate glaciers globally
        fn_shp = glob(os.path.join(dir_shp,'**/*'+region+'*.shp'),recursive=True)[0]
        dir_stack = out_dir
        list_fn_stack = glob(os.path.join(dir_stack, '**/*_final.nc'), recursive=True)
        tt.hypsocheat_postproc_stacks_tvol(list_fn_stack,fn_shp,nproc=nproc,outfile=outfile)

        # add info from base glacier dataframe (missing glaciers, identify region, etc...)
        infile = os.path.join(results_dir,'dh_'+region+'_'+name+'_int.csv')
        fn_int_base = os.path.join(os.path.dirname(infile),os.path.splitext(os.path.basename(infile))[0]+'_base.csv')
        tt.df_int_to_base(infile, fn_base=fn_base, outfile=fn_int_base)

        # aggregate by region
        infile_reg = os.path.join(os.path.dirname(fn_int_base),os.path.splitext(os.path.basename(fn_int_base))[0]+'_reg.csv')
        if not os.path.exists(infile_reg):
            tt.df_int_to_reg(fn_int_base,nproc=nproc)

        # aggregate by periods
        tt.df_region_to_multann(infile_reg, fn_tarea=fn_tarea)

    else:
        print('Already processed.')

print('>>> FIN INTEGRATION')









