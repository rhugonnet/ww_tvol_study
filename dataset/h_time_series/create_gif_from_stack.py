import xarray as xr
import pyddem.fit_tools as ft
import numpy as np
import matplotlib
matplotlib.use('Agg')

#!!this requires cartopy 0.18

# fn_stack='/home/atom/ongoing/work_worldwide/figures/esapolar/N63W020_final.nc'
# fn_shp = '/home/atom/data/inventory_products/RGI/00_rgi60_neighb_merged/06_rgi60_Iceland/06_rgi60_Iceland.shp'
fn_stack = '/data/icesat/travail_en_cours/romain/figures/Klinaklini_agu.nc'
fn_shp = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/00_rgi60_neighb_merged/01_02_rgi60_Alaska_WesternCanadaUS/02_rgi60_WesternCanadaUS.shp'
month_a_year=1
var='z'
# out_gif ='/home/atom/ongoing/work_worldwide/figures/esapolar/Myrdals_cumul.gif'
# out_gif_full = '/home/atom/ongoing/work_worldwide/figures/esapolar/Myrdals_rate.gif'
out_gif = '/data/icesat/travail_en_cours/romain/figures/Klinaklini_cumul.gif'
out_gif_full = '/data/icesat/travail_en_cours/romain/figures/Klinaklini_rates.gif'

ds = xr.open_dataset(fn_stack)
rates=True
t0=np.datetime64('2000-01-01')
t1=None
figsize=(15,15)
dh_max=100
cmap='RdYlBu'
xlbl='easting (km)'
ylbl='northing (km)'

fig, ims = ft.make_dh_animation(ds,fn_shp=fn_shp,t0=t0,month_a_year=1,dh_max=40,var='z',label='Elevation change since 2000 (m)')
ft.write_animation(fig, ims, outfilename=out_gif,interval=500)

fig, ims = ft.make_dh_animation(ds,fn_shp=fn_shp,t0=t0,month_a_year=1,rates=True,dh_max=2,var='z',label='Elevation change rate (m yr$^{-1}$)')
ft.write_animation(fig, ims, outfilename=out_gif_full,interval=500)
