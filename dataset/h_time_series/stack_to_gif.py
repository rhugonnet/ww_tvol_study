from __future__ import print_function
import xarray as xr
import pyddem.fit_tools as ft
import numpy as np
import matplotlib
matplotlib.use('Agg')

# fn_stack='/data/icesat/travail_en_cours/romain/tmp/N60W149_final.nc'
fn_stack='/data/icesat/travail_en_cours/romain/N63W020_final.nc'
fn_shp = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/00_rgi60_neighb_merged/06_rgi60_Iceland/06_rgi60_Iceland.shp'

out_gif ='/data/icesat/travail_en_cours/romain/Myrdals_cumul.gif'
out_gif_full = '/data/icesat/travail_en_cours/romain/Myrdals_rate.gif'

ds = xr.open_dataset(fn_stack)

t0=np.datetime64('2000-01-01')

fig, ims = ft.make_dh_animation(ds,fn_shp=fn_shp,t0=t0,month_a_year=1,dh_max=100,var='z')
ft.write_animation(fig, ims, outfilename=out_gif,interval=500)

fig, ims = ft.make_dh_animation(ds,fn_shp=fn_shp,t0=t0,month_a_year=1,rates=True,dh_max=5,var='z')
ft.write_animation(fig, ims, outfilename=out_gif_full,interval=500)
