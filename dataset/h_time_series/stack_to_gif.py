from __future__ import print_function
import xarray as xr
import pyddem.fit_tools as ft
import numpy as np

fn_stack='/data/icesat/travail_en_cours/romain/data/stacks/old_06_rgi60/28N/N64W018_final.nc'
out_gif ='/data/icesat/travail_en_cours/romain/Vatnajokull_sept.gif'
out_gif_full = '/data/icesat/travail_en_cours/romain/Vatna_all.gif'

ds = xr.open_dataset(fn_stack)

t0=np.datetime64('2000-09-01')

fig, ims = ft.make_dh_animation(ds,t0=t0,month_a_year=None,dh_max=50,var='z')
ft.write_animation(fig, ims, outfilename=out_gif)

fig, ims = ft.make_dh_animation(ds,month_a_year=9,dh_max=50,var='z')
ft.write_animation(fig, ims, outfilename=out_gif_full)
