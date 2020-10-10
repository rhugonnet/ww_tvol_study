"""
@author: hugonnet
derive DEM differences from SwissTopo DEMs (high-resolution Alps DEMs)
"""
from __future__ import print_function

#MATTHIAS DEMs:
#gdalwarp -s_srs EPSG:21781 -t_srs EPSG:32632 -of GTiff -r bilinear dhm_alet2009.grid dhm_alet2009_utm.tif

import os, sys
from pybob.ddem_tools import difference
from glob import glob
import numpy as np

utm_dir = '/home/atom/data/validation/Matthias_2000_2020/DEMs_periods/utm'
names = ['alet','gor','gries','mort','plm','sil','rho','uaar']
date_1 = ['2009-09-08','2007-09-13','2012-08-27','2008-09-09','2012-09-14','2012-08-27','2000-08-24','2003-07-14']
date_2 = ['2017-08-29','2015-08-26','2018-08-19','2015-08-29','2018-08-28','2018-08-16','2007-09-12','2009-08-19']

gla_shp = '/home/atom/data/inventory_products/RGI/00_rgi60/rgi60_all.shp'

for n in [names[4]]:

    list_dems = glob(os.path.join(utm_dir,'*'+n+'*.tif'),recursive=True)


    dem_1 = list_dems[0]
    dem_2 = list_dems[1]
    year = dem_1.split(n)[1][0:4]

    if year == date_1[names.index(n)][0:4]:
        dem_early = dem_1
        dem_late = dem_2
    else:
        dem_early = dem_2
        dem_late = dem_1

    date_diff = (np.datetime64(date_2[names.index(n)])-np.datetime64(date_1[names.index(n)])).astype(int)/365.2524

    outdir = os.path.join(utm_dir,n+'_coreg')
    ddem1 = difference(dem_late, dem_early, glaciermask=gla_shp, outdir=outdir)
    fmask = np.greater(np.abs(ddem1.img), 100)
    ddem1.img[fmask] = np.nan
    ddem1.img[:] = ddem1.img[:]/date_diff
    ddem1.write(os.path.join(outdir, 'dhdt_'+n+'_'+date_2[names.index(n)]+'_'+date_1[names.index(n)]+'.tif'))