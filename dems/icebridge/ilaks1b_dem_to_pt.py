"""
@author: hugonnet
convert ILAKS1B DEMs to point data per region
"""

import os
import numpy as np
import pandas as pd
import datetime as dt
from pybob.GeoImg import GeoImg
import gdal, ogr
import pyddem.vector_tools as ot
from pybob.bob_tools import mkdir_p
import multiprocessing as mp

icebridge_dir = '/data/icesat/travail_en_cours/romain/data/dems/icebridge/ilaks1b/dem/'
output_dir = '/calcul/santo/hugonnet/icebridge/ilaks1b_point/'

# icebridge_dir = '/home/atom/ongoing/work_worldwide/icebridge/iodem3'
# output_dir = '/home/atom/ongoing/work_worldwide/icebridge/iodem3_pt'

out_res = 50
nproc = 64

# reg_dir = os.listdir(icebridge_dir)
# reg_dir = ['01_02_rgi60']

def point_to_lonlat_trans(epsg_in,list_tup):

    trans = ot.coord_trans(False,epsg_in,False,4326)
    list_tup_out = []
    for tup in list_tup:
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(tup[0],tup[1])
        point.Transform(trans)
        coord_out = point.GetPoint()[0:2]

        list_tup_out.append(coord_out)

    return list_tup_out

def parse_ilak_datetime(fn_icb):
    bname = os.path.basename(fn_icb)
    d = dt.datetime.strptime(bname.split('_')[1], '%Y%m%d')
    return d.date()

def parse_ilak_framenb(fn_icb):
    bname = os.path.basename(fn_icb)
    f = bname.split('_')[2]
    return f

def raster_to_point(fn_dem):

    extent, proj_wkt = ot.extent_rast(fn_dem)
    poly = ot.poly_from_extent(extent)
    transform = ot.coord_trans(True, proj_wkt, False, 4326)
    poly.Transform(transform)
    center_lon, center_lat = ot.get_poly_centroid(poly)

    epsg, utm_zone = ot.latlon_to_UTM(center_lat, center_lon)

    print('Reprojecting in '+str(epsg))

    img_vhr = GeoImg(fn_dem)

    dest = gdal.Warp('', img_vhr.gd, format='MEM', dstSRS='EPSG:{}'.format(epsg),
                     xRes=out_res, yRes=out_res, resampleAlg=gdal.GRA_Bilinear, dstNodata=-9999)

    img_lr = GeoImg(dest)

    print('Extracting coords...')

    elevs = img_lr.img.flatten()
    x, y = img_lr.xy(ctype='center')
    coords = list(zip(x.flatten(), y.flatten()))
    coords_latlon = point_to_lonlat_trans(int(epsg), coords)
    lon, lat = zip(*coords_latlon)
    lon = np.array(lon)
    lat = np.array(lat)

    keep = ~np.isnan(elevs)
    h = elevs[keep]
    lat = lat[keep]
    lon = lon[keep]

    print('Done for this DEM')

    return h, lat, lon

def wrapper_raster_to_point(argsin):

    fn_dem, i, imax = argsin
    print('Working: DEM ' + str(i + 1) +' out of '+str(imax))

    return raster_to_point(fn_dem)


all_dems = [os.path.join(icebridge_dir,fn) for fn in os.listdir(icebridge_dir) if fn.endswith('.tif')]
print('Found '+str(len(all_dems))+' DEMs.')

list_ak = []
list_ant = []
for fn_dem in all_dems:

    print('DEM number '+str(all_dems.index(fn_dem))+': '+fn_dem)

    img = GeoImg(fn_dem)

    if img.npix_x>5: #need to figure out why these LAS files don't process properly?

        extent, proj_wkt = ot.extent_rast(fn_dem)
        poly = ot.poly_from_extent(extent)
        print(proj_wkt)
        transform = ot.coord_trans(True, proj_wkt, False, 4326)
        poly.Transform(transform)
        center_lon, center_lat = ot.get_poly_centroid(poly)
        if center_lat > 0:
            list_ak.append(fn_dem)
        else:
            list_ant.append(fn_dem)

reg_dir = ['01_02_rgi60','19_rgi60']

for reg in reg_dir:

    print('Working on region: '+reg)

    df_reg = pd.DataFrame()

    if reg == '01_02_rgi60':
        list_dem = list_ak
    else:
        list_dem = list_ant
    list_dt = [parse_ilak_datetime(fn_dem) for fn_dem in list_dem]
    list_f = [parse_ilak_framenb(fn_dem) for fn_dem in list_dem]

    bin_dt = list(set(list_dt))

    if nproc ==1:
        for i, fn_dem in enumerate(list_dem):
            print('Working: DEM '+str(i+1)+' out of '+str(len(list_dem)))

            list_h, list_lat, list_lon = ([] for i in range(3))

            h, lat, lon = raster_to_point(fn_dem)

            list_h.append(h)
            list_lat.append(lat)
            list_lon.append(lon)
    else:
        print('Working on: '+str(nproc)+' cores...')
        argsin = [(fn_dem,i,len(list_dem)) for i, fn_dem in enumerate(list_dem)]
        pool = mp.Pool(nproc, maxtasksperchild=1)
        outputs = pool.map(wrapper_raster_to_point, argsin, chunksize=1)
        pool.close()
        pool.join()

        zipped = list(zip(*outputs))

        list_h = zipped[0]
        list_lat = zipped[1]
        list_lon = zipped[2]

    dfs = []
    for i in range(len(list_h)):
        df = pd.DataFrame()
        df = df.assign(h=list_h[i],lat=list_lat[i],lon=list_lon[i])
        df['t'] = list_dt[i]
        df['frame'] = list_f[i]
        dfs.append(df)

    df_reg = pd.concat(dfs)

    fn_out = os.path.join(output_dir,reg,'ILAKS1B_'+reg+'_'+str(out_res)+'_pt.csv')
    mkdir_p(os.path.dirname(fn_out))
    df_reg.to_csv(fn_out,index=None)