"""
@author: hugonnet
retrieve ASTER, ArcticDEM and REMA strip coverage for Fig. S2
"""

import os
import pyddem.vector_tools as ot
from glob import glob
import gdal, osr
from pybob.GeoImg import GeoImg
import numpy as np
import multiprocessing as mp
from pyddem.stack_tools import parse_date
import pandas as pd

aster_dir = '/data/icesat/travail_en_cours/romain/data/dems/aster_corr/'
arcticdem_dir = '/data/icesat/travail_en_cours/romain/data/dems/arcticdem/'
rema_dir = '/data/icesat/travail_en_cours/romain/data/dems/rema_2m/'
nproc = 64

def rasterize_list_poly(list_poly,in_met,i):

    print('Poly stack number ' + str(i + 1))

    # create input image
    gt, proj, npix_x, npix_y = in_met
    drv = gdal.GetDriverByName('MEM')
    dst = drv.Create('', npix_x, npix_y, 1, gdal.GDT_Float32)
    sp = dst.SetProjection(proj)
    sg = dst.SetGeoTransform(gt)
    band = dst.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.Fill(0, 0)
    del sp, sg

    img = GeoImg(dst)

    out_density = np.zeros(np.shape(img.img))

    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)

    for j, poly in enumerate(list_poly):

        print('Poly '+str(j + 1)+' out of '+str(len(list_poly)))

        ds_shp = ot.create_mem_shp(poly,srs)
        mask = ot.geoimg_mask_on_feat_shp_ds(ds_shp, img)

        out_density[mask] += 1

    return out_density

def wrapper_rasterize(argdict):

    return rasterize_list_poly(**argdict)


def worldwide_coverage_density(list_poly,fn_out,res=0.05, nproc=1):

    #worldwide raster in lat/lon proj
    xmin = -180
    ymax = 90
    gt = (xmin, res, 0, ymax, 0, -res)
    npix_x = int(360/res)
    npix_y = int(180/res)
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(4326)

    ds_out = gdal.GetDriverByName('GTiff').Create(fn_out, npix_x, npix_y, 1, gdal.GDT_Int16)
    ds_out.SetGeoTransform(gt)
    ds_out.SetProjection(proj.ExportToWkt())
    band = ds_out.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.Fill(0, 0)

    img = GeoImg(ds_out)

    out_density = np.zeros(np.shape(img.img))
    if nproc == 1:

        for i,poly in enumerate(list_poly):

            print('Rasterizing poly number '+str(i+1)+' in '+str(len(list_poly)))

            ds_shp=ot.create_mem_shp(poly,proj)

            mask = ot.geoimg_mask_on_feat_shp_ds(ds_shp,img)
            out_density[mask] += 1
    else:
        print('Using '+str(nproc)+' processors...')
        # speed up things with multiprocessing
        pool = mp.Pool(nproc, maxtasksperchild=1)
        in_met = (img.gt, img.proj_wkt, img.npix_x, img.npix_y)

        pack_size = int(np.ceil(len(list_poly) / nproc))
        argsin_packs = [{'list_poly': list_poly[i:min(i + pack_size, len(list_poly))],'in_met':in_met,'i':k} for k, i in
                    enumerate(np.arange(0, len(list_poly), pack_size))]

        outputs = pool.map(wrapper_rasterize,argsin_packs, chunksize=1)
        pool.close()
        pool.join()

        for output in outputs:
            out_density += output

    ds_out.GetRasterBand(1).WriteArray(out_density)
    ds_out = None

def inter_poly_coords(polygon_coords):
    list_lat_interp = []
    list_lon_interp = []
    for i in range(len(polygon_coords) - 1):
        lon_interp = np.linspace(polygon_coords[i][0], polygon_coords[i + 1][0], 100)
        lat_interp = np.linspace(polygon_coords[i][1], polygon_coords[i + 1][1], 100)

        list_lon_interp.append(lon_interp)
        list_lat_interp.append(lat_interp)

    all_lon_interp = np.concatenate(list_lon_interp)
    all_lat_interp = np.concatenate(list_lat_interp)

    return np.array(list(zip(all_lon_interp, all_lat_interp)))




# #for ASTER
list_aster_dem = glob(os.path.join(aster_dir,'**/*_final.zip'),recursive=True)

aster_datelist = np.array([parse_date(aster_dem) for aster_dem in list_aster_dem])

df = pd.DataFrame()
df = df.assign(date=aster_datelist)
df.to_csv('/data/icesat/travail_en_cours/romain/tcov_aster.csv')

#
# print('Found '+str(len(list_aster_dem))+ ' ASTER DEMs.')
#
# list_aster_poly = []
# for i, aster_dem in enumerate(list_aster_dem):
#     print('Adding ASTER DEM extent poly '+str(i+1)+'...')
#     poly = ot.l1astrip_polygon(os.path.dirname(aster_dem))
#     list_aster_poly.append(poly)
#
# worldwide_coverage_density(list_aster_poly, '/data/icesat/travail_en_cours/romain/ww_cov_aster.tif',nproc=nproc)

list_arcticdem = glob(os.path.join(arcticdem_dir,'**/*.tif'),recursive=True)

arcticdem_datelist = np.array([parse_date(arcdem) for arcdem in list_arcticdem])

df = pd.DataFrame()
df = df.assign(date=arcticdem_datelist)
df.to_csv('/data/icesat/travail_en_cours/romain/tcov_arcticdem.csv')

# print('Found '+str(len(list_arcticdem))+ ' ArcticDEMs.')
#
# list_arcticdem_poly = []
# for i, arcticdem in enumerate(list_arcticdem):
#     print('Adding ArcticDEM extent poly '+str(i+1)+'...')
#     ext, proj = ot.extent_rast(arcticdem)
#     poly = ot.poly_from_extent(ext)
#     epsg = int(''.join(filter(lambda x: x.isdigit(), proj.split(',')[-1])))
#     utm = ot.utm_from_epsg(epsg)
#     print(utm)
#     trans0 = ot.coord_trans(False,4326,False,int(epsg))
#     trans = ot.coord_trans(True, proj, False, 4326)
#     if utm == '01N' or utm == '01S' or utm=='02N' or utm=='02S':
#         utm_ext = [(-179.9, -85), (-179.9, 85), (-162, 85), (-162, -85), (-179.9, -85)]
#         utm_poly = ot.poly_from_coords(inter_poly_coords(utm_ext))
#         utm_poly.Transform(trans0)
#         poly = poly.Intersection(utm_poly)
#     elif utm == '60N' or utm == '60S' or utm=='59N' or utm=='59S':
#         utm_ext = [(162, -85), (162, 85), (179.9, 85), (179.9, -85), (162, -85)]
#         utm_poly = ot.poly_from_coords(inter_poly_coords(utm_ext))
#         utm_poly.Transform(trans0)
#         poly = poly.Intersection(utm_poly)
#
#     poly.Transform(trans)
#
#     if max(poly.GetEnvelope()[0:2])>160 and min(poly.GetEnvelope()[0:2])<-160:
#         print('HEEEERE')
#     else:
#         list_arcticdem_poly.append(poly)
#
# worldwide_coverage_density(list_arcticdem_poly, '/data/icesat/travail_en_cours/romain/ww_cov_arcticdem_raw.tif',nproc=nproc)
#
# print('Fin ArcticDEM.')


list_arcticdem = glob(os.path.join(rema_dir,'**/*.tif'),recursive=True)

arcticdem_datelist = np.array([parse_date(arcdem) for arcdem in list_arcticdem])
df = pd.DataFrame()
df = df.assign(date=arcticdem_datelist)
df.to_csv('/data/icesat/travail_en_cours/romain/tcov_rema.csv')

# print('Found '+str(len(list_arcticdem))+ ' REMA DEMs.')
#
# list_rema_poly = []
# for i, arcticdem in enumerate(list_arcticdem):
#     print('Adding REMA DEM extent poly '+str(i+1)+'...')
#     ext, proj = ot.extent_rast(arcticdem)
#     poly = ot.poly_from_extent(ext)
#     epsg = int(''.join(filter(lambda x: x.isdigit(), proj.split(',')[-1])))
#     utm = ot.utm_from_epsg(epsg)
#     trans0 = ot.coord_trans(False,4326,False,int(epsg))
#     trans = ot.coord_trans(True, proj, False, 4326)
#     if utm == '01N' or utm == '01S':
#         utm_ext = [(-179.999, -85), (-179.999, 85), (-168, 85), (-168, -85), (-179.999, -85)]
#         utm_poly = ot.poly_from_coords(inter_poly_coords(utm_ext))
#         utm_poly.Transform(trans0)
#         poly = poly.Intersection(utm_poly)
#     elif utm == '60N' or utm == '60S':
#         utm_ext = [(168, -85), (168, 85), (179.99, 85), (179.999, -85), (168, -85)]
#         utm_poly = ot.poly_from_coords(inter_poly_coords(utm_ext))
#         utm_poly.Transform(trans0)
#         poly = poly.Intersection(utm_poly)
#
#     poly.Transform(trans)
#
#     list_rema_poly.append(poly)
#
# worldwide_coverage_density(list_rema_poly, '/data/icesat/travail_en_cours/romain/ww_cov_rema.tif',nproc=nproc)
#
# print('Fin REMA.')
