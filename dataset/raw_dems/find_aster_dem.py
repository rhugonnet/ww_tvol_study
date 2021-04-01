"""
@author: hugonnet
find ASTER L1A DEMs intersecting specific glaciers among the corrected archive per region
"""

import os
import sys
from pybob.bob_tools import mkdir_p
import pyddem.vector_tools as ot
import pymmaster.mmaster_tools as mt
from glob import glob
import gdal, ogr
from pybob.GeoImg import GeoImg
import numpy as np
import pandas as pd
import shutil

out_dir = '/data/icesat/travail_en_cours/romain/extracts/'

#example for glacier of South Patagonia
ohiggins = ['RGI60-17.05184','ohiggins']
hps12 = ['RGI60-17.05022','hps12']
jorge = ['RGI60-17.06074','jorge']
hpn1 = ['RGI60-17.15899','hpn1']

list_17 = [ohiggins,hps12,jorge,hpn1]

aster_dir = '/data/icesat/travail_en_cours/romain/data/dems/aster_corr'
region = '17_rgi60'
in_dir = os.path.join(aster_dir,region)

def get_footprints_inters_ext(filelist, poly_base, epsg_base, use_l1a_met=False):

    list_poly = []
    for f in filelist:
        if use_l1a_met:
            poly=ot.l1astrip_polygon(os.path.dirname(f))
            trans=ot.coord_trans(False,4326,False,epsg_base)
            poly.Transform(trans)
        else:
            ext, proj = ot.extent_rast(f)
            poly = ot.poly_from_extent(ext)
            trans= ot.coord_trans(True,proj,False,epsg_base)
            poly.Transform(trans)
        list_poly.append(poly)

    poly_ext = poly_base

    filelist_out = []
    for poly in list_poly:
        if poly.Intersect(poly_ext):
            filelist_out.append(filelist[list_poly.index(poly)])

    return filelist_out

list_files = glob(os.path.join(in_dir,'**/AST*.zip'),recursive=True)
fn_shp = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/regions/rgi60_merge.shp'

ds_shp = gdal.OpenEx(fn_shp, gdal.OF_VECTOR)
layer_name = os.path.splitext(os.path.basename(fn_shp))[0]
layer = ds_shp.GetLayer()
epsg_base = 4326

for gla in list_17:

    list_final = []
    list_cov = []
    list_date = []

    print('Working on glacier: '+gla[1])

    gla_dir = os.path.join(out_dir,gla[1])
    mkdir_p(gla_dir)

    for feature in layer:
        if feature.GetField('RGIId') == gla[0]:
            poly = feature.GetGeometryRef()
            area = feature.GetField('Area')
            break
    layer.ResetReading()

    list_inters = get_footprints_inters_ext(list_files,poly,epsg_base,use_l1a_met=True)

    print('Found '+str(len(list_inters))+ ' DEMs out of '+str(len(list_files))+' intersecting glacier '+gla[1])

    for fn_inters in list_inters:

        print('Working on file:'+fn_inters)

        bname = os.path.splitext(os.path.basename(fn_inters))[0]
        splitname = bname.split('_')
        fn_z = '_'.join(splitname[0:3]) + '_Z_adj_XAJ_final.tif'
        fn_corr = '_'.join(splitname[0:3]) + '_CORR_adj_final.tif'

        fn_tmp = os.path.join(gla_dir,fn_z)
        fn_tmp_corr = os.path.join(gla_dir,fn_corr)
        mt.extract_file_from_zip(fn_inters,fn_z,fn_tmp)
        mt.extract_file_from_zip(fn_inters,fn_corr,fn_tmp_corr)

        tmp_img = GeoImg(fn_tmp)
        tmp_img_corr = GeoImg(fn_tmp_corr)
        tmp_img.img[tmp_img_corr.img<60.]=np.nan

        mask_feat = ot.geoimg_mask_on_feat_shp_ds(ds_shp, tmp_img, layer_name=layer_name, feat_id='RGIId',
                                                  feat_val=gla[0])
        nb_px_valid = np.count_nonzero(~np.isnan(tmp_img.img[mask_feat]))
        area_valid = nb_px_valid * tmp_img.dx ** 2 / 1000000
        cov = area_valid / area * 100
        print('DEM ' + fn_inters + ' has intersection of ' + str(cov))

        #set minimum coverage of 30% of the glacier area
        if cov > 30.:
            list_final.append(os.path.basename(fn_inters))
            list_cov.append(cov)
            list_date.append(tmp_img.datetime)

            fn_z_2 = '_'.join(splitname[0:3]) + '_Z_adj_XAJ_final_mincov30.tif'

            fn_out = os.path.join(gla_dir,fn_z_2)
            tmp_img.write(fn_out)
            mt.create_zip_from_flist([fn_out],os.path.join(gla_dir,os.path.splitext(os.path.basename(fn_out))[0]+'.zip'))
            os.remove(fn_out)

        os.remove(fn_tmp)
        os.remove(fn_tmp_corr)

    df = pd.DataFrame()
    df = df.assign(dem_name=list_final,perc_valid_coverage=list_cov,date=list_date)
    df.to_csv(os.path.join(gla_dir,'aster_dem_coverage.csv'))



