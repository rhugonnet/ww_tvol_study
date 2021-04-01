"""
@author: hugonnet
find ArcticDEM or REMA DEMs intersecting specific glaciers among the PGC archive
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
import tarfile

out_dir = '/data/icesat/travail_en_cours/romain/extracts/'

#example in South Georgia
nordens = ['RGI60-19.02274','nordens']
neumayer = ['RGI60-19.02483','neumayer']
risting = ['RGI60-19.02565','risting']

list_19 = [nordens,neumayer,risting]

setsm_dir = '/calcul/santo/hugonnet/setsm/2m'

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

def extract_file_from_tar_gz(tar_in,filename_in,file_out):

    #ref: https://stackoverflow.com/questions/37752400/how-do-i-extract-only-the-file-of-a-tar-gz-member
    with tarfile.open(tar_in, "r") as tar:
        counter = 0

        for member in tar:
            if member.isfile():
                filename = os.path.basename(member.name)
                if filename != filename_in:  # do your check
                    continue

                with open(file_out, "wb") as output:
                    print('Extracting '+filename_in + ' from archive '+tar_in+' to '+file_out+'...')
                    shutil.copyfileobj(tar.fileobj, output, member.size)


                break  # got our file

            counter += 1
            if counter % 1000 == 0:
                tar.members = []  # free ram... yes we have to do this manually


tiles = os.listdir(setsm_dir)

for tile in tiles:
    print('Searching for tile ' + tile + ' in folder ' + setsm_dir + '...')
    subtile_dir = os.path.join(setsm_dir, tile)
    seg_tar_gz_list = [os.path.join(subtile_dir, tar_file) for tar_file in os.listdir(subtile_dir) if
                       tar_file.endswith('.tar.gz')]
    print('Found ' + str(len(seg_tar_gz_list)) + ' segments in tile folder.')

    # 2/ EXTRACT ALL STRIPS

    tmp_dir = os.path.join(setsm_dir,tile, 'all_strips')

    mkdir_p(tmp_dir)

    list_tmp_dem = [os.path.join(tmp_dir, os.path.splitext(os.path.splitext(os.path.basename(seg_tar_gz))[0])[0] + '_dem.tif') for seg_tar_gz in seg_tar_gz_list]
    for seg_tar_gz in seg_tar_gz_list:
        print('Extracting dem file of segment ' + str(seg_tar_gz_list.index(seg_tar_gz) + 1) + ' out of ' + str(len(seg_tar_gz_list)))
        extract_file_from_tar_gz(seg_tar_gz, os.path.splitext(os.path.splitext(os.path.basename(seg_tar_gz))[0])[0] + '_dem.tif',
                                 list_tmp_dem[seg_tar_gz_list.index(seg_tar_gz)])


list_files = glob(os.path.join(setsm_dir,'**/*_dem.tif'),recursive=True)
fn_shp = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/regions/rgi60_merge.shp'

ds_shp = gdal.OpenEx(fn_shp, gdal.OF_VECTOR)
layer_name = os.path.splitext(os.path.basename(fn_shp))[0]
layer = ds_shp.GetLayer()
epsg_base = 4326

for gla in list_19:

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

    list_inters = get_footprints_inters_ext(list_files,poly,epsg_base,use_l1a_met=False)

    print('Found '+str(len(list_inters))+ ' DEMs out of '+str(len(list_files))+' intersecting glacier '+gla[1])

    for fn_inters in list_inters:

        print('Working on file:'+fn_inters)

        tmp_img = GeoImg(fn_inters)

        mask_feat = ot.geoimg_mask_on_feat_shp_ds(ds_shp, tmp_img, layer_name=layer_name, feat_id='RGIId',
                                                  feat_val=gla[0])
        # nb_px_mask = np.count_nonzero(mask_feat)
        nb_px_valid = np.count_nonzero(~np.isnan(tmp_img.img[mask_feat]))
        area_valid = nb_px_valid*tmp_img.dx**2/1000000
        cov = area_valid/area*100
        print('DEM ' + fn_inters + ' has intersection of ' + str(cov))
        if cov > 5.:
            list_final.append(os.path.basename(fn_inters))
            list_cov.append(cov)
            list_date.append(tmp_img.datetime)

            mt.create_zip_from_flist([fn_inters],os.path.join(gla_dir,os.path.splitext(os.path.basename(fn_inters))[0]+'.zip'))

    df = pd.DataFrame()
    df = df.assign(dem_name=list_final,perc_valid_coverage=list_cov,date=list_date)
    df.to_csv(os.path.join(gla_dir,'setsm_dem_coverage.csv'))



