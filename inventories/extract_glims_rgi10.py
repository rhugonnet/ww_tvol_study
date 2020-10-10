"""
@author: hugonnet
extract GLIMS outline to replace RGI in region 10
"""
from __future__ import print_function
import ogr, osr, gdal, gdalconst
import numpy as np
import os

fn_shp = '/home/atom/data/inventory_products/RGI/glims_rgi_region_12/glims_polygons.shp'
out_shp = '/home/atom/data/inventory_products/RGI/00_rgi60_neighb_merged/12_rgi60_CaucasusMiddleEast/12_rgi60_CaucasusMiddleEast.shp'

driver = ogr.GetDriverByName("ESRI Shapefile")
ds = driver.Open(fn_shp, 0)
layer = ds.GetLayer()
proj_shp = layer.GetSpatialRef()

# srs = osr.SpatialReference
# srs.ImportFromWkt(proj_shp)

list_glacid = []
list_date = []
for feat in layer:
    glims_id = feat.GetField('glac_id')
    date = feat.GetField('src_date')

    list_glacid.append(glims_id)
    list_date.append(np.datetime64(date))

unique_glacid = np.array(list(set(list_glacid)))
list_glacid =  np.array(list_glacid)
list_date = np.array(list_date)
max_date_by_unique_glacid = [max(list_date[list_glacid == glacid]) for glacid in unique_glacid]

layer.ResetReading()

if os.path.exists(out_shp):
    driver.DeleteDataSource(out_shp)

ds_out = driver.CreateDataSource(out_shp)
layer_out = ds_out.CreateLayer( os.path.splitext(os.path.basename(out_shp))[0], srs=proj_shp, geom_type=ogr.wkbMultiPolygon)

layer_defn = layer.GetLayerDefn()
for i in range(0, layer_defn.GetFieldCount()):
    fieldDefn = layer_defn.GetFieldDefn(i)
    fieldName = fieldDefn.GetName()
    layer_out.CreateField(fieldDefn)
layer_out_defn = layer_out.GetLayerDefn()

for feat in layer:

    glims_id = feat.GetField('glac_id')
    date = feat.GetField('src_date')

    for i in range(len(unique_glacid)):

        if glims_id == unique_glacid[i] and np.datetime64(date) == max_date_by_unique_glacid[i]:

            feat_out = ogr.Feature(layer_out_defn)

            for i in range(0, layer_out_defn.GetFieldCount()):
                field_defn = layer_out_defn.GetFieldDefn(i)
                field_name = field_defn.GetName()
                feat_out.SetField(layer_out_defn.GetFieldDefn(i).GetNameRef(),
                                    feat.GetField(i))

            geom = feat.GetGeometryRef()
            feat_out.SetGeometry(geom.Clone())
            layer_out.CreateFeature(feat_out)

ds.Destroy()
ds_out.Destroy()