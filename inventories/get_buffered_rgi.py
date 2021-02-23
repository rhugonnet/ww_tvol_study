"""
@author: hugonnet
buffer RGI: 10 km for processing stable terrain in GP series, and 200 km for display on Figures
"""

import gdal, ogr, osr
import pyddem.vector_tools as ot
import numpy as np

fn_shp_in = '/home/atom/data/inventory_products/RGI/rgi60_all.shp'
# fn_shp_in = '/home/atom/data/inventory_products/RGI/00_rgi60_neighb_merged/18_rgi60_NewZealand/rgi60_reg18.shp'
buffer_km = 200
fn_shp_out = '/home/atom/data/inventory_products/RGI/buff/rgi60_all_buff_'+str(buffer_km)+'.shp'
fn_shp_out_2 = '/home/atom/data/inventory_products/RGI/buff/rgi60_all_diss_'+str(buffer_km)+'.shp'


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


ds_shp_in = ogr.GetDriverByName('ESRI Shapefile').Open(fn_shp_in, 0)
ds_shp_out = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource(fn_shp_out)

srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
layer_out = ds_shp_out.CreateLayer('buff',srs=srs,geom_type=ogr.wkbPolygon)

layer_in = ds_shp_in.GetLayer()

multipoly = ogr.Geometry(ogr.wkbMultiPolygon)

for feature in layer_in:
    geom = feature.GetGeometryRef()

    print('Working on:' +feature.GetField('RGIId'))

    centroid = geom.Centroid()
    center_lon, center_lat, _ = centroid.GetPoint()
    epsg, utm = ot.latlon_to_UTM(center_lat,center_lon)

    trans = ot.coord_trans(False,4326,False,int(epsg))

    geom.Transform(trans)
    geom = geom.Buffer(buffer_km*1000)

    if utm == '01N' or utm == '01S':
        utm_ext = [(-179.999, -85), (-179.999, 85), (-168, 85), (-168, -85), (-179.999, -85)]
        utm_poly = ot.poly_from_coords(inter_poly_coords(utm_ext))

        utm_poly.Transform(trans)

        geom = geom.Intersection(utm_poly)
    elif utm == '60N' or utm == '60S':
        utm_ext = [(168, -85), (168, 85), (179.99, 85), (179.999, -85), (168, -85)]
        utm_poly = ot.poly_from_coords(inter_poly_coords(utm_ext))

        utm_poly.Transform(trans)

        geom = geom.Intersection(utm_poly)

    trans2 = ot.coord_trans(False,int(epsg),False,4326)

    geom.Transform(trans2)

    feat_out = ogr.Feature(layer_out.GetLayerDefn())
    feat_out.SetGeometry(geom)
    layer_out.CreateFeature(feat_out)

    multipoly.AddGeometry(geom)
#
# final_poly = multipoly.UnionCascaded()
#
ds_shp_out = None
#
# ds_shp_out_2 = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource(fn_shp_out_2)
#
# srs = osr.SpatialReference()
# srs.ImportFromEPSG(4326)
#
# layer_out = ds_shp_out_2.CreateLayer('buff', srs, ogr.wkbMultiPolygon)
# layer_out.CreateField(ogr.FieldDefn('FID', ogr.OFTInteger))
# defn = layer_out.GetLayerDefn()
# feat = ogr.Feature(defn)
# feat.SetField('FID', 1)
# geom = ogr.CreateGeometryFromWkt(multipoly.ExportToWkt())
# feat.SetGeometry(geom)
#
# layer_out.CreateFeature(feat)
# ds_shp_out_2 = None
#
# ds = layer = feat = geom = None


