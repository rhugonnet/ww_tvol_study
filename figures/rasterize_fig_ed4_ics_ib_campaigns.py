"""
@author: hugonnet
rasterize the mean z-score of differences to ICESat/IceBridge for Saint-Elias Moutains, ED Fig. 4
"""

import os, sys
import numpy as np
import pandas as pd
import gdal
from pybob.GeoImg import GeoImg
from pybob.ddem_tools import nmad
import ogr, osr
import pyddem.vector_tools as ot
import multiprocessing as mp

#
# def rasterize_list_point(list_point,proj_point,attr_val,in_met,i):
#
#     print('Poly stack number ' + str(i + 1))
#
#     # create input image
#     gt, proj, npix_x, npix_y = in_met
#     drv = gdal.GetDriverByName('MEM')
#     dst = drv.Create('', npix_x, npix_y, 1, gdal.GDT_Float32)
#     sp = dst.SetProjection(proj)
#     sg = dst.SetGeoTransform(gt)
#     band = dst.GetRasterBand(1)
#     band.SetNoDataValue(0)
#     band.Fill(0, 0)
#     del sp, sg
#
#     img = GeoImg(dst)
#
#     out_count = np.zeros(np.shape(img.img))
#     out_sum = np.zeros(np.shape(img.img))
#
#     srs=osr.SpatialReference()
#     srs.ImportFromWkt(proj_point)
#
#     for j, point in enumerate(list_point):
#
#         print('Point '+str(j + 1)+' out of '+str(len(list_point)))
#
#         attr='val'
#         ds_shp = ot.create_mem_shp(point, srs=srs, layer_type=ogr.wkbPoint, field_id=attr, field_val=attr_val[i],
#                                    field_type=ogr.OFTReal)
#         count = geoimg_rast_on_feat_shp_ds(ds_shp, img, burn=1)
#         val = geoimg_rast_on_feat_shp_ds(ds_shp, img, attr=attr)
#
#         out_count += count
#         out_sum += val
#
#     return out_count, out_sum
#
# def wrapper_rasterize(argdict):
#
#     return rasterize_list_point(**argdict)
#
# def create_raster_on_geoimg(geoimg,dtype=gdal.GDT_Byte):
#     tgt = gdal.GetDriverByName('MEM').Create('', geoimg.npix_x, geoimg.npix_y, 1, dtype)
#     tgt.SetGeoTransform((geoimg.xmin, geoimg.dx, 0, geoimg.ymax, 0, geoimg.dy))
#     tgt.SetProjection(geoimg.proj_wkt)
#     tgt.GetRasterBand(1).SetNoDataValue(-9999)
#     tgt.GetRasterBand(1).Fill(0)
#
#     return tgt
#
# def geoimg_rast_on_feat_shp_ds(shp_ds, geoimg,dtype=gdal.GDT_Byte,attr=None,burn=None):
#     ds_out = create_raster_on_geoimg(geoimg,dtype=dtype)
#     rasterize_shp_ds(shp_ds, ds_out,attr=attr,burn=burn)
#     rast = ds_out.GetRasterBand(1).ReadAsArray()
#     rast = rast.astype(float)
#
#     return rast
#
# def rasterize_shp_ds(shp_ds,raster_ds,attr=None,burn=None):
#
#     if attr is not None:
#         opts = gdal.RasterizeOptions(attribute=attr, bands=[1])
#     else:
#         if burn is None:
#             burn = 1
#         opts = gdal.RasterizeOptions(burnValues=[burn], bands=[1])
#
#     gdal.Rasterize(raster_ds, shp_ds, options=opts)
#
# def rasterize_add_attr_points(fn_ref,list_point,attr_val,proj_point,nproc=1):
#
#     #worldwide raster in lat/lon proj
#     ref = GeoImg(fn_ref)
#     # img = create_raster_on_geoimg(ref,dtype=gdal.GDT_Float32)
#     img = ref.copy()
#     img.img = np.zeros(np.shape(img.img))
#
#     out_count = np.zeros(np.shape(img.img))
#     out_sum = np.zeros(np.shape(img.img))
#
#     if nproc == 1:
#
#         for i,point in enumerate(list_point):
#
#             print('Rasterizing point number '+str(i+1)+' in '+str(len(list_point)))
#
#             attr='val'
#
#             ds_shp=ot.create_mem_shp(point,proj_point,layer_type=ogr.wkbPoint,field_id=attr,field_val=attr_val[i],field_type=ogr.OFTReal)
#             count = geoimg_rast_on_feat_shp_ds(ds_shp,img,burn=1)
#             val = geoimg_rast_on_feat_shp_ds(ds_shp,img,attr=attr)
#             print(np.mean(val))
#             print(np.std(val))
#
#             out_count += count
#             out_sum += val
#     else:
#         print('Using '+str(nproc)+' processors...')
#         # speed up things with multiprocessing
#         pool = mp.Pool(nproc, maxtasksperchild=1)
#         in_met = (img.gt, img.proj_wkt, img.npix_x, img.npix_y)
#
#         pack_size = int(np.ceil(len(list_point) / nproc))
#         argsin_packs = [{'list_point': list_point[i:min(i + pack_size, len(list_point))],'proj_point':proj_point.ExportToWkt()
#                             ,'attr_val':attr_val[i:min(i+pack_size,len(list_point))],'in_met':in_met,'i':k}
#                         for k, i in enumerate(np.arange(0, len(list_point), pack_size))]
#
#         outputs = pool.map(wrapper_rasterize,argsin_packs, chunksize=1)
#         pool.close()
#         pool.join()
#
#         zipped = list(zip(*outputs))
#
#         for i in range(len(zipped[0])):
#             out_count += zipped[0][i]
#             out_sum += zipped[1][i]
#
#
#     print('Here')
#     fn_out_count = '/home/atom/ongoing/work_worldwide/test_count.tif'
#     img.img = out_count
#     img.write(fn_out_count)
#
#     fn_out_sum = '/home/atom/ongoing/work_worldwide/test_sum.tif'
#     img.img = out_sum
#     img.write(fn_out_sum)

#
# fn_raster_ref = '/home/atom/ongoing/work_worldwide/dh_maps/01_02_rgi60_fig/fig3.vrt'
# fn_vrt_point = '/home/atom/ongoing/work_worldwide/validation/icesat/ala.vrt'
# fn_raster_out = '/home/atom/ongoing/work_worldwide/test.tif'
#
# ref = GeoImg(fn_raster_ref)
# img = ref.copy()
# img.img = np.zeros(np.shape(img.img))*np.nan
# img.write(fn_raster_out)
#
# tmp_shp = os.path.join(os.path.dirname(fn_raster_out),os.path.splitext(os.path.basename(fn_raster_out))[0]+'_tmp.shp')
# if os.path.exists(tmp_shp):
#     ogr.GetDriverByName('ESRI Shapefile').DeleteDataSource(tmp_shp)
#
# ds_shp_out = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource(tmp_shp)
#
# srs = osr.SpatialReference()
# srs.ImportFromEPSG(4326)
# layer_out = ds_shp_out.CreateLayer('buff', srs=srs, geom_type=ogr.wkbPoint25D)
# new_field = ogr.FieldDefn('val', ogr.OFTReal)
# layer_out.CreateField(new_field)
#
# fn_svalbard = '/home/atom/ongoing/work_worldwide/validation/icesat/valid_ICESat_01_0_rgi50_Alaska.csv'
# df = pd.read_csv(fn_svalbard)
# # ind = np.logical_and.reduce((df.lon>-145,df.lon<-137,df.lat>59,df.lat<62))
# ind = np.logical_and.reduce((df.lon>-140,df.lon<-139,df.lat>60,df.lat<60.5))
# df = df[ind]
# nmad_zsc = nmad(df.zsc)
# ind = np.abs(df.zsc)<nmad_zsc
# print('Removing with NMAD: '+str(np.count_nonzero(~ind))+' out of '+str(len(ind)))
# df = df[ind]
# df = df[df.pos==2]
# coords = list(zip(df.lon.values,df.lat.values))
# vals = df.zsc.values
# for coord in coords:
#     print('Working on point: ' + str(coords.index(coord) + 1) + ' out of ' + str(len(coords)))
#
#
#     point = ogr.Geometry(ogr.wkbPoint25D)
#     point.AddPoint(coord[0], coord[1])
#     feat_out = ogr.Feature(layer_out.GetLayerDefn())
#     feat_out.SetGeometry(point)
#     feat_out.SetField('val', vals[coords.index(coord)])
#     layer_out.CreateFeature(feat_out)
#     feat_out = None
#
# ds_shp_out = None
# opts = gdal.GridOptions(format='GTiff', width=ref.npix_x, height=ref.npix_y,
#                         outputBounds=[ref.xmin, ref.ymax, ref.xmax, ref.ymin], outputSRS='EPSG:' + str(ref.epsg),
#                         noData=-9999.0, algorithm='average:radius1=50.0:radius2=50.0:nodata=-9999.0',zfield='val')
# # opts = gdal.GridOptions(algorithm='average:radius1=50.0:radius2=50.0:nodata=-9999.0',zfield='val')
# output = gdal.Grid(fn_raster_out, tmp_shp, options=opts)
#
# arr = output.GetRasterBand(1).ReadAsArray()
# img.img = arr
#
# img.write(fn_raster_out)
#



#
# def rasterize_mean_shp_point(pt_coords,pt_vals,fn_raster_out,fn_hr_out,res,final_res,fn_ref_raster=None):
#
#     tmp_shp = os.path.join(os.path.dirname(fn_raster_out),os.path.splitext(os.path.basename(fn_raster_out))[0]+'_tmp.shp')
#     if os.path.exists(tmp_shp):
#         ogr.GetDriverByName('ESRI Shapefile').DeleteDataSource(tmp_shp)
#
#     ds_shp_out = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource(tmp_shp)
#
#     srs = osr.SpatialReference()
#     srs.ImportFromEPSG(4326)
#     layer_out = ds_shp_out.CreateLayer('buff', srs=srs, geom_type=ogr.wkbPoint)
#     new_field = ogr.FieldDefn('val', ogr.OFTReal)
#     layer_out.CreateField(new_field)
#
#     for coord in coords:
#         print('Working on point: ' + str(coords.index(coord) + 1) + ' out of ' + str(len(coords)))
#
#         point = ogr.Geometry(ogr.wkbPoint)
#         point.AddPoint(coord[0], coord[1])
#         feat_out = ogr.Feature(layer_out.GetLayerDefn())
#         feat_out.SetGeometry(point)
#         feat_out.SetField('val', vals[coords.index(coord)])
#         layer_out.CreateFeature(feat_out)
#         feat_out = None
#
#     ds_shp_out = None
#
#     if fn_ref_raster is None:
#         lon = np.array(list(zip(*coords))[0])
#         lat = np.array(list(zip(*coords))[1])
#         ext = (np.nanmin(lon),np.nanmin(lat),np.nanmax(lon),np.nanmax(lat))
#         center = (np.nanmean(lon),np.nanmean(lat))
#
#         epsg, utm = ot.latlon_to_UTM(center[1],center[0])
#         trans = ot.coord_trans(False,4326,False,int(epsg))
#         poly = ot.poly_from_extent(ext)
#         poly.Transform(trans)
#
#         ext_utm = ot.extent_from_poly(poly)
#
#         wide = int(np.ceil(np.abs(ext_utm[2]-ext_utm[0])/res))
#         high = int(np.ceil(np.abs(ext_utm[3]-ext_utm[1])/res))
#
#         ds_out = gdal.GetDriverByName('GTiff').Create(fn_raster_out, wide, high, 1, gdal.GDT_Float32)
#         gt = (ext_utm[0], int(res), 0., ext_utm[3], 0., -int(res))
#         ds_out.SetGeoTransform(gt)
#         srs = osr.SpatialReference()
#         srs.ImportFromEPSG(int(epsg))
#         ds_out.SetProjection(srs.ExportToWkt())
#         band = ds_out.GetRasterBand(1)
#         band.SetNoDataValue(-9999)
#         band.Fill(0, 0)
#     else:
#
#         img = GeoImg(fn_ref_raster)
#         ds_out=create_mem_raster_on_geoimg(img)
#
#     # open .shp with GDAL
#     shp_ds = gdal.OpenEx(tmp_shp, gdal.OF_VECTOR)
#     # rasterize
#     opts = gdal.RasterizeOptions(options='ADD',attribute='val', bands=[1])
#     gdal.Rasterize(ds_out, shp_ds, options=opts)
#
#     shp_ds = None
#
#     final = GeoImg(ds_out)
#     arr = ds_out.GetRasterBand(1).ReadAsArray()
#     print(np.mean(arr))
#     print(np.std(arr))
#     print(np.mean(final.img))
#     print(np.std(final.img))
#     print('Count: '+str(np.count_nonzero(arr!=-9999)))
#     # final.img = arr
#     final.write(fn_raster_out)
#
#     # img_raw = GeoImg(fn_raster_out)
#     #
#     # dest = gdal.Warp('', img_raw.gd, format='MEM', dstSRS='EPSG:{}'.format(epsg),
#     #                  xRes=final_res, yRes=final_res, resampleAlg=gdal.GRA_Bilinear, dstNodata=-9999)
#     #
#     # img_vr = GeoImg(dest)
#     # img_vr.write(fn_hr_out)

#
# coords = list(zip(df.lon.values,df.lat.values))
# list_point = []
# for coord in coords:
#     point = ogr.Geometry(ogr.wkbPoint)
#     point.AddPoint(coord[0], coord[1])
#     list_point.append(point)
#
# vals = df.zsc.values
#
# proj_in = osr.SpatialReference()
# proj_in.ImportFromEPSG(4326)
#
# rasterize_add_attr_points(fn_ref_raster,list_point,vals,proj_in,nproc=1)


# res = 500.
# res_final = 100.
# fn_shp_out = '/home/atom/ongoing/work_worldwide/test.shp'
# fn_rast_out = '/home/atom/ongoing/work_worldwide/test.tif'
# fn_hr_out = '/home/atom/ongoing/work_worldwide/test_hr.tif'

# rasterize_mean_shp_point(coords,vals,fn_rast_out,fn_hr_out,500,100,fn_ref_raster=fn_ref_raster)

# latlon_coords_to_buff_point_shp(list(zip(df.lon.values,df.lat.values)),df.zsc.values,fn_shp_out,200.)



#AU FINAL:

fn_ref_raster = '/home/atom/ongoing/work_worldwide/dh_maps/01_02_rgi60_fig/fig3.vrt'

#get extent of interest
fn_icesat = '/home/atom/ongoing/work_worldwide/validation/old/icesat/valid_ICESat_01_0_rgi50_Alaska.csv'
df = pd.read_csv(fn_icesat,index_col=False)
ind = np.logical_and.reduce((df.lon>-145,df.lon<-137,df.lat>59,df.lat<62))
df = df[ind]
df = df[df.pos==2]
nmad_zsc = nmad(df.zsc)
ind = np.abs(df.zsc)<10*nmad_zsc
print('Removing with NMAD: '+str(np.count_nonzero(~ind))+' out of '+str(len(ind)))
df = df[ind]
df = df.drop(columns=['dh','dh_ref','curv','slp','dt','t','pos','h'])
df.to_csv('/home/atom/ongoing/work_worldwide/fig3_icesat_df.csv')

#same
fn_ib = '/home/atom/ongoing/work_worldwide/validation/icebridge/valid_ILAKS1B_01_02_rgi60_50_pt.csv'
df = pd.read_csv(fn_ib,index_col=False)
ind = np.logical_and.reduce((df.lon>-145,df.lon<-137,df.lat>59,df.lat<62))
df = df[ind]
df = df[df.pos==2]
nmad_zsc = nmad(df.zsc)
ind = np.abs(df.zsc)<10*nmad_zsc
print('Removing with NMAD: '+str(np.count_nonzero(~ind))+' out of '+str(len(ind)))
df = df[ind]
df = df.drop(columns=['dh','dh_ref','curv','slp','dt','t','pos','h'])
df.to_csv('/home/atom/ongoing/work_worldwide/fig3_icebridge_df.csv')

#reproj and save as ESRI in QGIS, takes too long in Python...

#"add option doesn't exist in Python bindings... can't seem to find a way to replicate it and keep the speed, so here I go:

#count
#"gdal_rasterize -l fig3_icesat_utm -add -burn 1 -ts 5656.0 5318.0 -init 0.0 -a_nodata -9999.0 -te 223478.0 6541348.0 789141.0 7073207.0 -ot Float32 -of GTiff fig3_icesat_utm.shp icesat_count.tif"

#attr
#"gdal_rasterize -l fig3_icesat_utm -add -a zsc -ts 5656.0 5318.0 -init 0.0 -a_nodata -9999.0 -te 223478.0 6541348.0 789141.0 7073207.0 -ot Float32 -of GTiff fig3_icesat_utm.shp icesat_sum.tif"

#and same for icebridge

#then
fn_count_icesat = '/home/atom/ongoing/work_worldwide/figures/fig3/icesat_count_800m.tif'
fn_sum_icesat = '/home/atom/ongoing/work_worldwide/figures/fig3/icesat_sum_800m.tif'

count_ics = GeoImg(fn_count_icesat)
sum_ics = GeoImg(fn_sum_icesat)

nodata = count_ics.img == 0.
mean_ics = np.zeros(np.shape(sum_ics.img))*np.nan

mean_ics[~nodata] = sum_ics.img[~nodata]/count_ics.img[~nodata]

out = count_ics.copy()
out.img = np.abs(mean_ics)
out.write('/home/atom/ongoing/work_worldwide/figures/fig3/zsc_icesat_800m.tif')


fn_count_ib = '/home/atom/ongoing/work_worldwide/figures/fig3/ib_count_300m.tif'
fn_sum_ib = '/home/atom/ongoing/work_worldwide/figures/fig3/ib_sum_300m.tif'

count_ics = GeoImg(fn_count_ib)
sum_ics = GeoImg(fn_sum_ib)

nodata = count_ics.img == 0.
mean_ics = np.zeros(np.shape(sum_ics.img))*np.nan

mean_ics[~nodata] = sum_ics.img[~nodata]/count_ics.img[~nodata]

out = count_ics.copy()
out.img = np.abs(mean_ics)
out.write('/home/atom/ongoing/work_worldwide/figures/fig3/zsc_icebridge_300m.tif')
