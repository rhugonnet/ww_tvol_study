from __future__ import print_function
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.patheffects import Stroke
import shapely.geometry as sgeom
import matplotlib.patches as mpatches
import os
import pandas as pd
import numpy as np
import gdal, osr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import ConvexHull
from pymmaster.other_tools import SRTMGL1_naming_to_latlon, latlon_to_SRTMGL1_naming
import pymmaster.other_tools as ot
from pybob.image_tools import create_mask_from_shapefile
from pybob.GeoImg import GeoImg
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
mpl.use('Agg')

# main_dir = '/home/atom/proj/ww_tvol_study/worldwide/'
# fn_land = '/home/atom/data/inventory_products/NaturalEarth/ne_50m_land/ne_50m_land.shp'
# fn_hs = '/home/atom/documents/paper/Hugonnet_2020/figures/world_robin_rs.tif'
# shp_buff = '/home/atom/data/inventory_products/RGI/00_rgi60/rgi60_buff_diss.shp'
# in_csv = '/home/atom/ongoing/work_worldwide/vol/reg/dh_world_tiles_1deg.csv'
# in_csv_2 =  '/home/atom/ongoing/work_worldwide/vol/tile/old_dh_world_tiles_1deg.csv'

period = '2000-01-01_2020-01-01'

# out_png = '/home/atom/ongoing/work_worldwide/figures/Figure_2_main.png'

fn_land = '/data/icesat/travail_en_cours/romain/figures/ne_50m_land.shp'
main_dir = '/data/icesat/travail_en_cours/romain/ww_tvol_study/worldwide/'
fn_hs = '/data/icesat/travail_en_cours/romain/figures/world_robin_rs.tif'
shp_buff = '/data/icesat/travail_en_cours/romain/figures/rgi60_buff_diss.shp'
out_png = '/data/icesat/travail_en_cours/romain/results/figures/Figure_2.png'
in_csv = '/data/icesat/travail_en_cours/romain/results/vol4/dh_world_tiles_1deg.csv'
in_csv_2 =  '/data/icesat/travail_en_cours/romain/results/vol4/old_dh_world_tiles_1deg.csv'

rgi_naming_txt = '/data/icesat/travail_en_cours/romain/ww_tvol_study/worldwide/rgi_neighb_merged_naming_convention.txt'
# rgi_naming_txt=os.path.join(main_dir,'rgi_neighb_merged_naming_convention.txt')
nb_tasks = 8

group_by_spec = True
def latlon_to_2x2_tile_naming(lat, lon):

    lon_sw = np.floor(lon / 2) * 2
    lat_sw = np.floor(lat/2) * 2
    if lat_sw >= 0:
        str_lat = 'N'
    else:
        str_lat = 'S'
    if lon_sw >= 0:
        str_lon = 'E'
    else:
        str_lon = 'W'
    tile_name_tdx = str_lat + str(int(abs(lat_sw))).zfill(2) + str_lon + str(int(abs(lon_sw))).zfill(3)
    return tile_name_tdx

def latlon_to_spec_tile_naming(lat,lon):

    if np.abs(lat)>=74:
        lon_sw=np.floor((lon-0.5)/2)*2
        lat_sw = np.floor((lat-0.5)/2)*2
    elif np.abs(lat) >=60:
        lon_sw = np.floor((lon-0.5)/ 2) * 2
        lat_sw = lat
    else:
        lon_sw = lon
        lat_sw = lat

    if lat_sw>=0:
        str_lat='N'
    else:
        str_lat='S'
    if lon_sw>=0:
        str_lon='E'
    else:
        str_lon='W'
    tile_name_tdx= str_lat + str(int(abs(lat_sw))).zfill(2) + str_lon + str(int(abs(lon_sw))).zfill(3)

    return tile_name_tdx

def latlon_to_spec_center(lat,lon):

    if np.abs(lat) >= 74:
        center_lon = lon + 1.5
        center_lat = lat + 1.5

    elif np.abs(lat) >= 60:
        center_lon = lon + 1.5
        center_lat = lat + 0.5

    else:
        center_lon = lon + 0.5
        center_lat = lat + 0.5


    return center_lat, center_lon

df_2 = pd.read_csv(in_csv_2)
df_2[np.logical_and.reduce((df_2.category=='all',df_2.period==period))]

df_all = pd.read_csv(in_csv)

ind = np.logical_and.reduce((df_all.category=='all',df_all.period==period))

df_all = df_all[ind]

filt = np.logical_or.reduce((df_all.perc_area_meas<0.7,df_all.valid_obs<4))

df_all.loc[filt,'dhdt']=np.nan

# tiles = df_all.tile.tolist()
tiles = [latlon_to_SRTMGL1_naming(df_all.tile_latmin.values[i],df_all.tile_lonmin.values[i]) for i in range(len(df_all))]

#to display grey on nominal glaciers, which are thrown out by the latest tiling script... (old one didn't)

tiles2 = [latlon_to_SRTMGL1_naming(df_2.tile_latmin.values[i],df_2.tile_lonmin.values[i]) for i in range(len(df_2))]
areas2 = df_2.area.tolist()
areas = df_all.area.tolist()
dhs = df_all.dhdt.tolist()
errs = df_all.err_dhdt.tolist()

for tile2 in tiles2:
    if tile2 not in tiles:
        tiles.append(tile2)
        areas.append(areas2[tiles2.index(tile2)])
        dhs.append(np.nan)
        errs.append(np.nan)

if group_by_spec:
    list_tile_grouped = []
    final_areas = []
    final_dhs = []
    final_errs = []
    for tile in tiles:
        lat, lon = SRTMGL1_naming_to_latlon(tile)
        list_tile_grouped.append(latlon_to_spec_tile_naming(lat,lon))
    final_tiles = list(set(list_tile_grouped))
    for tile in final_tiles:
        group_areas = np.array([ar for ar in areas if tile in list_tile_grouped[areas.index(ar)]])
        group_dhs = np.array([dh for dh in dhs if tile in list_tile_grouped[dhs.index(dh)]])
        group_errs = np.array([err for err in errs if tile in list_tile_grouped[errs.index(err)]])
        final_areas.append(np.nansum(group_areas))
        if np.count_nonzero(~np.isnan(group_dhs))!=0:
            final_dhs.append(np.nansum(group_dhs*group_areas)/np.nansum(group_areas))
        else:
            final_dhs.append(np.nan)
        final_errs.append(np.nansum(group_errs**2*group_areas**2)/np.nansum(group_areas)**2)
    tiles = final_tiles
    areas = final_areas
    dhs = final_dhs
    errs = final_errs

areas = [area/1000000 for _, area in sorted(zip(tiles,areas))]
dhs = [dh for _, dh in sorted(zip(tiles,dhs))]
errs = [err for _, err in sorted(zip(tiles,errs))]
tiles = sorted(tiles)

def latlon_extent_to_axes_units(extent):

    extent = np.array(extent)

    lons = (extent[0:2] + 179.9) / 359.8
    lats = (extent[2:4] + 89.9) / 179.8

    return [lons[0],lons[1],lats[0],lats[1]]

def axes_pos_to_rect_units(units):

    return [min(units[0:2]),min(units[2:4]),max(units[0:2])-min(units[0:2]),max(units[2:4])-min(units[2:4])]

def rect_units_to_verts(rect_u):

    return np.array([[rect_u[0],rect_u[1]],[rect_u[0]+rect_u[2],rect_u[1]],[rect_u[0]+rect_u[2],rect_u[1] +rect_u[3]],[rect_u[0],rect_u[1]+rect_u[3]],[rect_u[0],rect_u[1]]])

def coordXform(orig_crs, target_crs, x, y):
    return target_crs.transform_points( orig_crs, x, y )

def poly_from_extent(ext):

    poly = np.array([(ext[0],ext[2]),(ext[1],ext[2]),(ext[1],ext[3]),(ext[0],ext[3]),(ext[0],ext[2])])

    return poly

def latlon_extent_to_robinson_axes_verts(polygon_coords):

    list_lat_interp = []
    list_lon_interp = []
    for i in range(len(polygon_coords)-1):
        lon_interp = np.linspace(polygon_coords[i][0],polygon_coords[i+1][0],50)
        lat_interp =  np.linspace(polygon_coords[i][1],polygon_coords[i+1][1],50)

        list_lon_interp.append(lon_interp)
        list_lat_interp.append(lat_interp)

    all_lon_interp = np.concatenate(list_lon_interp)
    all_lat_interp = np.concatenate(list_lat_interp)

    robin = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),all_lon_interp,all_lat_interp)

    limits_robin = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([-179.99,179.99,0,0]),np.array([0,0,-89.99,89.99]))

    ext_robin_x = limits_robin[1][0] - limits_robin[0][0]
    ext_robin_y = limits_robin[3][1] - limits_robin[2][1]

    verts = robin.copy()
    verts[:,0] = (verts[:,0] + limits_robin[1][0])/ext_robin_x
    verts[:,1] = (verts[:,1] + limits_robin[3][1])/ext_robin_y

    return verts[:,0:2]

def shades_main_to_inset(main_pos,inset_pos,inset_verts,label):

    center_x = main_pos[0] + main_pos[2]/2
    center_y = main_pos[1] + main_pos[3]/2

    left_x = center_x - inset_pos[2]/2
    left_y = center_y - inset_pos[3]/2

    shade_ax = fig.add_axes([left_x,left_y,inset_pos[2],inset_pos[3]],projection=ccrs.Robinson(),label=label+'shade')
    shade_ax.set_extent([-179.99,179.99,-89.99,89.99],ccrs.PlateCarree())

    #first, get the limits of the manually positionned exploded polygon in projection coordinates
    limits_robin = coordXform(ccrs.PlateCarree(), ccrs.Robinson(), np.array([-179.99, 179.99, 0, 0]),
                              np.array([0, 0, -89.99, 89.99]))

    ext_robin_x = limits_robin[1][0] - limits_robin[0][0]
    ext_robin_y = limits_robin[3][1] - limits_robin[2][1]

    inset_mod_x = inset_verts[:,0] +  (inset_pos[0]-left_x)/inset_pos[2]
    inset_mod_y = inset_verts[:,1] +  (inset_pos[1]-left_y)/inset_pos[3]

    #then, get the limits of the polygon in the manually positionned center map
    main_mod_x = (inset_verts[:, 0]*main_pos[2] - left_x + main_pos[0])/inset_pos[2]
    main_mod_y = (inset_verts[:, 1]*main_pos[3] - left_y + main_pos[1])/inset_pos[3]

    points = np.array(list(zip(np.concatenate((inset_mod_x,main_mod_x)),np.concatenate((inset_mod_y,main_mod_y)))))

    chull = ConvexHull(points)

    chull_robin_x = points[chull.vertices,0]*ext_robin_x - limits_robin[1][0]
    chull_robin_y = points[chull.vertices,1]*ext_robin_y - limits_robin[3][1]

    col_contour = mpl.cm.Greys(0.8)

    shade_ax.plot(main_mod_x*ext_robin_x - limits_robin[1][0],main_mod_y*ext_robin_y - limits_robin[3][1],color='white',linewidth=1.5)
    shade_ax.fill(chull_robin_x, chull_robin_y, transform=ccrs.Robinson(), color='indigo', alpha=0.05, zorder=1)
    verts = mpath.Path(np.column_stack((chull_robin_x,chull_robin_y)))
    shade_ax.set_boundary(verts, transform=shade_ax.transAxes)

def only_shade(position,bounds,label,polygon=None):
    main_pos = [0.375, 0.21, 0.25, 0.25]

    if polygon is None and bounds is not None:
        polygon = poly_from_extent(bounds)

    shades_main_to_inset(main_pos, position, latlon_extent_to_robinson_axes_verts(polygon), label=label)

def add_inset(fig,extent,position,bounds=None,label=None,polygon=None,shades=True, hillshade=True, list_shp=None, main=False, markup=None,markpos='left',markadj=0,markup_sub=None,sub_pos='lt'):

    main_pos = [0.375, 0.21, 0.25, 0.25]

    if polygon is None and bounds is not None:
        polygon = poly_from_extent(bounds)

    if shades:
        shades_main_to_inset(main_pos, position, latlon_extent_to_robinson_axes_verts(polygon), label=label)

    sub_ax = fig.add_axes(position,
                          projection=ccrs.Robinson(),label=label)
    sub_ax.set_extent(extent, ccrs.Geodetic())

    sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor='gainsboro'))
    sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor='dimgrey'))

    if hillshade:
        def out_of_poly_mask(geoimg, poly_coords):

            poly = ot.poly_from_coords(inter_poly_coords(poly_coords))
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)

            # put in a memory vector
            ds_shp = ot.create_mem_shp(poly, srs)

            return ot.geoimg_mask_on_feat_shp_ds(ds_shp, geoimg)

        def inter_poly_coords(polygon_coords):
            list_lat_interp = []
            list_lon_interp = []
            for i in range(len(polygon_coords) - 1):
                lon_interp = np.linspace(polygon_coords[i][0], polygon_coords[i + 1][0], 50)
                lat_interp = np.linspace(polygon_coords[i][1], polygon_coords[i + 1][1], 50)

                list_lon_interp.append(lon_interp)
                list_lat_interp.append(lat_interp)

            all_lon_interp = np.concatenate(list_lon_interp)
            all_lat_interp = np.concatenate(list_lat_interp)

            return np.array(list(zip(all_lon_interp,all_lat_interp)))

        img = GeoImg(fn_hs)
        hs_tmp = hs_land.copy()
        hs_tmp_nl = hs_notland.copy()
        mask = out_of_poly_mask(img,polygon)

        hs_tmp[~mask] = 0
        hs_tmp_nl[~mask] = 0

        sub_ax.imshow(np.flip(hs_tmp[:, :],axis=0), extent=ext, transform=ccrs.Robinson(), cmap=cmap2, zorder=2)
        sub_ax.imshow(np.flip(hs_tmp_nl[:,:],axis=0),extent=ext, transform=ccrs.Robinson(),cmap=cmap22, zorder=2)

    if main:
        shape_feature = ShapelyFeature(Reader(list_shp).geometries(), ccrs.PlateCarree(),alpha=1,facecolor='indigo',linewidth=1,edgecolor='indigo')
        sub_ax.add_feature(shape_feature)

    if bounds is not None:
        verts = mpath.Path(latlon_extent_to_robinson_axes_verts(polygon))
        sub_ax.set_boundary(verts, transform=sub_ax.transAxes)

    if not main:
        for i in range(len(tiles)):
            lat, lon = SRTMGL1_naming_to_latlon(tiles[i])
            if group_by_spec:
                lat, lon = latlon_to_spec_center(lat,lon)
            else:
                lat = lat+0.5
                lon = lon+0.5

            if label=='Arctic West' and ((lat < 71 and lon > 60) or (lat <76 and lon>100)):
                continue

            if label=='HMA' and lat >=46:
                continue

            # fac = 0.02
            fac = 1000

            rad = 12000 + np.sqrt(areas[i]) * fac
            # cmin = -1
            # cmax = 1
            col_bounds = np.array([-1.5,-1,-0.8,-0.6,-0.4,-0.2,0,0.1,0.2,0.3,0.4,0.5,0.6])
            # col_bounds = np.array([-1, -0.7, -0.4, -0.2, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5])
            cb = []
            cb_val = np.linspace(0, 1, len(col_bounds))
            for j in range(len(cb_val)):
                cb.append(mpl.cm.RdYlBu(cb_val[j]))
            cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(
                zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), cb)), N=1000)

            if ~np.isnan(dhs[i]) and areas[i]>0.2 and errs[i]<0.5: #and ((areas[i]>=5.) or label in ['Mexico','Indonesia','Africa']):
                dhdt = dhs[i]
                dhdt_col = max(0.0001,min(0.9999,(dhdt - min(col_bounds))/(max(col_bounds)-min(col_bounds))))

                # ind = max(0, min(int((dhs[i]/20. - cmin) / (cmax - cmin) * 100), 99))
                # if dhs[i]>=0:
                #     ind = max(0, min(int((np.sqrt(dhs[i]/20.) - cmin) / (cmax - cmin) * 100), 99))
                # else:
                #     ind = max(0, min(int((-np.sqrt(-dhs[i]/20.) - cmin) / (cmax - cmin) * 100), 99))
                col = cmap_cus(dhdt_col)
            # elif areas[i] <= 5:
            #     continue
            elif areas[i]>0.2:
                col = plt.cm.Greys(0.7)
                # col = 'black'

            # xy = [lon,lat]
            xy = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([lon]),np.array([lat]))[0][0:2]

            sub_ax.add_patch(
                mpatches.Circle(xy=xy, radius=rad, color=col, alpha=1, transform=ccrs.Robinson(), zorder=30))
            # sub_ax.add_patch(
            #     mpatches.Circle(xy=xy, radius=rad, facecolor='None', edgecolor='dimgrey', alpha=1, transform=ccrs.Robinson(), zorder=30))

    if markup is not None:
        if markpos=='left':
            lon_upleft = np.min(list(zip(*polygon))[0])
            lat_upleft = np.max(list(zip(*polygon))[1])
        else:
            lon_upleft = np.max(list(zip(*polygon))[0])
            lat_upleft = np.max(list(zip(*polygon))[1])

        robin = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([lon_upleft]),np.array([lat_upleft]))

        rob_x = robin[0][0]
        rob_y = robin[0][1]

        size_y = 200000
        size_x = 80000 * len(markup) + markadj

        if markpos=='right':
            rob_x = rob_x-50000
        else:
            rob_x = rob_x+50000

        sub_ax_2 = fig.add_axes(position,
                                projection=ccrs.Robinson(), label=label+'markup')

        # sub_ax_2.add_patch(mpatches.Rectangle((rob_x, rob_y), size_x, size_y , linewidth=1, edgecolor='grey', facecolor='white',transform=ccrs.Robinson()))

        sub_ax_2.set_extent(extent, ccrs.Geodetic())
        verts = mpath.Path(rect_units_to_verts([rob_x,rob_y,size_x,size_y]))
        sub_ax_2.set_boundary(verts, transform=sub_ax.transAxes)

        sub_ax_2.text(rob_x,rob_y+50000,markup,
                 horizontalalignment=markpos, verticalalignment='bottom',
                 transform=ccrs.Robinson(), color='black',fontsize=12, fontweight='bold',bbox= dict(facecolor='white', alpha=1))

    if markup_sub is not None:

        lon_min = np.min(list(zip(*polygon))[0])
        lon_max = np.max(list(zip(*polygon))[0])
        lon_mid = 0.5*(lon_min+lon_max)

        lat_min = np.min(list(zip(*polygon))[1])
        lat_max = np.max(list(zip(*polygon))[1])
        lat_mid = 0.5*(lat_min+lat_max)

        size_y = 150000
        size_x = 150000

        robin = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([lon_min,lon_min,lon_min,lon_mid,lon_mid,lon_max,lon_max,lon_max]),np.array([lat_min,lat_mid,lat_max,lat_min,lat_max,lat_min,lat_mid,lat_max]))

        if sub_pos=='lb':
            rob_x = robin[0][0]
            rob_y = robin[0][1]
            ha='left'
            va='bottom'
        elif sub_pos=='lm':
            rob_x = robin[1][0]
            rob_y = robin[1][1]
            ha='left'
            va='center'
        elif sub_pos=='lt':
            rob_x = robin[2][0]
            rob_y = robin[2][1]
            ha='left'
            va='top'
        elif sub_pos=='mb':
            rob_x = robin[3][0]
            rob_y = robin[3][1]
            ha='center'
            va='bottom'
        elif sub_pos=='mt':
            rob_x = robin[4][0]
            rob_y = robin[4][1]
            ha='center'
            va='top'
        elif sub_pos=='rb':
            rob_x = robin[5][0]
            rob_y = robin[5][1]
            ha='right'
            va='bottom'
        elif sub_pos=='rm':
            rob_x = robin[6][0]
            rob_y = robin[6][1]
            ha='right'
            va='center'
        elif sub_pos=='rt':
            rob_x = robin[7][0]
            rob_y = robin[7][1]
            ha='right'
            va='top'

        if sub_pos[0] == 'r':
            rob_x = rob_x - 50000
        elif sub_pos[0] == 'l':
            rob_x = rob_x + 50000

        if sub_pos[1] == 'b':
            rob_y = rob_y + 50000
        elif sub_pos[1] == 't':
            rob_y = rob_y - 50000

        sub_ax_3 = fig.add_axes(position,
                                projection=ccrs.Robinson(), label=label+'markup2')

        # sub_ax_3.add_patch(mpatches.Rectangle((rob_x, rob_y), size_x, size_y , linewidth=1, edgecolor='grey', facecolor='white',transform=ccrs.Robinson()))

        sub_ax_3.set_extent(extent, ccrs.Geodetic())
        verts = mpath.Path(rect_units_to_verts([rob_x,rob_y,size_x,size_y]))
        sub_ax_3.set_boundary(verts, transform=sub_ax.transAxes)

        sub_ax_3.text(rob_x,rob_y,markup_sub,
                 horizontalalignment=ha, verticalalignment=va,
                 transform=ccrs.Robinson(), color='black',fontsize=10,bbox=dict(facecolor='white', alpha=1),fontweight='bold',zorder=25)

    if not main:
        sub_ax.outline_patch.set_edgecolor('white')
    else:
        sub_ax.outline_patch.set_edgecolor('lightgrey')


#TODO: careful here! figure size determines everything else, found no way to do it otherwise in cartopy
fig_width_inch=19.
fig = plt.figure(figsize=(fig_width_inch,fig_width_inch/1.9716))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

ax.set_global()
ax.outline_patch.set_linewidth(0)

#FIGURE

img = GeoImg(fn_hs)
land_mask = create_mask_from_shapefile(img,fn_land)
ds = gdal.Open(fn_hs)
hs = ds.ReadAsArray()
hs = hs.astype(float)

def stretch_hs(hs,stretch_factor=1.):

    max_hs = 255
    min_hs = 0

    hs_s = (hs - (max_hs-min_hs)/2)*stretch_factor + (max_hs-min_hs)/2

    return hs_s

hs = stretch_hs(hs,stretch_factor=0.9)

hs_land = hs.copy()
hs_land[~land_mask]=0
hs_notland = hs.copy()
hs_notland[land_mask]=0

gt = ds.GetGeoTransform()  # Defining bounds
ext = (gt[0], gt[0] + ds.RasterXSize * gt[1],
           gt[3] + ds.RasterYSize * gt[5], gt[3])
# hs[:, :] = -hs[:, :] * (hs > -1e20)

color1 = mpl.colors.to_rgba('black')
color2 = mpl.colors.to_rgba('white')
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2', [color1, color2], 256)
cmap2._init()
cmap2._lut[0:1, -1] = 0.0  # We made transparent de 10 first levels of hillshade,
cmap2._lut[1:, -1] = 0.35

cmap22 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap22', [color1, color2], 256)
cmap22._init()
cmap22._lut[0:1, -1] = 0.0  # We made transparent de 10 first levels of hillshade,
cmap22._lut[1:, -1] = 0.10

add_inset(fig,[-179.99,179.99,-89.99,89.99],[0.375, 0.21, 0.25, 0.25],bounds=[-179.99,179.99,-89.99,89.99],shades=False, hillshade = False, main=True, list_shp=shp_buff)

poly_aw = np.array([(-158,-79),(-135,-62),(-110,-62),(-50,-62),(-50,-79.25),(-158,-79.25)])
add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.4,-0.065,2,2],bounds=[-158, -50, -62.5, -79],label='Antarctica_West',polygon=poly_aw,shades=True)

poly_ae = np.array([(135,-81.5),(152,-63.7),(165,-65),(175,-70),(175,-81.25),(135,-81.75)])
add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.71,-0.045,2,2],bounds=[130, 175, -64.5, -81],label='Antarctica_East',polygon=poly_ae,shades=True)

poly_ac = np.array([(-25,-62),(106,-62),(80,-79.25),(-25,-79.25),(-25,-62)])
add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.52,-0.065,2,2],bounds=[-25, 106, -62.5, -79],label='Antarctica_Center',polygon=poly_ac,shades=True,markup='Antarctic and Subantarctic (19)',markpos='right',markadj=0)

add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.68,-0.18,2,2],bounds=[64, 78, -48, -55],label='Antarctica_Australes',shades=True,markup_sub='b',sub_pos='lt')

add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.42,-0.155,2,2],bounds=[-40, -23, -53, -60],label='Antarctica_South_Georgia',shades=True,markup_sub='a',sub_pos='rt')

add_inset(fig, [-179.99,179.99,-89.99,89.99], [-0.52, -0.225, 2, 2],bounds=[-82,-65,13,-57],label='Andes',markup='Low Latitudes (16) &\nSouthern Andes (17)',markadj=0)
add_inset(fig, [-179.99,179.99,-89.99,89.99], [-0.352, -0.39, 2, 2],bounds=[-100,-95,22,16],label='Mexico',markup_sub='a',sub_pos='rb')
add_inset(fig, [-179.99,179.99,-89.99,89.99], [-1.078, -0.24, 2, 2],bounds=[28,42,2,-5],label='Africa',markup_sub='b',sub_pos='rb')
add_inset(fig, [-179.99,179.99,-89.99,89.99], [-1.640, -0.3, 2, 2],bounds=[133,140,-2,-7],label='Indonesia',markup_sub='c',sub_pos='rb')

poly_arctic = np.array([(-105,84.5),(115,84.5),(110,68),(30,68),(18,57),(-70,57),(-100,75),(-105,84.5)])
add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.48,-1.003,2,2],bounds=[-100, 106, 57, 84],label='Arctic West',polygon=poly_arctic,markup='Arctic (03-09)',markadj=0)

add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.92,-0.17,2,2],bounds=[164,176,-47,-40],label='New Zealand',markup='New Zealand (18)',markpos='right',markadj=0)

poly_na = np.array([(-170,72),(-140,72),(-120,63),(-101,35),(-126,35),(-165,55),(-170,72)])
add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.1,-1.22,2,2],bounds=[-177,-105, 36, 70],label='North America',polygon=poly_na,markup='Alaska (01) & Western\nCanada and USA (02)',markadj=0)

poly_asia = np.array([(148,49),(160,65),(178,65),(170,55),(160,49),(148,49)])
poly_asia_ne = np.array([(142,71),(142,80),(163,80),(155,71),(142,71)])
add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.655,-1.165,2,2],bounds=[142,160,71,80],polygon=poly_asia_ne,label='North Asia North E',markup_sub='d',sub_pos='rt')
add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.565,-1.107,2,2],bounds=[87,112,68,77],label='North Asia North W',markup_sub='c',sub_pos='mt')
poly_asia_e2 = np.array([(125,58),(125,72),(153.8,72),(148,58),(125,58)])

only_shade([-0.71,-1.142,2,2],[125,148,58,72],polygon=poly_asia_e2,label='tmp_NAE2')
add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.822,-1.218,2,2],bounds=[128,179.9,50,64.8],label='North Asia East',polygon=poly_asia,markup_sub='f',sub_pos='lb')
add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.71,-1.142,2,2],bounds=[125,148,58,72],polygon=poly_asia_e2,label='North Asia East 2',markup_sub='e',sub_pos='lb',shades=False)

only_shade([-0.517,-1.035,2,2],[53,70,62,69.8],label='tmp_NAW')
add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.74,-1,2,2],bounds=[82,120,45.5,58.9],label='South Asia North',markup='North Asia (10)',markup_sub='a',sub_pos='mb',markadj=0)
add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.512,-1.035,2,2],bounds=[53,70,62,69.8],label='North Asia West',markup_sub='b',sub_pos='lb',shades=False)

add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.685,-1.065,2,2],bounds=[65, 105, 46.5, 25],label='HMA',markup='High Mountain Asia (13-15)',markadj=0)

add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.58,-0.982,2,2],bounds=[-4.9,19,38.2,50.5],label='Europe',markup='Central Europe (11)',markadj=0)

add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.66,-0.89,2,2],bounds=[38,54,29,44.75],label='Middle East',markup='Caucasus (12)',markadj=0)

plt.savefig(out_png,dpi=400)
# plt.show()



#LEGEND
#
# fig_width_inch=19.
# fig = plt.figure(figsize=(fig_width_inch,fig_width_inch/1.9716))
#
#
# axleg4 = fig.add_axes([0,-0.37,1,1],projection=ccrs.Robinson(),label='legend4')
# axleg4.set_extent([-179.99,179.99,-89.99,89.99], ccrs.Geodetic())
# axleg4.outline_patch.set_linewidth(0)
# bounds = [-10,10,-10,10]
# polygon = poly_from_extent(bounds)
# verts = mpath.Path(latlon_extent_to_robinson_axes_verts(polygon))
# axleg4.set_boundary(verts, transform=axleg4.transAxes)
#
# axleg4.text(13500000, 200000, 'North\nAsia (10)', fontsize=12,fontweight='bold',ha='center')
# axleg4.text(11000000,0,'a.',horizontalalignment='left',verticalalignment='top',fontsize=12,fontweight='bold')
# axleg4.text(11400000,0,'Altay and\nSayan',horizontalalignment='left',verticalalignment='top',fontsize=12)
# axleg4.text(11000000,-600000,'b.',horizontalalignment='left',verticalalignment='top',fontsize=12,fontweight='bold')
# axleg4.text(11400000,-600000,'Ural',horizontalalignment='left',verticalalignment='top',fontsize=12)
# axleg4.text(11000000,-900000,'c.',horizontalalignment='left',verticalalignment='top',fontsize=12,fontweight='bold')
# axleg4.text(11400000,-900000,'North Siberia',horizontalalignment='left',verticalalignment='top',fontsize=12)
# axleg4.text(14000000,0,'d.',horizontalalignment='left',verticalalignment='top',fontsize=12,fontweight='bold')
# axleg4.text(14400000,0,'Bulunsky',horizontalalignment='left',verticalalignment='top',fontsize=12)
# axleg4.text(14000000,-300000,'e.',horizontalalignment='left',verticalalignment='top',fontsize=12,fontweight='bold')
# axleg4.text(14400000,-300000,'Cherskiy and\nSuntar Khayata',horizontalalignment='left',verticalalignment='top',fontsize=12)
# axleg4.text(14000000,-900000,'f.',horizontalalignment='left',verticalalignment='top',fontsize=12,fontweight='bold')
# axleg4.text(14400000,-900000,'Kamchatka Krai',horizontalalignment='left',verticalalignment='top',fontsize=12)
#
# axleg4.text(-15000000, 200000, 'Low\nLatitudes (16)', fontsize=12,fontweight='bold',ha='center')
# axleg4.text(-16500000,0,'a.',horizontalalignment='left',verticalalignment='top',fontsize=12,fontweight='bold')
# axleg4.text(-16100000,0,'Tropical Andes',horizontalalignment='left',verticalalignment='top',fontsize=12)
# axleg4.text(-16500000,-300000,'b.',horizontalalignment='left',verticalalignment='top',fontsize=12,fontweight='bold')
# axleg4.text(-16100000,-300000,'Mexico',horizontalalignment='left',verticalalignment='top',fontsize=12)
# axleg4.text(-16500000,-600000,'c.',horizontalalignment='left',verticalalignment='top',fontsize=12,fontweight='bold')
# axleg4.text(-16100000,-600000,'East Africa',horizontalalignment='left',verticalalignment='top',fontsize=12)
# axleg4.text(-16500000,-900000,'d.',horizontalalignment='left',verticalalignment='top',fontsize=12,fontweight='bold')
# axleg4.text(-16100000,-900000,'New Guinea',horizontalalignment='left',verticalalignment='top',fontsize=12)
#
# axleg4.text(-11200000, 200000, 'Antarctic and\nSubantarctic (19)', fontsize=11,fontweight='bold',ha='center')
# axleg4.text(-13000000,0,'a.',horizontalalignment='left',verticalalignment='top',fontsize=12,fontweight='bold')
# axleg4.text(-12600000,0,'West and Peninsula',horizontalalignment='left',verticalalignment='top',fontsize=12)
# axleg4.text(-13000000,-300000,'b.',horizontalalignment='left',verticalalignment='top',fontsize=12,fontweight='bold')
# axleg4.text(-12600000,-300000,'South Georgia\nand Central Islands',horizontalalignment='left',verticalalignment='top',fontsize=12)
# axleg4.text(-13000000,-900000,'c,e.',horizontalalignment='left',verticalalignment='top',fontsize=12,fontweight='bold')
# axleg4.text(-12300000,-900000,'East',horizontalalignment='left',verticalalignment='top',fontsize=12)
# axleg4.text(-13000000,-1200000,'d.',horizontalalignment='left',verticalalignment='top',fontsize=12,fontweight='bold')
# axleg4.text(-12600000,-1200000,'Kerguelen and\nHeard Islands',horizontalalignment='left',verticalalignment='top',fontsize=12)
#
#
# axleg2 = fig.add_axes([0,0,1,1],projection=ccrs.Robinson(),label='legend2')
# axleg2.outline_patch.set_linewidth(0)
#
# col_bounds = np.array([-1.4,-1,-0.8,-0.6,-0.4,-0.2,0,0.1,0.2,0.3,0.4,0.5,0.6])
# col_bounds = np.array(col_bounds)
# # cmap = plt.get_cmap('RdYlBu')
# cb = []
# cb_val = np.linspace(0, 1, len(col_bounds))
# for j in range(len(cb_val)):
#     cb.append(mpl.cm.RdYlBu(cb_val[j]))
# cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(zip((col_bounds-min(col_bounds))/(max(col_bounds-min(col_bounds))),cb)), N=1000, gamma=1.0)
# # norm = mpl.colors.BoundaryNorm(boundaries=col_bounds, ncolors=256)
# norm = mpl.colors.Normalize(vmin=-1.4,vmax=0.6)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cb = plt.colorbar(sm ,ax=axleg2, ticks=[-1.4,-1.2,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6], orientation='horizontal',extend='both',shrink=0.35)
# cb.ax.tick_params(labelsize=12)
# cb.set_label('Mean elevation change rate (m yr$^{-1}$)',fontsize=12)
#
# axleg = fig.add_axes([-0.248,-0.86,2,2],projection=ccrs.Robinson(),label='legend')
# axleg.set_extent([-179.99,179.99,-89.99,89.99], ccrs.Geodetic())
# axleg.outline_patch.set_linewidth(0)
#
# u=0
# rad_tot = 0
# for a in [100, 1000, 10000]:
#     rad = (12000+np.sqrt(a)*1000)
#     axleg.add_patch(mpatches.Circle(xy=[-700000+rad_tot+u*600000,0],radius=rad,edgecolor='k',label=str(a)+' km$^2$', transform = ccrs.Robinson(),fill=False, zorder=30))
#     u=u+1
#     rad_tot += rad
# axleg.text(0, -2.5, '100     1000     10000', transform=ccrs.Geodetic(),horizontalalignment='center',verticalalignment='center',fontsize=12)
# axleg.text(0, -5, 'Glacierized area (km$^2$)', transform=ccrs.Geodetic(),horizontalalignment='center',verticalalignment='center',fontsize=12)
#
# bounds = [-10,10,-10,10]
# polygon = poly_from_extent(bounds)
# verts = mpath.Path(latlon_extent_to_robinson_axes_verts(polygon))
# axleg.set_boundary(verts, transform=axleg.transAxes)
# #
# # axleg3 = fig.add_axes([0.30,-0.37,1,1],projection=ccrs.Robinson(),label='legend3')
# # axleg3.set_extent([-179.99,179.99,-89.99,89.99], ccrs.Geodetic())
# # axleg3.outline_patch.set_linewidth(0)
# #
# # def add_latlon_grid(ax,center_x,center_y,spacing=100000,edge=100000,size='11'):
# #
# #
# #     ax.add_patch(mpatches.Arrow(center_x-spacing-edge, center_y, 2*spacing+2*edge, 0, edgecolor='black',linewidth=0.5,
# #                        transform=ccrs.Robinson(),zorder=30))
# #     ax.add_patch(mpatches.Arrow(center_x-spacing-edge, center_y+spacing, 2*spacing+2*edge, 0, edgecolor='black',linewidth=0.5,
# #                        transform=ccrs.Robinson(),zorder=30))
# #     ax.add_patch(mpatches.Arrow(center_x-spacing-edge, center_y-spacing, 2*spacing+2*edge, 0, edgecolor='black',linewidth=0.5,
# #                        transform=ccrs.Robinson(),zorder=30))
# #     ax.add_patch(mpatches.Arrow(center_x, center_y-spacing-edge, 0, 2*spacing+2*edge, edgecolor='black',linewidth=0.5,
# #                        transform=ccrs.Robinson(),zorder=30))
# #     ax.add_patch(mpatches.Arrow(center_x-spacing, center_y-spacing-edge, 0, 2*spacing+2*edge, edgecolor='black',linewidth=0.5,
# #                        transform=ccrs.Robinson(),zorder=30))
# #     ax.add_patch(mpatches.Arrow(center_x+spacing, center_y-spacing-edge, 0, 2*spacing+2*edge, edgecolor='black',linewidth=0.5,
# #                        transform=ccrs.Robinson(),zorder=30))
# #
# #     if size=='1°x1°':
# #         ax.add_patch(mpatches.Rectangle((center_x -spacing, center_y-spacing), spacing, spacing, edgecolor='black',
# #                                     linewidth=2,
# #                                     transform=ccrs.Robinson(),facecolor='None',alpha=1))
# #     elif size=='2°x1°':
# #         ax.add_patch(mpatches.Rectangle((center_x - spacing, center_y - spacing), 2*spacing, spacing, edgecolor='black',
# #                                         linewidth=2,
# #                                         transform=ccrs.Robinson(),facecolor='None',alpha=1))
# #     elif size=='2°x2°':
# #         ax.add_patch(mpatches.Rectangle((center_x - spacing, center_y - spacing), 2*spacing, 2*spacing, edgecolor='black',
# #                                     linewidth=2,
# #                                     transform=ccrs.Robinson(), facecolor='None',alpha=1))
# #
# #     ax.text(center_x, center_y+500000, size, transform=ccrs.Robinson(), horizontalalignment='center',
# #                 verticalalignment='center', fontsize=10, weight='bold')
# #
# #
# # rad=200000
# # # axleg3.text(-2000000,0,'1 x',transform=ccrs.Robinson(),horizontalalignment='center',verticalalignment='center',fontsize=12)
# # axleg3.add_patch(mpatches.Circle(xy=(-1600000, 0), radius=rad, edgecolor='black',transform=ccrs.Robinson(), fill=False))
# # axleg3.text(-1000000,0,'$\in$',transform=ccrs.Robinson(),horizontalalignment='center',verticalalignment='center',fontsize=12)
# #
# # add_latlon_grid(axleg3,center_x=0,center_y=0,spacing=250000,size='1°x1°')
# #
# # add_latlon_grid(axleg3,center_x=1300000,center_y=0,spacing=250000,size='2°x1°')
# #
# # add_latlon_grid(axleg3,center_x=2600000,center_y=0,spacing=250000,size='2°x2°')
# #
# # axleg3.text(12,-6,'             0   $\\leq$   60°   $\\leq$   74°  $\\leq$   85° N-S  ',transform=ccrs.Geodetic(),horizontalalignment='center',verticalalignment='center',fontsize=12)
# # axleg3.text(12,-10,'Tile aggregation',transform=ccrs.Geodetic(),horizontalalignment='center',verticalalignment='center',fontsize=12)
# # axleg3.text(44,0,'$\\approx$ 10,000 km$^{2}$',transform=ccrs.Geodetic(),horizontalalignment='center',verticalalignment='center',fontsize=12)
# #
# # bounds = [-10,10,-10,10]
# # polygon = poly_from_extent(bounds)
# # verts = mpath.Path(latlon_extent_to_robinson_axes_verts(polygon))
# # axleg3.set_boundary(verts, transform=axleg3.transAxes)
#
#
# axleg5 = fig.add_axes([-0.23,-0.365,1,1],projection=ccrs.Robinson(),label='legend5')
# axleg5.set_extent([-179.99,179.99,-89.99,89.99], ccrs.Geodetic())
# axleg5.outline_patch.set_linewidth(0)
# bounds = [-10,10,-10,10]
# polygon = poly_from_extent(bounds)
# verts = mpath.Path(latlon_extent_to_robinson_axes_verts(polygon))
# axleg5.set_boundary(verts, transform=axleg5.transAxes)
#
# axleg5.add_patch(mpatches.Rectangle((-250000,-250000), 500000, 500000, edgecolor='black',
#                                 linewidth=1,
#                                 transform=ccrs.Robinson(), facecolor='indigo', alpha=1))
#
# axleg5.text(0,-1000000,'Glacier\ncontours\n(minimap)',horizontalalignment='center',verticalalignment='center',fontsize=12)
#
# plt.savefig('/home/atom/ongoing/work_worldwide/figures/Figure_2_legend.png',dpi=400)
