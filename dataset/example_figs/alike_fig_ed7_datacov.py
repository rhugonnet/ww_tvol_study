
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
from pyddem.vector_tools import SRTMGL1_naming_to_latlon, latlon_to_SRTMGL1_naming
import pyddem.vector_tools as ot
from pybob.image_tools import create_mask_from_shapefile
from pybob.GeoImg import GeoImg
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from glob import glob
mpl.use('Agg')
plt.rcParams.update({'font.size': 5})
plt.rcParams.update({'lines.linewidth':0.35})
plt.rcParams.update({'axes.linewidth':0.35})
plt.rcParams.update({'lines.markersize':2.5})
plt.rcParams.update({'axes.labelpad':1.5})

# shp_buff = '/home/atom/data/inventory_products/RGI/00_rgi60/rgi60_buff_diss.shp'
shp_buff = '/data/icesat/travail_en_cours/romain/figures/rgi60_buff_diss.shp'

# in_csv='/home/atom/ongoing/work_worldwide/vol/final/dh_world_tiles_1deg.csv'
in_csv = '/data/icesat/travail_en_cours/romain/results/vol_final/dh_world_tiles_1deg.csv'

fig_width_inch = 3.1
fig = plt.figure(figsize=(2*fig_width_inch, 2.35 * fig_width_inch / 1.9716))

print('Plotting: '+in_csv)

df=pd.read_csv(in_csv)
df=df[df.category=='all']

# out_png = '/home/atom/ongoing/work_worldwide/figures/revised/ED_Figure_6.png'
# out_png = '/data/icesat/travail_en_cours/romain/figures/ED_Figure_6.png'

df_tmp = df[df.period=='2000-01-01_2005-01-01']
tiles = [latlon_to_SRTMGL1_naming(df_tmp.tile_latmin.values[i],df_tmp.tile_lonmin.values[i]) for i in range(len(df_tmp))]
areas = df_tmp.area.tolist()
# areas = [area/1000000 for  area in areas]

#FOR EXAMPLE: CHANGING HERE TO VALID OBS + COLORBAR
dhs_1 = df[df.period=='2000-01-01_2005-01-01'].valid_obs.tolist()
dhs_2 = df[df.period=='2005-01-01_2010-01-01'].valid_obs.tolist()
dhs_3 = df[df.period=='2010-01-01_2015-01-01'].valid_obs.tolist()
dhs_4 = df[df.period=='2015-01-01_2020-01-01'].valid_obs.tolist()

errs_1 = df[df.period=='2000-01-01_2005-01-01'].err_dhdt.tolist()
errs_2 = df[df.period=='2005-01-01_2010-01-01'].err_dhdt.tolist()
errs_3 = df[df.period=='2010-01-01_2015-01-01'].err_dhdt.tolist()
errs_4 = df[df.period=='2015-01-01_2020-01-01'].err_dhdt.tolist()

areas = [area/1000000 for _, area in sorted(zip(tiles,areas))]
dhs_1 = [dh for _, dh in sorted(zip(tiles,dhs_1))]
dhs_2 = [dh for _, dh in sorted(zip(tiles,dhs_2))]
dhs_3 = [dh for _, dh in sorted(zip(tiles,dhs_3))]
dhs_4 = [dh for _, dh in sorted(zip(tiles,dhs_4))]

errs_1 = [err for _, err in sorted(zip(tiles,errs_1))]
errs_2 = [err for _, err in sorted(zip(tiles,errs_2))]
errs_3 = [err for _, err in sorted(zip(tiles,errs_3))]
errs_4 = [err for _, err in sorted(zip(tiles,errs_4))]

tiles = sorted(tiles)

group_by_spec = False
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
        center_lon = lon + 1
        center_lat = lat + 1
        size = (2,2)

    elif np.abs(lat) >= 60:
        center_lon = lon + 1
        center_lat = lat + 0.5
        size=(2,1)

    else:
        center_lon = lon + 0.5
        center_lat = lat + 0.5
        size=(1,1)


    return center_lat, center_lon, size

def coordXform(orig_crs, target_crs, x, y):
    return target_crs.transform_points( orig_crs, x, y )

def poly_from_extent(ext):

    poly = np.array([(ext[0],ext[2]),(ext[1],ext[2]),(ext[1],ext[3]),(ext[0],ext[3]),(ext[0],ext[2])])

    return poly

def latlon_extent_to_robinson_axes_verts(polygon_coords):

    robin = np.transpose(np.array(list(zip(*polygon_coords)),dtype=float))


    limits_robin = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([-179.99,179.99,0,0]),np.array([0,0,-89.99,89.99]))

    ext_robin_x = limits_robin[1][0] - limits_robin[0][0]
    ext_robin_y = limits_robin[3][1] - limits_robin[2][1]

    verts = robin.copy()
    verts[:,0] = (verts[:,0] + limits_robin[1][0])/ext_robin_x
    verts[:,1] = (verts[:,1] + limits_robin[3][1])/ext_robin_y

    return verts[:,0:2]

def add_inset(fig,extent,pos,bounds,label=None,polygon=None,anom=None,draw_cmap_y=None,hillshade=True,markup_sub=None,sub_pos=None,sub_adj=None):


    sub_ax = fig.add_axes(pos,
                          projection=ccrs.Robinson(), label=label)
    sub_ax.set_extent(extent, ccrs.Geodetic())

    sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor='gainsboro'))
    sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor='dimgrey'))
    # if anom is not None:
    #     shape_feature = ShapelyFeature(Reader(shp_buff).geometries(), ccrs.PlateCarree(), edgecolor='black', alpha=0.5,
    #                                    facecolor='black', linewidth=1)
    #     sub_ax.add_feature(shape_feature)

    if polygon is None and bounds is not None:
        polygon = poly_from_extent(bounds)

    if bounds is not None:
        verts = mpath.Path(latlon_extent_to_robinson_axes_verts(polygon))
        sub_ax.set_boundary(verts, transform=sub_ax.transAxes)

    if hillshade:
        def out_of_poly_mask(geoimg, poly_coords):

            poly = ot.poly_from_coords(inter_poly_coords(poly_coords))
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(54030)

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

            return np.array(list(zip(all_lon_interp, all_lat_interp)))

        img = GeoImg(fn_hs)
        hs_tmp = hs_land.copy()
        hs_tmp_nl = hs_notland.copy()
        mask = out_of_poly_mask(img, polygon)

        hs_tmp[~mask] = 0
        hs_tmp_nl[~mask] = 0

        sub_ax.imshow(hs_tmp[:, :], extent=ext, transform=ccrs.Robinson(), cmap=cmap2, zorder=2,interpolation='nearest')
        sub_ax.imshow(hs_tmp_nl[:, :], extent=ext, transform=ccrs.Robinson(), cmap=cmap22, zorder=2, interpolation='nearest')

    sub_ax.outline_patch.set_edgecolor('white')

    if anom is not None:

        if anom == 'dhs_1' or 'dhs_2' or 'dhs_3' or 'dhs_3':

            col_bounds = np.array([0,1,2,3,5,7,10,15,20])
            cb = []
            cb_val = np.linspace(0, 1, len(col_bounds))
            for j in range(len(cb_val)):
                cb.append(mpl.cm.viridis(cb_val[j]))
            cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(
                zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), cb)), N=1000)

            if anom == 'dhs_1':
                vals = dhs_1
            elif anom == 'dhs_2':
                vals = dhs_2
            elif anom == 'dhs_3':
                vals = dhs_3
            elif anom == 'dhs_4':
                vals = dhs_4

            lab = 'Number of valid observations'

        # elif anom == 'dt':
        #     col_bounds = np.array([-0.3, -0.15, 0, 0.3, 0.6])
        #     cb = []
        #     cb_val = np.linspace(0, 1, len(col_bounds))
        #     for j in range(len(cb_val)):
        #         cb.append(mpl.cm.RdBu_r(cb_val[j]))
        #     cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(
        #         zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), cb)), N=1000)
        #
        #     vals = dts
        #     lab = 'Decadal difference in temperature (K)'
        #
        # elif anom == 'dp':
        #     col_bounds = np.array([-0.2, -0.1, 0, 0.1, 0.2])
        #     cb = []
        #     cb_val = np.linspace(0, 1, len(col_bounds))
        #     for j in range(len(cb_val)):
        #         cb.append(mpl.cm.BrBG(cb_val[j]))
        #     cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(
        #         zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), cb)), N=1000)
        #
        #     vals = dps
        #     lab = 'Decadal difference in precipitation (m)'

        # elif anom == 'du':
        #     col_bounds = np.array([-1, -0.5, 0, 0.5, 1])
        #     cb = []
        #     cb_val = np.linspace(0, 1, len(col_bounds))
        #     for j in range(len(cb_val)):
        #         cb.append(mpl.cm.RdBu_r(cb_val[j]))
        #     cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(
        #         zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), cb)), N=1000)
        #
        #     vals = dus
        #     lab = 'Wind speed anomaly (m s$^{-1}$)'
        #
        # elif anom == 'dz':
        #
        #     col_bounds = np.array([-100, -50, 0, 50, 100])
        #     cb = []
        #     cb_val = np.linspace(0, 1, len(col_bounds))
        #     for j in range(len(cb_val)):
        #         cb.append(mpl.cm.RdBu_r(cb_val[j]))
        #     cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(
        #         zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), cb)), N=1000)
        #
        #     vals = dzs
        #     lab = 'Geopotential height anomaly at 500 hPa (m)'
        #
        # elif anom =='dk':
        #
        #     col_bounds = np.array([-200000, -100000, 0, 100000, 200000])
        #     cb = []
        #     cb_val = np.linspace(0, 1, len(col_bounds))
        #     for j in range(len(cb_val)):
        #         cb.append(mpl.cm.RdBu_r(cb_val[j]))
        #     cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(
        #         zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), cb)), N=1000)
        #
        #     vals = dks
        #     lab = 'Net clear-sky downwelling SW surface radiation anomaly (J m$^{-2}$)'

        if draw_cmap_y is not None:

            sub_ax_2 = fig.add_axes([0.2, draw_cmap_y, 0.6, 0.05])
            sub_ax_2.set_xticks([])
            sub_ax_2.set_yticks([])
            sub_ax_2.spines['top'].set_visible(False)
            sub_ax_2.spines['left'].set_visible(False)
            sub_ax_2.spines['right'].set_visible(False)
            sub_ax_2.spines['bottom'].set_visible(False)

            cbaxes = sub_ax_2.inset_axes([0,0.85,1,0.2],label='legend_'+label)
            norm = mpl.colors.Normalize(vmin=min(col_bounds), vmax=max(col_bounds))
            sm = plt.cm.ScalarMappable(cmap=cmap_cus, norm=norm)
            sm.set_array([])
            cb = plt.colorbar(sm, cax=cbaxes, ticks=col_bounds, orientation='horizontal', extend='both', shrink=0.9)
            # cb.ax.tick_params(labelsize=12)
            cb.set_label(lab)

        for i in range(len(tiles)):
            lat, lon = SRTMGL1_naming_to_latlon(tiles[i])
            if group_by_spec:
                lat, lon, s = latlon_to_spec_center(lat, lon)
            else:
                lat = lat + 0.5
                lon = lon + 0.5
                s = (1,1)
            # fac = 0.02
            fac = 7000000.

            if anom == 'dhs_1':
                errs = errs_1
            elif anom == 'dhs_2':
                errs = errs_2
            elif anom == 'dhs_3':
                errs = errs_3
            elif anom == 'dhs_4':
                errs = errs_4

            if np.isnan(errs[i]):
                continue

            #need to square because Rectangle already shows a surface
            f=np.sqrt(((1/min(max(errs[i],0.25),1)**2-1/1**2)/(1/0.25**2-1/1**2)))*(1-np.sqrt(0.1))

            if ~np.isnan(vals[i]) and areas[i] > 0.2:
                val = vals[i]
                val_col = max(0.0001, min(0.9999, (val - min(col_bounds)) / (max(col_bounds) - min(col_bounds))))
                col = cmap_cus(val_col)
            elif areas[i] <= 5:
                continue
            else:
                col = plt.cm.Greys(0.7)

            # xy = [lon,lat]
            xy = coordXform(ccrs.PlateCarree(), ccrs.Robinson(), np.array([lon]), np.array([lat]))[0][0:2]
            # sub_ax.add_patch(
            #     mpatches.Circle(xy=xy, radius=rad, color=col, alpha=1, transform=ccrs.Robinson(), zorder=30))
            xl = np.sqrt(0.1)*s[0] + f*s[0]
            yl = np.sqrt(0.1)*s[1] + f*s[1]
            sub_ax.add_patch(mpatches.Rectangle((lon-xl/2,lat-yl/2),xl,yl,facecolor=col,alpha=1,transform=ccrs.PlateCarree(),zorder=30))

    if markup_sub is not None and anom == 'dhs_1':

        lon_min = np.min(list(zip(*polygon))[0])
        lon_max = np.max(list(zip(*polygon))[0])
        lon_mid = 0.5*(lon_min+lon_max)

        lat_min = np.min(list(zip(*polygon))[1])
        lat_max = np.max(list(zip(*polygon))[1])
        lat_mid = 0.5*(lat_min+lat_max)

        robin = np.array(list(zip([lon_min,lon_min,lon_min,lon_mid,lon_mid,lon_max,lon_max,lon_max],[lat_min,lat_mid,lat_max,lat_min,lat_max,lat_min,lat_mid,lat_max])))

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
            rob_x = rob_x - 100000
        elif sub_pos[0] == 'l':
            rob_x = rob_x + 100000

        if sub_pos[1] == 'b':
            rob_y = rob_y + 100000
        elif sub_pos[1] == 't':
            rob_y = rob_y - 100000

        if sub_adj is not None:
            rob_x += sub_adj[0]
            rob_y += sub_adj[1]

        sub_ax.text(rob_x,rob_y,markup_sub,
                 horizontalalignment=ha, verticalalignment=va,
                 transform=ccrs.Robinson(), color='black',fontsize=4.5,bbox=dict(facecolor='white', alpha=1,linewidth=0.35,pad=1.5),fontweight='bold',zorder=25)



def add_compartment_world_map(fig,pos,anom=None,ymarg=None,label=None,label2=None):

    yr = pos[3]
    yb = pos[1]

    xr = pos[2]
    xb = pos[0]

    #Antarctic Peninsula
    bounds_ap = [-5500000,-3400000,-8000000,-5900000]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-0.735*xr,yb -0.06*yr, 2.7*xr, 2.7*yr],bounds=bounds_ap,label='AP',anom=anom,markup_sub='19a',sub_pos='lt',sub_adj=(30000,-170000))

    #Antarctic West
    bounds_aw=[-9500000,-5600000,-7930000,-7320000]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-0.4205*xr, yb-0.1513*yr, 2.7*xr, 2.7*yr],bounds=bounds_aw,label='AW0',anom=anom,markup_sub='19a',sub_pos='lt',sub_adj=(70000,-15000))

    #Antarctic East
    bounds_ae1 = [-1960000,2250000,-7700000,-7080000]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-0.7087*xr, yb-0.1872*yr, 2.7*xr, 2.7*yr],bounds=bounds_ae1,label='AW',anom=anom,markup_sub='19c',sub_pos='lt',sub_adj=(0,-25000))

    #Antartic East 2
    bounds_ae2 = [2450000,7500000,-7570000,-6720000]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-1.0585*xr, yb-0.113*yr, 2.7*xr, 2.7*yr],bounds=bounds_ae2,label='AW2',anom=anom,markup_sub='19e',sub_pos='rt',sub_adj=(-1075000,-45000))

    #Antartic East 3
    bounds_ae3=[9430000,11900000,-8200000,-6770000]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-1.2960*xr, yb-0.109*yr, 2.7*xr, 2.7*yr],bounds=bounds_ae3,label='AW3',anom=anom,markup_sub='19e',sub_pos='lm',sub_adj=(15000,0))

    #South America
    bounds_sa = [-7340000,-5100000,-5900000,0]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-0.765*xr, yb-0.4694*yr, 2.7*xr, 2.7*yr],bounds=bounds_sa,label='South America',anom=anom,markup_sub='16-17',sub_pos='lm',sub_adj=(20000,-2000000))

    #Europe
    bounds_eu = [0,1500000,4500000,5400000]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-0.865*xr, yb-1.645*yr, 2.7*xr, 2.7*yr],bounds=bounds_eu,label='EU',anom=anom,markup_sub='11',sub_pos='lt',sub_adj=(20000,-135000))

    #Caucasus
    bounds_cau = [3200000,4800000,3300000,4800000]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-1.122*xr,yb -1.685*yr, 2.7*xr, 2.7*yr],bounds=bounds_cau,label='Cau',anom=anom,markup_sub='12',sub_pos='lb',sub_adj=(60000,20000))

    #New Zealand
    bounds_nz=[13750000,15225000,-5400000,-3800000]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-1.56*xr, yb-0.3235*yr, 2.7*xr, 2.7*yr],bounds=bounds_nz,label='NZ',anom=anom,markup_sub='18',sub_pos='lt',sub_adj=(135000,-100000))

    #Kamchatka Krai
    bounds_kam = [11500000,13200000,5100000,6700000]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-1.3991*xr, yb-1.73*yr, 2.7*xr, 2.7*yr],bounds=bounds_kam,label='Kam',anom=anom,markup_sub='10f',sub_pos='lb',sub_adj=(365000,15000))

    #HMA and North Asia 1
    bounds_hma = [5750000,9550000,2650000,5850000]
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-1.216*xr,yb -1.5835*yr, 2.7*xr, 2.7*yr],bounds=bounds_hma,label='HMA',anom=anom,markup_sub='13-15\n& 10a',sub_pos='rt',sub_adj=(-20000,-100000))

    #Arctic
    bounds_arctic = [-6060000,6420000,6100000,8400000]
    poly_arctic = np.array([(-6050000,7650000),(-5400000,6800000),(-4950000,6400000),(-3870000,5710000),(-2500000,5710000),(-2000000,5720000),(1350000,5720000),(2300000,6600000),(6500000,6600000),(6500000,8400000),(-6050000,8400000),(-6050000,7650000)])
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-0.8675*xr, yb-1.715*yr, 2.7*xr, 2.7*yr],bounds=bounds_arctic,polygon=poly_arctic,label='Arctic',anom=anom,markup_sub='03-09\n& 10b',sub_pos='mt',sub_adj=(-200000,-820000))

    #North America
    bounds_na = [-13600000,-9000000,3700000,7350000]
    poly_na = np.array([(-13600000,5600000),(-13600000,6000000),(-12900000,6000000),(-12900000,6800000),(-12500000,6800000),(-11500000,7420000),(-9000000,7420000),(-9000000,3750000),(-11000000,3750000),(-11000000,5600000),(-13600000,5600000)])
    add_inset(fig, [-179.99,179.99,-89.99,89.99], [xb-0.15*xr, yb-1.885*yr, 2.7*xr, 2.7*yr],bounds=bounds_na,polygon=poly_na,label='North America',anom=anom,markup_sub='01-02',sub_pos='rm',sub_adj=(-1970000,230000))

    out_axes = fig.add_axes([pos[0], pos[1], 0.2, 0])
    out_axes.set_xticks([])
    out_axes.set_yticks([])
    out_axes.spines['top'].set_visible(False)
    out_axes.spines['left'].set_visible(False)
    out_axes.spines['right'].set_visible(False)
    out_axes.spines['bottom'].set_visible(False)
    if label is not None:
        out_axes.text(0.05, 0.025, label, transform=out_axes.transAxes,fontsize=8, fontweight='bold', ha='left', va='bottom',zorder=30)

    out_axes_2 = fig.add_axes([pos[0],pos[1]+pos[3]-0.0225,pos[2],0])
    out_axes_2.set_xticks([])
    out_axes_2.set_yticks([])
    out_axes_2.spines['top'].set_visible(False)
    out_axes_2.spines['left'].set_visible(False)
    out_axes_2.spines['right'].set_visible(False)
    out_axes_2.spines['bottom'].set_visible(False)
    if label2 is not None:
        out_axes_2.text(0.5, 0, label2, transform=out_axes_2.transAxes, fontsize=6, fontweight='bold', ha='center', va='bottom',
              zorder=30)

fn_hs = '/data/icesat/travail_en_cours/romain/figures/world_robin_rs.tif'
fn_land = '/data/icesat/travail_en_cours/romain/figures/ne_50m_land.shp'
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

axleg5 = fig.add_axes([-0.32,-0.42,1,1],projection=ccrs.Robinson(),label='legend_5')
axleg5.set_extent([-179.99,179.99,-89.99,89.99], ccrs.Geodetic())
axleg5.outline_patch.set_linewidth(0)

axleg5.add_patch(mpatches.Rectangle((-250000+5000000,-250000), 500000, 500000, edgecolor='black',
                                linewidth=0.35,
                                transform=ccrs.Robinson(), facecolor='None', alpha=1))

axleg5.text(5000000,-600000,'< 0.5 m yr$^{-1}$',horizontalalignment='center',verticalalignment='center')
axleg5.text(5000000,-1200000,'100%',horizontalalignment='center',verticalalignment='center')

f1 = np.sqrt(((1/min(max(0.5,0.25),1)**2-1/1**2)/(1/0.25**2-1/1**2)))*(1-np.sqrt(0.1)) + np.sqrt(0.1)
axleg5.add_patch(mpatches.Rectangle((-f1*500000/2+2500000,-f1*500000/2), f1*500000, f1*500000, edgecolor='black',
                                linewidth=0.35,
                                transform=ccrs.Robinson(), facecolor='None', alpha=1))

axleg5.text(2500000,-600000,'= 1 m yr$^{-1}$',horizontalalignment='center',verticalalignment='center')
axleg5.text(2500000,-1200000,'39%',horizontalalignment='center',verticalalignment='center')

axleg5.add_patch(mpatches.Rectangle((-np.sqrt(0.1)*500000/2,-np.sqrt(0.1)*500000/2), np.sqrt(0.1)*500000, np.sqrt(0.1)*500000, edgecolor='black',
                                linewidth=0.35,
                                transform=ccrs.Robinson(), facecolor='None', alpha=1))

axleg5.text(0,-600000,'> 2 m yr$^{-1}$',horizontalalignment='center',verticalalignment='center')
axleg5.text(0,-1200000,'10%',horizontalalignment='center',verticalalignment='center')

axleg5.text(-1200000,-600000,'95% CI',horizontalalignment='right',verticalalignment='center')
axleg5.text(-1200000,-1200000,'Tile size',horizontalalignment='right',verticalalignment='center')


def rect_units_to_verts(rect_u):

    return np.array([[rect_u[0],rect_u[1]],[rect_u[0]+rect_u[2],rect_u[1]],[rect_u[0]+rect_u[2],rect_u[1] +rect_u[3]],[rect_u[0],rect_u[1]+rect_u[3]],[rect_u[0],rect_u[1]]])

verts = mpath.Path(rect_units_to_verts([-10000000,-10000000,20000000,20000000]))
axleg5.set_boundary(verts, transform=axleg5.transAxes)

list_anom = ['dhs_2','dhs_1','dhs_4','dhs_3']
labs=['b','a','d','c']
labs2 = ['2005-2009','2000-2004','2015-2019','2010-2014']

for i in range(len(list_anom)):

    a = i % 2
    b = np.floor(i/2)

    print('Adding compartiment for variable: '+list_anom[i])
    yb= 0.03
    yl = 1/2.35

    pos = [1-0.5*(a+1),1+yb-(yl+yb)*(b+1),0.5,yl]

    add_compartment_world_map(fig,pos,anom=list_anom[i],ymarg=yb,label=labs[i],label2=labs2[i])

# LEGEND

sub_ax_2 = fig.add_axes([0.4, 0.03, 0.4, 0.05])
sub_ax_2.set_xticks([])
sub_ax_2.set_yticks([])
sub_ax_2.spines['top'].set_visible(False)
sub_ax_2.spines['left'].set_visible(False)
sub_ax_2.spines['right'].set_visible(False)
sub_ax_2.spines['bottom'].set_visible(False)

lab = 'Number of valid observations'

col_bounds = np.array([0,1,2,3,5,7,10,15,20])
cb = []
cb_val = np.linspace(0, 1, len(col_bounds))
for j in range(len(cb_val)):
    cb.append(mpl.cm.viridis(cb_val[j]))
cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(
    zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), cb)), N=1000)

cbaxes = sub_ax_2.inset_axes([0, 0.85, 1, 0.4], label='legend_1')
norm = mpl.colors.Normalize(vmin=min(col_bounds), vmax=max(col_bounds))
sm = plt.cm.ScalarMappable(cmap=cmap_cus, norm=norm)
sm.set_array([])
cb = plt.colorbar(sm, cax=cbaxes, ticks=np.arange(0,21,2), orientation='horizontal', extend='max', shrink=0.9)
cb.ax.tick_params(width=0.35,length=2.5)
cb.set_label(lab,labelpad=1.5)



# ADD A MINIMAP
# yleft = 1-(pos[1]+pos[3])
#
# main_ax = fig.add_axes([0,pos[1]+pos[3]+yleft*0.15,1,yleft*0.85],projection=ccrs.Robinson(),label='main')
# main_ax.set_global()
# # main_ax.outline_patch.set_edgecolor('black')
#
# main_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor='gainsboro'))
# main_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor='dimgrey'))
#
# shape_feature = ShapelyFeature(Reader(shp_buff).geometries(), ccrs.PlateCarree(), edgecolor='indigo', alpha=1,
#                                facecolor='indigo', linewidth=1)
# main_ax.add_feature(shape_feature)
#
#
# bounds_ap = [-5500000, -3400000, -8000000, -5900000]
# poly_ap = poly_from_extent(bounds_ap)
# main_ax.add_patch(mpatches.Polygon(poly_ap,facecolor='None',edgecolor='white',zorder=30,lw=2))
#
# # Antarctic West
# bounds_aw = [-9500000, -5600000, -7900000, -7350000]
# poly_aw = poly_from_extent(bounds_aw)
# main_ax.add_patch(mpatches.Polygon(poly_aw,facecolor='None',edgecolor='white',zorder=30,lw=2))
#
# # Antarctic East
# bounds_ae1 = [-1960000, 2250000, -7700000, -7100000]
# poly_ae1 = poly_from_extent(bounds_ae1)
# main_ax.add_patch(mpatches.Polygon(poly_ae1,facecolor='None',edgecolor='white',zorder=30,lw=2))
# # Antartic East 2
# bounds_ae2 = [2450000, 7500000, -7600000, -6770000]
# poly_ae2= poly_from_extent(bounds_ae2)
# main_ax.add_patch(mpatches.Polygon(poly_ae2,facecolor='None',edgecolor='white',zorder=30,lw=2))
# # Antartic East 3
# bounds_ae3 = [9430000, 11700000, -8150000, -6800000]
# poly_ae3 = poly_from_extent(bounds_ae3)
# main_ax.add_patch(mpatches.Polygon(poly_ae3,facecolor='None',edgecolor='white',zorder=30,lw=2))
# # South America
# bounds_sa = [-7100000, -5100000, -5900000, 0]
# poly_sa = poly_from_extent(bounds_sa)
# main_ax.add_patch(mpatches.Polygon(poly_sa,facecolor='None',edgecolor='white',zorder=30,lw=2))
# # Europe
# bounds_eu = [0, 1500000, 4500000, 5400000]
# poly_eu = poly_from_extent(bounds_eu)
# main_ax.add_patch(mpatches.Polygon(poly_eu,facecolor='None',edgecolor='white',zorder=30,lw=2))
# # Caucasus
# bounds_cau = [3200000, 4800000, 3300000, 4800000]
# poly_cau = poly_from_extent(bounds_cau)
# main_ax.add_patch(mpatches.Polygon(poly_cau,facecolor='None',edgecolor='white',zorder=30,lw=2))
# # New Zealand
# bounds_nz = [13800000, 15200000, -5400000, -3800000]
# poly_nz = poly_from_extent(bounds_nz)
# main_ax.add_patch(mpatches.Polygon(poly_nz,facecolor='None',edgecolor='white',zorder=30,lw=2))
# # Kamchatka Krai
# bounds_kam = [11600000, 13150000, 5100000, 6700000]
# poly_kam = poly_from_extent(bounds_kam)
# main_ax.add_patch(mpatches.Polygon(poly_kam,facecolor='None',edgecolor='white',zorder=30,lw=2))
# # HMA and North Asia 1
# bounds_hma = [5750000, 9550000, 2650000, 5850000]
# poly_hma = poly_from_extent(bounds_hma)
# main_ax.add_patch(mpatches.Polygon(poly_hma,facecolor='None',edgecolor='white',zorder=30,lw=2))
# # Arctic
# poly_arctic = np.array(
#     [(-5950000, 7650000), (-5400000, 6800000), (-4950000, 6400000), (-3900000, 5720000), (-2500000, 5720000),
#      (-2000000, 5720000), (1350000, 5720000), (2300000, 6600000), (6390000, 6600000), (6390000, 8400000),
#      (-5950000, 8400000), (-5950000, 7650000)])
# main_ax.add_patch(mpatches.Polygon(poly_arctic,facecolor='None',edgecolor='white',zorder=30,lw=2))
#
#
# # North America
# poly_na = np.array(
#     [(-13600000, 5600000), (-13600000, 6000000), (-12900000, 6000000), (-12900000, 6800000), (-12500000, 6800000),
#      (-11500000, 7420000), (-9000000, 7420000), (-9000000, 3750000), (-11000000, 3750000), (-11000000, 5600000),
#      (-13600000, 5600000)])
# main_ax.add_patch(mpatches.Polygon(poly_na,facecolor='None',edgecolor='white',zorder=30,lw=2))
#

# plt.savefig('/home/atom/ongoing/work_worldwide/figures/github_examples/ED_Figure_7_datacov.png',dpi=400)
plt.savefig('/data/icesat/travail_en_cours/romain/results/figures/ED_Figure_7_datacov.png',dpi=400)

# plt.savefig(out_png,dpi=400)

print('Figure saved.')
