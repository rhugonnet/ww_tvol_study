
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gdal
import numpy as np
import pandas as pd
from matplotlib.dates import MonthLocator, YearLocator
import pyddem.vector_tools as ot
import osr
from pybob.GeoImg import GeoImg
from pybob.image_tools import create_mask_from_shapefile
plt.rcParams.update({'font.size': 5})
plt.rcParams.update({'lines.linewidth':0.35})
plt.rcParams.update({'axes.linewidth':0.35})

# fn_density = '/home/atom/ongoing/work_worldwide/coverage/ww_cov_aster_rob.tif'
fn_density = '/home/atom/ongoing/work_worldwide/coverage/ww_cov_arcticdem_raw_rob.tif'
fn_hs = '/home/atom/documents/paper/Hugonnet_2020/figures/world_robin_rs.tif'
fn_land = '/home/atom/data/inventory_products/NaturalEarth/ne_50m_land/ne_50m_land.shp'

bounds = [-179.99,179.99,-89.99,89.99]

def poly_from_extent(ext):

    poly = np.array([(ext[0],ext[2]),(ext[1],ext[2]),(ext[1],ext[3]),(ext[0],ext[3]),(ext[0],ext[2])])

    return poly
polygon = poly_from_extent(bounds)

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

def out_of_poly_mask(geoimg, poly_coords):

    poly = ot.poly_from_coords(inter_poly_coords(poly_coords))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    # put in a memory vector
    ds_shp = ot.create_mem_shp(poly, srs)

    return ot.geoimg_mask_on_feat_shp_ds(ds_shp, geoimg)


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
img = GeoImg(fn_hs)
hs_tmp_l = hs_land.copy()
hs_tmp_nl = hs_notland.copy()
mask = out_of_poly_mask(img, polygon)

hs_tmp_l[~mask] = 0
hs_tmp_nl[~mask] = 0

color1 = mpl.colors.to_rgba('black')
color2 = mpl.colors.to_rgba('white')
cmaphs2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmaphs2', [color1, color2], 256)
cmaphs2._init()
cmaphs2._lut[0:1, -1] = 0.0  # We made transparent de 10 first levels of hillshade,
cmaphs2._lut[1:, -1] = 0.30

cmaphs22 = mpl.colors.LinearSegmentedColormap.from_list('my_cmaphs22', [color1, color2], 256)
cmaphs22._init()
cmaphs22._lut[0:1, -1] = 0.0  # We made transparent de 10 first levels of hillshade,
cmaphs22._lut[1:, -1] = 0.15

fig = plt.figure(figsize=(7.2,4))
grid = plt.GridSpec(24, 11, wspace=0.4, hspace=0.3)

sub_ax = fig.add_subplot(grid[:3, :8],projection=ccrs.Robinson())
sub_ax.set_extent([-179.999,179.999,50,89.99])
sub_ax.text(0, 1, 'a', transform=sub_ax.transAxes,
        fontsize=8, fontweight='bold', va='top', ha='left')

ds = gdal.Open(fn_density)
hs = ds.ReadAsArray()
gt = ds.GetGeoTransform()  # Defining bounds
ext = (gt[0], gt[0] + ds.RasterXSize * gt[1],
           gt[3] + ds.RasterYSize * gt[5], gt[3])
hs_tmp = hs.astype(float)


# cmap2 = plt.get_cmap('Blues',100)
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('cmap1', ['white', 'tab:purple'], 100)

cmap2._init()
cmap2._lut[:, -1] = np.linspace(1,1,103)
cmap2._lut[0:2, -1] = 0.0  # We made transparent de 10 first levels of hillshade,

sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor=plt.cm.Greys(0.4)))
sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor=plt.cm.Greys(0.8)))
sub_ax.imshow(hs_tmp_l[:, :], extent=ext, transform=ccrs.Robinson(), cmap=cmaphs2, zorder=2,interpolation='nearest')
sub_ax.imshow(hs_tmp_nl[:, :], extent=ext, transform=ccrs.Robinson(), cmap=cmaphs22, zorder=2,interpolation='nearest')

sub_ax.outline_patch.set_edgecolor('lightgrey')

sub_ax.imshow(hs_tmp[:, :], extent=ext, transform=ccrs.Robinson(), cmap=cmap2, zorder=2,vmin=0,vmax=100, resample=True, interpolation='nearest')

fn_density = '/home/atom/ongoing/work_worldwide/coverage/ww_cov_aster_rob.tif'
sub_ax_2 = fig.add_subplot(grid[3:19, :8],projection=ccrs.Robinson())
sub_ax_2.set_global()
sub_ax_2.text(0, 1, 'b', transform=sub_ax_2.transAxes,
        fontsize=8, fontweight='bold', va='top', ha='left')


ds = gdal.Open(fn_density)
hs = ds.ReadAsArray()
gt = ds.GetGeoTransform()  # Defining bounds
ext = (gt[0], gt[0] + ds.RasterXSize * gt[1],
           gt[3] + ds.RasterYSize * gt[5], gt[3])
hs_tmp = hs.astype(float)

# cmap = plt.get_cmap('Reds',200)
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap1', ['white', 'tab:orange'], 200)

cmap._init()
cmap._lut[:, -1] = np.linspace(1,1,203)
cmap._lut[0:2, -1] = 0.0  # We made transparent de 10 first levels of hillshade,


sub_ax_2.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor=plt.cm.Greys(0.4)))
sub_ax_2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor=plt.cm.Greys(0.8)))


sub_ax_2.imshow(hs_tmp_l[:, :], extent=ext, transform=ccrs.Robinson(), cmap=cmaphs2, zorder=2, interpolation='nearest')
sub_ax_2.imshow(hs_tmp_nl[:, :], extent=ext, transform=ccrs.Robinson(), cmap=cmaphs22, zorder=2, interpolation='nearest')

sub_ax_2.outline_patch.set_edgecolor('lightgrey')

sub_ax_2.imshow(hs_tmp[:, :], extent=ext, transform=ccrs.Robinson(), cmap=cmap, zorder=2,vmin=0,vmax=200, resample=True, interpolation='nearest')

fn_density_2 = '/home/atom/ongoing/work_worldwide/coverage/ww_cov_rema_rob.tif'

sub_ax_3 = fig.add_subplot(grid[19:22, :8],projection=ccrs.Robinson())

sub_ax_3.set_extent([-179.999,179.999,-50,-89.99])

sub_ax_3.text(0, 0, 'c', transform=sub_ax_3.transAxes,
        fontsize=8, fontweight='bold', va='bottom', ha='left')
ds2 = gdal.Open(fn_density_2)
hs2 = ds2.ReadAsArray()
gt2 = ds2.GetGeoTransform()  # Defining bounds
ext2 = (gt2[0], gt2[0] + ds2.RasterXSize * gt2[1],
           gt2[3] + ds2.RasterYSize * gt2[5], gt2[3])
hs_tmp2 = hs2.astype(float)

# cmap3 = plt.get_cmap('Greens',50)
cmap3 = mpl.colors.LinearSegmentedColormap.from_list('cmap1', ['white', 'tab:blue'], 50)

cmap3._init()
cmap3._lut[:, -1] = np.linspace(1,1,53)
cmap3._lut[0:2, -1] = 0.0  # We made transparent de 10 first levels of hillshade,
sub_ax_3.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor=plt.cm.Greys(0.4)))
sub_ax_3.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor=plt.cm.Greys(0.8)))
sub_ax_3.imshow(hs_tmp_l[:, :], extent=ext, transform=ccrs.Robinson(), cmap=cmaphs2, zorder=2, interpolation='nearest')
sub_ax_3.imshow(hs_tmp_nl[:, :], extent=ext, transform=ccrs.Robinson(), cmap=cmaphs22, zorder=2, interpolation='nearest')

sub_ax_3.outline_patch.set_edgecolor('lightgrey')
sub_ax_3.imshow(hs_tmp2[:, :], extent=ext2, transform=ccrs.Robinson(), cmap=cmap3, zorder=2,vmin=0,vmax=50, resample=True, interpolation='nearest')


#COLORBARS


cb_ax = fig.add_subplot(grid[20:,:4],projection=ccrs.Robinson())
cb2_ax = fig.add_subplot(grid[20:,4:8],projection=ccrs.Robinson())
cb3_ax = fig.add_subplot(grid[20:,2:6],projection=ccrs.Robinson())
cb_ax.outline_patch.set_visible(False)
cb_ax.background_patch.set_visible(False)
cb2_ax.outline_patch.set_visible(False)
cb2_ax.background_patch.set_visible(False)
cb3_ax.outline_patch.set_visible(False)
cb3_ax.background_patch.set_visible(False)


norm = mpl.colors.Normalize(vmin=0,vmax=100)
sm = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
sm.set_array([])
norm = mpl.colors.Normalize(vmin=0,vmax=50)
sm2 = plt.cm.ScalarMappable(cmap=cmap3,norm=norm)
sm2.set_array([])
norm = mpl.colors.Normalize(vmin=0,vmax=200)
sm3 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm3.set_array([])

cb = plt.colorbar(sm , ax=cb_ax, ticks=np.arange(0,120,50),orientation='horizontal',extend='max',shrink=0.4,pad=0.05)
cb2 = plt.colorbar(sm2 , ax=cb2_ax, ticks=np.arange(0,70,50),orientation='horizontal',extend='max',shrink=0.4,pad=0.05)
cb.ax.tick_params(width=0.35,length=2.5)
cb.set_label('ArcticDEM strip count')
cb2.ax.tick_params(width=0.35,length=2.5)
cb2.set_label('REMA strip count')
cb3 = plt.colorbar(sm3 , ax=cb3_ax, ticks=np.arange(0,220,100), orientation='horizontal',extend='max',shrink=0.4,pad=0.05)
cb3.ax.tick_params(width=0.35,length=2.5)
cb3.set_label('ASTER strip count')


fn_tcov_arc = '/home/atom/ongoing/work_worldwide/coverage/tcov_arcticdem.csv'
fn_tcov_rema = '/home/atom/ongoing/work_worldwide/coverage/tcov_rema.csv'
fn_tcov_aster = '/home/atom/ongoing/work_worldwide/coverage/tcov_aster.csv'
df_arc = pd.read_csv(fn_tcov_arc)
df_rema = pd.read_csv(fn_tcov_rema)
df_aster = pd.read_csv(fn_tcov_aster)

ax2 = fig.add_subplot(grid[:,9:])

ax2.text(1, 1, 'd', transform=ax2.transAxes,
        fontsize=8, fontweight='bold', va='top', ha='right')

def datetime_to_decyears(t_vals):
    y0 = min(t_vals).astype('datetime64[D]').astype(object).year
    y1 = max(t_vals).astype('datetime64[D]').astype(object).year
    ftime = t_vals
    total_delta = np.datetime64('{}-01-01'.format(int(y1))) - np.datetime64('{}-01-01'.format(int(y0)))
    ftime_delta = np.array([t - np.datetime64('{}-01-01'.format(int(y0))) for t in ftime])
    time_vals = (ftime_delta / total_delta) * (int(y1) - int(y0)) + y0

    return time_vals

rema_dec = datetime_to_decyears(df_rema.date.values.astype('datetime64[D]'))
aster_dec = datetime_to_decyears(df_aster.date.values.astype('datetime64[D]'))
arc_dec = datetime_to_decyears(df_arc.date.values.astype('datetime64[D]'))

ax2.hist([aster_dec,arc_dec,rema_dec],120,color=['tab:orange','tab:purple','tab:blue'],label=['ASTER: 154,565','ArcticDEM: 67,986','REMA: 9,369'],edgecolor='black',lw=0.25,stacked=True,orientation='horizontal')
ax2.set_xlabel('Bi-mensual strip count')
# ax2.set_ylabel('Time')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# ax2.spines['bottom'].set_visible(False)
# ax2.spines['left'].set_visible(False)
# ax2.get_xaxis().set_ticks([])
ax2.set_ylim([2020,2000])
ax2.get_yaxis().set_ticks([2000,2004,2008,2012,2016,2020])
ax2.tick_params(width=0.35,length=2.5)
ax2.legend(loc=[0.4,0.5])
# yloc = YearLocator()
# mloc = MonthLocator()
# ax.xaxis.set_major_locator(yloc)
# ax.xaxis.set_minor_locator(mloc)

plt.savefig('/home/atom/ongoing/work_worldwide/figures/final/ED_Figure_2.jpg',dpi=500,bbox_inches='tight')