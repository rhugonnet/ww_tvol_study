
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gdal
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import ConvexHull
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import matplotlib.colors as mcolors
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
from pybob.GeoImg import GeoImg
from pybob.image_tools import create_mask_from_shapefile
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pybob.ddem_tools import nmad
from glob import glob
import scipy.interpolate
from sklearn.linear_model import LinearRegression
import ogr, osr
import pyddem.vector_tools as ot
import matplotlib
matplotlib.rcParams.update({'font.size': 12})


fn_dh = '/home/atom/ongoing/work_worldwide/figures/fig3/01_02_rgi60_fig/fig3.vrt'
fn_shp ='/home/atom/data/inventory_products/RGI/00_rgi60/rgi60_buff_diss.shp'
fn_hs = '/home/atom/documents/paper/Hugonnet_2020/figures/world_robin_ak.tif'
fn_hs_dateline='/home/atom/documents/paper/Hugonnet_2020/figures/world_robin_ak_dateline.tif'
fn_land = '/home/atom/data/inventory_products/NaturalEarth/ne_50m_land/ne_50m_land.shp'
fn_rgi_sub = '/home/atom/ongoing/work_worldwide/figures/fig3/rgi_sub_fig3.shp'
fn_ib_zsc='/home/atom/ongoing/work_worldwide/figures/fig3/zsc_icebridge_300m.tif'
fn_ics_zsc ='/home/atom/ongoing/work_worldwide/figures/fig3/zsc_icesat_800m.tif'
fn_water_poly = '/home/atom/ongoing/work_worldwide/figures/fig3/fig3_water_poly.shp'
fn_land_poly= '/home/atom/ongoing/work_worldwide/figures/fig3/fig3_land_poly.shp'

fig = plt.figure(figsize=(10,15.5))
# grid = plt.GridSpec(20, 20, wspace=0.3, hspace=0.3)

#PANEL A

ax2 = fig.add_axes([0.0025,0.475,0.995,0.52],projection=ccrs.UTM(7),label='Zoom_1')

fn_ib_diss ='/home/atom/ongoing/work_worldwide/figures/fig3/fig3_icebridge_300m_diss.shp'
fn_ic_diss = '/home/atom/ongoing/work_worldwide/figures/fig3/fig3_icesat_800m_diss.shp'

ext=[290000,685000,6550000-25000,6845000]

ax2.set_extent(ext, ccrs.UTM(7))

shape_feature = ShapelyFeature(Reader(fn_land_poly).geometries(), ccrs.UTM(7), edgecolor='None', alpha=1,
                               facecolor=plt.cm.RdYlBu(0.5),linewidth=1,zorder=1)
ax2.add_feature(shape_feature)
ax2.outline_patch.set_edgecolor('black')

ds = gdal.Open(fn_dh)
hs = ds.ReadAsArray()
# hs[water_notglacier] = np.nan
gt = ds.GetGeoTransform()  # Defining bounds
ext = (gt[0], gt[0] + ds.RasterXSize * gt[1],
           gt[3] + ds.RasterYSize * gt[5], gt[3])
hs_tmp = hs.astype(float)
hs_tmp[hs_tmp<=-9999] = np.nan

cmap = plt.get_cmap('RdYlBu',100)
cmap._init()
cmap._lut[:, -1] = 1.
# cmap2._lut[0, -1] = 0.0  # We made transparent de 10 first levels of hillshade,
cmap.set_bad(color='None')
ax2.imshow(np.flip(hs_tmp[:, :], axis=0), extent=ext, transform=ccrs.UTM(7), vmin=-50,vmax=50,cmap=cmap,interpolation='bilinear',zorder=2)

shape_feature = ShapelyFeature(Reader(fn_rgi_sub).geometries(), ccrs.PlateCarree(), edgecolor='grey', alpha=1,
                               facecolor='None',linewidth=0.5,zorder=3)
ax2.add_feature(shape_feature)

ds = gdal.Open(fn_ics_zsc)
hs = ds.ReadAsArray()
# hs[water_notglacier] = np.nan
gt = ds.GetGeoTransform()  # Defining bounds
ext = (gt[0], gt[0] + ds.RasterXSize * gt[1],
           gt[3] + ds.RasterYSize * gt[5], gt[3])
hs_tmp = hs.astype(float)
hs_tmp[hs_tmp<=-9999] = np.nan

cmap2 = plt.get_cmap('Purples',100)
cmap2._init()
cmap2._lut[:, -1] = 1.
# cmap2._lut[0, -1] = 0.0  # We made transparent de 10 first levels of hillshade,
cmap2.set_bad(color='None')
ax2.imshow(np.flip(hs_tmp[:, :], axis=0), extent=ext, transform=ccrs.UTM(7), vmin=0,vmax=2,cmap=cmap2,interpolation='nearest',zorder=4)

shape_feature = ShapelyFeature(Reader(fn_ic_diss).geometries(), ccrs.UTM(7), edgecolor='black', alpha=1,
                               facecolor='None',linewidth=0.8,linestyle='dashed',zorder=5)
ax2.add_feature(shape_feature)

ds = gdal.Open(fn_ib_zsc)
hs = ds.ReadAsArray()
# hs[water_notglacier] = np.nan
gt = ds.GetGeoTransform()  # Defining bounds
ext = (gt[0], gt[0] + ds.RasterXSize * gt[1],
           gt[3] + ds.RasterYSize * gt[5], gt[3])
hs_tmp = hs.astype(float)
hs_tmp[hs_tmp<=-9999] = np.nan

ax2.imshow(np.flip(hs_tmp[:, :], axis=0), extent=ext, transform=ccrs.UTM(7), vmin=0,vmax=2,cmap=cmap2,interpolation='nearest',zorder=6)

shape_feature = ShapelyFeature(Reader(fn_ib_diss).geometries(), ccrs.UTM(7), edgecolor='black', alpha=1,
                               facecolor='None',linewidth=0.8,zorder=7)
ax2.add_feature(shape_feature)

shape_feature = ShapelyFeature(Reader(fn_water_poly).geometries(), ccrs.UTM(7), edgecolor='None', alpha=1,
                               facecolor='gainsboro',linewidth=1,zorder=8)
ax2.add_feature(shape_feature)
ax2.text(0.015, 0.975, 'b', transform=ax2.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left',zorder=20)


cbaxes = ax2.inset_axes([0.51,0.23,0.2,0.025],zorder=9)

norm = mpl.colors.Normalize(vmin=-50,vmax=50)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = plt.colorbar(sm ,cax=cbaxes, ticks=[-50,-25,0,25,50], orientation='horizontal',extend='both',shrink=0.4)
cb.set_label('Elevation change\nbetween 2000 and 2019 (m)')

cbaxes2 = ax2.inset_axes([0.51,0.11,0.2,0.025],zorder=9)

norm2 = mpl.colors.Normalize(vmin=0,vmax=2)
sm = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
sm.set_array([])
cb = plt.colorbar(sm ,cax=cbaxes2, ticks=[0,1,2], orientation='horizontal',extend='max',shrink=0.4)
cb.set_label('Standardized elevation difference\nto ICESat and IceBridge $z$\n(or z-score: see panel C)')

ax2.add_patch(mpatches.Rectangle((605000,6539000),25000,5000,edgecolor='black',facecolor='black',transform=ccrs.UTM(7),zorder=10,linewidth=0.5))
ax2.add_patch(mpatches.Rectangle((630000,6539000),25000,5000,edgecolor='black',facecolor='white',transform=ccrs.UTM(7),zorder=10,linewidth=0.5))
ax2.text(605000,6537000,'0 km',ha='center',va='top',transform=ccrs.UTM(7),zorder=10)
ax2.text(630000,6537000,'25 km',ha='center',va='top',transform=ccrs.UTM(7),zorder=10)
ax2.text(655000,6537000,'50 km',ha='center',va='top',transform=ccrs.UTM(7),zorder=10)

ax2.text(298000,6657500,'                                                             \n                                                            \n                                                             ',bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),va='top',ha='left',zorder=11)
ax2.add_patch(mpatches.Rectangle((300000,6651000),25000,2000,edgecolor="black",facecolor=plt.cm.Purples(0.25),linewidth=1,zorder=12,linestyle='dashed'))
# ax2.add_patch(mpatches.ConnectionPatch((300000,6651000),(325000,6651000),"data","data",linestyle='dashed',color='black',linewidth=1,zorder=12))
# ax2.add_patch(mpatches.ConnectionPatch((300000,6653000),(325000,6653000),"data","data",linestyle='dashed',color='black',linewidth=1,zorder=12))
ax2.text(328000,6652000,'ICESat: 95,492 points',ha='left',va='center',transform=ccrs.UTM(7),zorder=12)
ax2.add_patch(mpatches.Rectangle((300000,6641000),25000,2000,edgecolor="black",facecolor=plt.cm.Purples(0.25),linewidth=1,zorder=12))
# ax2.add_patch(mpatches.ConnectionPatch((300000,6641000),(325000,6641000),"data","data",color='black',linewidth=1,zorder=12))
# ax2.add_patch(mpatches.ConnectionPatch((300000,6643000),(325000,6643000),"data","data",color='black',linewidth=1,zorder=12))
ax2.text(328000,6642000,'IceBridge: 4,824,679 points',ha='left',va='center',transform=ccrs.UTM(7),zorder=13)

# ax2.text(0.5, 0.9939, 'ALA (01)', transform=ax2.transAxes,
#         fontsize=12, fontweight='bold', va='top', ha='center',zorder=20,bbox=dict(boxstyle='square', facecolor='white', alpha=1))
#Inset AK + Zoom Bagley Icefield + Zoom Temporal Series

ax = fig.add_axes([0.79,0.8625,0.2,0.16], projection=ccrs.NorthPolarStereo(central_longitude=-140),label='inset_ALA')

ax.set_extent([-158,-110,45,70])

ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor='gainsboro'))
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor='dimgrey'))
ax.outline_patch.set_edgecolor('black')

ax.text(-144,57,'Gulf\nof\nAlaska',ha='center',va='top',color='grey',transform=ccrs.PlateCarree(),fontsize=10,fontweight='bold')
# ax.text(-115,35,'US',ha='center',va='bottom',color='grey',transform=ccrs.PlateCarree(),fontsize=8,fontweight='bold')
# ax.text(-115,35,'British Columbia',ha='center',va='bottom',color='grey',transform=ccrs.PlateCarree(),fontsize=8,fontweight='bold')

def poly_from_extent(ext):

    poly = np.array([(ext[0],ext[2]),(ext[1],ext[2]),(ext[1],ext[3]),(ext[0],ext[3]),(ext[0],ext[2])])

    return poly
ax.add_patch(mpatches.Polygon(poly_from_extent(ext),transform=ccrs.UTM(7),facecolor='None',edgecolor=plt.cm.Greys(0.9),zorder=4))

ax.text(-130,62,'Saint-Elias\nMoutains',ha='left',va='center',color=plt.cm.Greys(0.9),transform=ccrs.PlateCarree(),fontsize=10,fontweight='bold',zorder=5)

ax.text(0.07, 0.05, 'a', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='bottom', ha='left',zorder=20)

#
# img = GeoImg(fn_hs_dateline)
# land_mask = create_mask_from_shapefile(img,fn_land)
# ds = gdal.Open(fn_hs_dateline)
# hs = ds.ReadAsArray()
# hs = hs.astype(float)

def stretch_hs(hs,stretch_factor=1.):

    max_hs = 255
    min_hs = 0

    hs_s = (hs - (max_hs-min_hs)/2)*stretch_factor + (max_hs-min_hs)/2

    return hs_s

# hs = stretch_hs(hs,stretch_factor=0.9)
#
# hs_land = hs.copy()
# hs_land[~land_mask]=0
# hs_notland = hs.copy()
# hs_notland[land_mask]=0
#
# gt = ds.GetGeoTransform()  # Defining bounds
# ext = (gt[0], gt[0] + ds.RasterXSize * gt[1],
#            gt[3] + ds.RasterYSize * gt[5], gt[3])
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

# ax.imshow(np.flip(hs_land[:, :],axis=0), extent=ext, transform=ccrs.Robinson(), cmap=cmap2, zorder=2)
# ax.imshow(np.flip(hs_notland[:,:],axis=0),extent=ext, transform=ccrs.Robinson(),cmap=cmap22, zorder=2)
# ds=None

img = GeoImg(fn_hs)
land_mask = create_mask_from_shapefile(img,fn_land)
ds = gdal.Open(fn_hs)
hs = ds.ReadAsArray()
hs = hs.astype(float)
hs_land = hs.copy()
hs_land[~land_mask]=0
hs_notland = hs.copy()
hs_notland[land_mask]=0

gt = ds.GetGeoTransform()  # Defining bounds
ext = (gt[0], gt[0] + ds.RasterXSize * gt[1],
           gt[3] + ds.RasterYSize * gt[5], gt[3])

ax.imshow(np.flip(hs_land[:, :],axis=0), extent=ext, transform=ccrs.Robinson(), cmap=cmap2, zorder=2)
ax.imshow(np.flip(hs_notland[:,:],axis=0),extent=ext, transform=ccrs.Robinson(),cmap=cmap22, zorder=2)
ds=None

shape_feature = ShapelyFeature(Reader(fn_shp).geometries(), ccrs.PlateCarree(), edgecolor='None', alpha=0.7,
                               facecolor='indigo',linewidth=1,zorder=3)
ax.add_feature(shape_feature)

# ax2.add_patch(mpatches.ConnectionPatch((501000,6655000),(300000,6630000),"data","data",linestyle='dotted',color='dimgrey',linewidth=1.2,zorder=11))
# ax2.add_patch(mpatches.ConnectionPatch((501000,6655000),(450000,6550000),"data","data",linestyle='dotted',color='dimgrey',linewidth=1.2,zorder=11))
polygon = np.array([(501000,6655000),(293000,6631000),(467000,6530000)])
ax2.add_patch(mpatches.Polygon(polygon,edgecolor='grey',facecolor='dimgrey',linestyle='dotted',zorder=11,alpha=0.25))

ax2.add_patch(mpatches.ConnectionPatch((501000-2000,6655000-2000),(501000+2000,6655000+2000),"data","data",color='black',linewidth=4,zorder=12))
ax2.add_patch(mpatches.ConnectionPatch((501000-2000,6655000+2000),(501000+2000,6655000-2000),"data","data",color='black',linewidth=4,zorder=12))

#SUBPANEL B

ax3 = fig.add_axes([0.01,0.4775,0.44,0.17],label='Zoom_2')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_xlim((0,1))
ax3.set_ylim((0,1))
ax3.text(0.025, 0.95, 'c', transform=ax3.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left',zorder=20)


fn_pred = '/home/atom/ongoing/work_worldwide/figures/fig3/pred_agassiz.csv'
fn_raw = '/home/atom/ongoing/work_worldwide/figures/fig3/raw_agassiz.csv'
df_pred = pd.read_csv(fn_pred)
df_raw = pd.read_csv(fn_raw)

sub_ax3 = ax3.inset_axes([0.175,0.175,0.8,0.8])

# fig, sub_ax3 = plt.subplots(figsize=(4,3))
ns=1000
for i in range(ns):
    sub_ax3.fill_between(2000+df_pred.t.values, df_pred.h_pred.values + i*3/ns*df_pred.sigma_pred.values,
                df_pred.h_pred.values + (i+1)*3/ns*df_pred.sigma_pred.values, facecolor=plt.cm.Purples(min(0.999,3/2*i/ns)), alpha=1,edgecolor='None')
    sub_ax3.fill_between(2000 + df_pred.t.values, df_pred.h_pred.values - i*3/ ns * df_pred.sigma_pred.values,
                         df_pred.h_pred.values - (i + 1)*3/ns * df_pred.sigma_pred.values,
                         facecolor=plt.cm.Purples(min(0.999,3/2*i/ns)), alpha=1,edgecolor='None')

# sub_ax3.plot(2000 + df_pred.t.values, df_pred.h_pred.values + df_pred.sigma_pred.values, alpha=1,color='lightgrey',lw=0.5)
# sub_ax3.text(2000 + df_pred.t.values[60], df_pred.h_pred.values[68] - df_pred.sigma_pred.values[68],'1$\sigma$',bbox= dict(boxstyle='round', facecolor='white', alpha=0.8),ha='right',va='center')
# sub_ax3.plot(2000 + df_pred.t.values, df_pred.h_pred.values + 2*df_pred.sigma_pred.values, alpha=1,color='lightgrey',lw=0.5)
# sub_ax3.text(2000 + df_pred.t.values[70], df_pred.h_pred.values[78] - 2*df_pred.sigma_pred.values[78], '2$\sigma$',bbox= dict(boxstyle='round', facecolor='white', alpha=0.8),ha='right',va='center')
# sub_ax3.plot(2000 + df_pred.t.values, df_pred.h_pred.values + 3*df_pred.sigma_pred.values, alpha=1,color='lightgrey',lw=0.5)
# sub_ax3.text(2000 + df_pred.t.values[80], df_pred.h_pred.values[88] - 3*df_pred.sigma_pred.values[88], '3$\sigma$',bbox= dict(boxstyle='round', facecolor='white', alpha=0.8),ha='right',va='center')
# sub_ax3.vlines(2000 + df_pred.t.values[68], df_pred.h_pred.values[68],df_pred.h_pred.values[68]  - 1*df_pred.sigma_pred.values[68],color='black',linestyle=(0,(4,4)),linewidth=0.5)
# sub_ax3.vlines(2000 + df_pred.t.values[78], df_pred.h_pred.values[78],df_pred.h_pred.values[78] - 2*df_pred.sigma_pred.values[78],color='black',linestyle=(0,(4,4)),linewidth=0.5)
# sub_ax3.vlines(2000 + df_pred.t.values[88], df_pred.h_pred.values[88],df_pred.h_pred.values[88] - 3*df_pred.sigma_pred.values[88],color='black',linestyle=(0,(4,4)),linewidth=0.5)
# sub_ax3.arrow(2000 + df_pred.t.values[68], df_pred.h_pred.values[68]- 0.99*df_pred.sigma_pred.values[68], 0,  - 0.05*df_pred.sigma_pred.values[68],color='black',shape='full',head_width=0.25,head_length=2.5,length_includes_head=True)
# sub_ax3.arrow(2000 + df_pred.t.values[78], df_pred.h_pred.values[78]- 1.99*df_pred.sigma_pred.values[78], 0,  - 0.05*df_pred.sigma_pred.values[78],color='black',shape='full',head_width=0.25,head_length=2.5,length_includes_head=True)
# sub_ax3.arrow(2000 + df_pred.t.values[88], df_pred.h_pred.values[88]- 2.99*df_pred.sigma_pred.values[88], 0,  - 0.05*df_pred.sigma_pred.values[88],color='black',shape='full',head_width=0.25,head_length=2.5,length_includes_head=True)

p1=sub_ax3.plot(2000 + df_pred.t.values, df_pred.h_pred.values, lw=1.5, color='black',label='Time series')
p2=sub_ax3.errorbar(2000 + df_raw.t_raw.values, df_raw.h_raw.values, df_raw.err_raw.values, fmt='x', color=plt.cm.Oranges(0.95),elinewidth=1.5,markersize=8,label='Obs. (1$\sigma$ CI)')
p3=sub_ax3.scatter(2004.3,356.2,marker='d',color='black',s=50,zorder=30,label='ICESat')


sub_ax3.text(2004.3, 356.2-10,'z=0',bbox= dict(boxstyle='round', facecolor='white', alpha=0.8),ha='center',va='top')
sub_ax3.text(2010, 358.2, 'z=1.5',bbox= dict(boxstyle='round', facecolor='white', alpha=0.8),ha='left',va='center')

# sub_ax3.vlines(2004.3,0,600,linestyles='dotted',colors='black')
sub_ax3.scatter(2008.8,358.2,marker='d',color='black',s=50,zorder=30)
# sub_ax3.vlines(2008.8,0,600,linestyles='dotted',colors='black')
p4=sub_ax3.scatter(2015.5,335.1,marker='o',color='black',s=50,zorder=30,label='IceBridge')
# sub_ax3.vlines(2015.5,0,600,colors='black')
sub_ax3.scatter(2019.8,310.3,marker='o',color='black',s=50,zorder=30)
# sub_ax3.vlines(2019.8,0,600,colors='black')
sub_ax3.set_ylim(290,430)

sub_ax3.set_xlabel('Year')
sub_ax3.set_ylabel('Elevation (m)')
list_labs=[p1[0],p2,p3,p4]
sub_ax3.legend(list_labs,[lab.get_label() for lab in list_labs],loc='upper right',ncol=2,columnspacing=0.2)
sub_ax3.set_xticks(np.arange(2000, 2024, 4))
# sub_ax3.grid()
# sub_ax3.set_axisbelow(True)



#PANEL WORLD LET'S GO !!!

# fig = plt.figure(figsize=(10,14.5))
#
ax0 = fig.add_axes([0.0025,0.005,0.995,0.465],label='Stats')

# fig = plt.figure(figsize=(10,7))
# ax0  = fig.add_axes([0,0,1,1],label='Stats')

ax0.text(0.015, 0.98, 'd', transform=ax0.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left',zorder=20)
ax0.text(0.02, 0.875, 'GLOBAL', transform=ax0.transAxes,
        fontsize=12, fontweight='bold', va='top', ha='left',zorder=20,rotation='vertical')
ax0.text(0.02, 0.475, 'CATEGORIES', transform=ax0.transAxes,
        fontsize=12, fontweight='bold', va='top', ha='left',zorder=20,rotation='vertical')
ax0.set_xticks([])
ax0.set_yticks([])

bin_dt = [0,60,120,180,240,300,360,540,720,900,1080]
bin_t = [np.datetime64('20'+str(i).zfill(2)+'-01-01') for i in range(21)]
bin_h = np.arange(0,1.1,0.1)
bin_dh_tot = [-150,-100,-50,-35,-15,-10,-5,0,5,10,15]
bin_reg = np.arange(1, 21)

# df = pd.read_csv('/home/atom/ongoing/work_worldwide/validation/compiled/valid_ICS_IB_all_bins_init.csv')
df = pd.read_csv('/home/atom/ongoing/work_worldwide/validation/compiled/seas_corr/valid_ICS_IB_all_bins_final_weight.csv')
df = df[df.seas_corr==1]

df_seas = pd.read_csv('/home/atom/ongoing/work_worldwide/validation/compiled/seas_corr/valid_ICS_IB_seas_corr_final_weight.csv')
#create the two colorbars
def add_inset(ax0,pos,df,bin_vec,xlab,cont=False,mu=True,mudt=False,ylab=True,add_mu_lab=None,month_xlab=False,draw_edges=True):

    ax = ax0.inset_axes(pos)

    if add_mu_lab is None:
        mu_lab='$\mu_{dh}$'
    else:
        mu_lab='$\mu_{dh}$\n'+add_mu_lab

    # FOR STD
    col_bounds = np.array([0,0.5,1])
    col_bounds = np.array(col_bounds)
    cb_val = np.linspace(0, 1, len(col_bounds))
    fun = scipy.interpolate.interp1d(col_bounds, cb_val, fill_value=(0, 0.999), bounds_error=False)

    # FOR MEAN
    col_bounds2 = np.array([-2, -1, 0, 1, 2])
    col_bounds2 = np.array(col_bounds2)
    cb_val2 = np.linspace(0, 1, len(col_bounds2))
    fun2 = scipy.interpolate.interp1d(col_bounds2, cb_val2, fill_value=(0, 0.999), bounds_error=False)

    col_bounds3 = np.array([0,0.1,0.2])
    col_bounds3 = np.array(col_bounds3)
    cb_val3 = np.linspace(0, 1, len(col_bounds3))
    fun3 = scipy.interpolate.interp1d(col_bounds3, cb_val3, fill_value=(0, 0.999), bounds_error=False)

    for i in range(len(bin_vec)-1):

        if mu and mudt:
            y_mu =2.5
            y_mudt = 1.25
            y_ns = 3.6
            ticks = [0.5, 1.75, 3]
            tickslab=['$\sigma_{z}$','|$ \\beta_{dh} $|',mu_lab]
        elif mu and not mudt:
            y_mu=1.25
            y_ns = 2.3
            ticks = [0.5, 1.75]
            tickslab=['$\sigma_{z}$',mu_lab]
        elif not mu and mudt:
            y_mudt =1.25
            y_ns=2.3
            ticks = [0.5, 1.75]
            tickslab=['$\sigma_{z}$','|$ \\beta_{dh} $|']
        else:
            y_ns = 1.05
            ticks = [0.5]
            tickslab=['$\sigma_{z}$']


        if mu and ~np.isnan(df.med_dh.values[i]):
            ax.fill_between([i,i+1],[y_mu,y_mu],[y_mu+1,y_mu+1],color=mpl.cm.RdYlBu(fun2(df.med_dh.values[i])))
        elif mu:
            ax.fill_between([i, i + 1], [y_mu, y_mu], [y_mu+1, y_mu+1], color='grey')

        if mudt and ~np.isnan(df.dzsc_dt.values[i]):
            ax.fill_between([i,i+1],[y_mudt,y_mudt],[y_mudt+1,y_mudt+1],color=mpl.cm.Oranges(fun3(np.abs(df.dzsc_dt.values[i]*df.nmad_dh.values[i]))))
        elif mudt:
            ax.fill_between([i, i + 1], [y_mudt, y_mudt], [y_mudt+1, y_mudt+1], color='grey')

        if ~np.isnan(df.nmad_zsc.values[i]):
            ax.fill_between([i,i+1],[0,0],[1,1],color=mpl.cm.Purples(fun(df.nmad_zsc.values[i])))
        else:
            ax.fill_between([i, i + 1], [0, 0], [1, 1], color='grey')

        if df.ns_ics.values[i] !=0:
            ax.fill_between([i+0.1,i+0.5],[y_ns,y_ns],[y_ns+4*df.ns_ics.values[i]/np.sum(df.ns_ics.values)]*2,color=plt.cm.Greys(0.3))
        if df.ns_ib.values[i] !=0:
            ax.fill_between([i+0.5,i+0.9],[y_ns,y_ns],[y_ns+4*df.ns_ib.values[i]/np.sum(df.ns_ib.values)]*2,color=plt.cm.Greys(0.7))

        if draw_edges:
            for j in range(len(ticks)):
                ax.vlines(i, ticks[j] - 0.5, ticks[j] + 0.5, lw=0.8)
                ax.hlines([ticks[j] - 0.5, ticks[j] + 0.5], [i, i], [i + 1, i + 1], lw=0.8)
                if i == len(bin_vec) - 2:
                    ax.vlines(i + 1, ticks[j] - 0.5, ticks[j] + 0.5, lw=0.8)
            # ax.hlines(ticks[len(ticks)]+0.5,i,i+1)

    ax.set_ylim(0,max(ticks)+2.25)
    ax.set_xlim(0,len(bin_vec)-1)
    if ylab:
        ax.set_yticks(ticks)
        ax.set_yticklabels(tickslab,rotation='horizontal')
    else:
        ax.set_yticks([])
    if not cont:
        ax.set_xticks(np.arange(0, len(bin_vec), 2))
        ax.set_xticklabels(bin_vec[::2])
    else:
        if not month_xlab:
            ax.set_xticks(np.arange(0.5,len(bin_vec)-1,1))
            ax.set_xticklabels(bin_vec[:-1])
        else:
            ax.set_xticks([0.5,1.5, 2.5,3.5 ,4.5,5.5])
            ax.set_xticklabels(bin_vec)
    ax.set_xlabel(xlab)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.get_yaxis().set_ticks([])

def inset_hist(ax0,pos,df,bin_vec,xlab,val_mean=None,val_std=None,val_dmean=None,draw_edges=True,normal=True):

    ax = ax0.inset_axes(pos)

    for i in range(len(bin_vec)-1):
        if df.ns_ics.values[i] !=0:
            ax.fill_between([i+0.1,i+0.5],[0,0],[10*df.ns_ics.values[i]/np.sum(df.ns_ics.values)]*2,color=plt.cm.Greys(0.3))
        if df.ns_ib.values[i] != 0:
            ax.fill_between([i + 0.5, i + 0.9], [0, 0], [10 * df.ns_ib.values[i] / np.sum(df.ns_ib.values)] * 2,
                    color=plt.cm.Greys(0.7))


    fun = scipy.interpolate.interp1d(bin_vec,np.arange(0,len(bin_vec)))
    if val_mean is not None:
        ax.vlines(fun(val_mean),0,4,lw=2,colors='black')
        ax.text(fun(val_mean)-1,3,'$\mu_{dh}$ =\n'+str(np.round(val_mean,2))+' m',va='center',ha='right',bbox=dict(boxstyle='round', facecolor=plt.cm.RdYlBu(0.5-0.47/10.), alpha=1))
    if val_std is not None:
        ax.vlines(fun(val_std),0,4,lw=2,linestyles='dashed',colors='black')
        ax.vlines(fun(-val_std),0,4,lw=2,linestyles='dashed',colors='black')
        ax.text(fun(-val_std)-1,3,'$\sigma_{z}$ = '+str(np.round(val_std,2)),va='center',ha='right',bbox= dict(boxstyle='round', facecolor=plt.cm.Purples(0.47), alpha=1))
    if val_dmean is not None:
        ax.text(fun(val_mean)+2.5,3,'$|d\mu_{dh}/dt|$ =\n'+str(np.round(val_dmean,3))+' m yr$^{-1}$',va='center',ha='left',bbox=dict(boxstyle='round', facecolor=plt.cm.Oranges(0.01/0.2), alpha=1))

    if normal:
        ax.set_ylim(0,5)
        ax.set_xlim(0, len(bin_vec) - 1)
        ax.set_yticks([])
        ax.set_xticks(np.arange(0, len(bin_vec), 2))
        if val_std:
            ax.set_xticklabels(bin_vec[::2].astype(int))
        else:
            ax.set_xticklabels(bin_vec[::2])
        ax.set_xlabel(xlab)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:
        ax.set_ylim(0, 2)
        ax.set_xlim(0, len(bin_vec) - 1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

def inset_wls(ax0,pos,x,y,mu_y,beta2):

    ax = ax0.inset_axes(pos)

    # x = 2000 + np.array(
    #     [(t - np.datetime64('{}-01-01'.format(int(2000)))).astype(int) / 365.2422 for t in x])

    ax.errorbar(x,y,mu_y,fmt='x',color=plt.cm.Greys(0.8))

    yvec = (2000-x)*beta2[0]
    ax.plot(x,yvec,color='black')

    ax.text(2002.5,1.75,'$\\beta_{dh}$ = '+str(-np.round(beta2[0],3))+'\n$\pm$ '+str(np.round(beta2[1],3))+' m yr$^{-1}$',bbox=dict(boxstyle='round', facecolor=plt.cm.Oranges(0.001/0.2), alpha=1))

    ax.set_xlabel('Year')
    ax.set_ylabel('Seasonally de-biased\n$\mu_{dh} (m)$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim((-5,5))
    ax.set_xlim((2000,2020))
    ax.set_xticks(np.arange(2000,2021,4))

h = df[df.type=='h']
t = df[df.type=='t']
dt = df[df.type=='dt']
reg = df[df.type=='reg']

sum_dh = []
for r in sorted(list(set(list(df_seas.reg)))):
    df_reg = df_seas[df_seas.reg==r]
    if r>15:
        summer_dh = df_reg.amp.values[0] * np.sin( (2.5+df_reg.phase.values[0])* 2 * np.pi / 12) + df_reg.h_shift.values[0]
    else:
        summer_dh = df_reg.amp.values[0] * np.sin( (8.5+df_reg.phase.values[0])* 2 * np.pi / 12) + df_reg.h_shift.values[0]
    sum_dh.append(summer_dh)

reg.med_dh = sum_dh

dh_tot = df[df.type=='dh_tot']

#seasons require... a bit more processing
df_north = df[df.type=='seas_north']
df_north.med_dh = df_north.med_dh - np.nanmean(df_north.med_dh)
df_south = df[df.type=='seas_south']
df_south.med_dh = df_south.med_dh - np.nanmean(df_south.med_dh)

# coefs1, _ = scipy.optimize.curve_fit(lambda t, a, b, c: a ** 2 * np.sin(t * 2 * np.pi / 12 + c) + b,
#                                          seas.mid_bin[~np.isnan(seas.med_dh)].values, seas.med_dh[~np.isnan(seas.med_dh)].values)
bin_seas = ['JF','MA','MJ','JA','SO','ND','JF']

add_inset(ax0,[0.395,0.2,0.25,0.14],h,np.round(bin_h*10)/10., 'Regionally normalized elevation',mu=False,mudt=True)
add_inset(ax0,[0.725,0.2,0.25,0.14],dh_tot,bin_dh_tot,'Elevation change over 20 years (m)',mu=False,mudt=True)
add_inset(ax0,[0.07,0.2,0.25,0.14],dt,bin_dt,'Time lag to closest observation (days)',mu=False,mudt=True)
add_inset(ax0,[0.09,0.42,0.15,0.185],df_north,bin_seas,'Season (N. hemi.)',mu=True,mudt=True,add_mu_lab='(seas.)',month_xlab=True,cont=True)
add_inset(ax0,[0.25,0.42,0.15,0.185],df_south,bin_seas,'Season (S. hemi)',mu=True,mudt=True,ylab=False,month_xlab=True,cont=True)
bin_reg_str = [str(reg).zfill(2) for reg in bin_reg]
add_inset(ax0,[0.5,0.42,0.48,0.185],reg,bin_reg_str,'RGI region',cont=True,mu=True,mudt=True,add_mu_lab='(summer)')
# add_inset(ax0,[0.35,0.87,0.6,0.115],t,[t.astype('object').year for t in bin_t], 'Year',mu=False,mudt=False)

df_tmp = df[df.type=='reg']
dzsc_dt_glob = np.nansum(df_tmp.dzsc_dt.values*(df_tmp.ns_ics.values+df_tmp.ns_ib.values/40.))/np.nansum(df_tmp.ns_ics.values+df_tmp.ns_ib.values/40.)
nmad_dh_glob = np.nansum(df_tmp.nmad_dh.values*(df_tmp.ns_ics.values+df_tmp.ns_ib.values/40.))/np.nansum(df_tmp.ns_ics.values+df_tmp.ns_ib.values/40.)
ddh_dt_glob = df[df.type=='all'].dzsc_dt.values[0]*df[df.type=='all'].nmad_dh.values[0]
ddt_dt_glob_2std = df[df.type=='all'].dzsc_dt_2std.values[0]*df[df.type=='all'].nmad_dh.values[0]
bin_dh = np.arange(-12,13,2)
bin_zsc = np.arange(-3,3.1,0.5)
dh = df[df.type=='dh']
zsc = df[df.type=='zsc']
inset_hist(ax0,[0.25,0.725,0.15,0.15],zsc,bin_zsc,'Z-scores $z$',val_std=df[df.type=='all'].nmad_zsc.values[0])
inset_hist(ax0,[0.062,0.725,0.15,0.15],dh,bin_dh,'Elevation differences\nto ICESat and IceBridge $dh$ (m)',val_mean=df[df.type=='all'].med_dh.values[0])

tmp_dt = df[df.type=='t']

df_ls = pd.read_csv('/home/atom/ongoing/work_worldwide/validation/compiled/valid_ICS_IB_all_bins_final_weight_all_ls.csv')

inset_wls(ax0,[0.52,0.675,0.45,0.225],2000+df_ls.t_zsc.values,df_ls.mu_zsc.values*df[df.type=='all'].nmad_dh.values[0],2*df_ls.w_zsc.values*df[df.type=='all'].nmad_dh.values[0],beta2=[ddh_dt_glob,ddt_dt_glob_2std])
# inset_wls(ax0,[0.52,0.675,0.45,0.225],tmp_dt.mid_bin.values.astype('datetime64[D]'),tmp_dt.med_zsc.values*tmp_dt.nmad_dh.values,2*tmp_dt.nmad_zsc / np.sqrt(tmp_dt.ns_ics.values + tmp_dt.ns_ib.values * 1./40)*tmp_dt.nmad_dh.values,beta2=[ddh_dt_glob,ddt_dt_glob_2std])

inset_hist(ax0,[0.52,0.9,0.45,0.09],t,bin_t,'',normal=False)
cbaxes = ax0.inset_axes([0.725,0.085,0.2,0.02])

col_bounds = np.array([0, 0.5, 1])
cb = []
cb_val = np.linspace(0, 1, len(col_bounds))
for j in range(len(cb_val)):
    cb.append(mpl.cm.Purples(cb_val[j]))
cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(zip((col_bounds-min(col_bounds))/(max(col_bounds-min(col_bounds))),cb)), N=1000, gamma=1.0)
norm = mpl.colors.Normalize(vmin=0,vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = plt.colorbar(sm ,cax=cbaxes, ticks=[0,0.5,1], orientation='horizontal',extend='max',shrink=0.5)
# cb.ax.tick_params(labelsize=12)
cb.set_label('Standardized elevation uncertainty $\sigma_{z}$')

cbaxes = ax0.inset_axes([0.075,0.085,0.2,0.02])

col_bounds2 = np.array([-2, -1, 0, 1, 2])
cb = []
cb_val = np.linspace(0, 1, len(col_bounds2))
for j in range(len(cb_val)):
    cb.append(mpl.cm.RdYlBu(cb_val[j]))
cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cb2', list(zip((col_bounds2-min(col_bounds2))/(max(col_bounds2-min(col_bounds2))),cb)), N=1000, gamma=1.0)
norm = mpl.colors.Normalize(vmin=-2,vmax=2)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = plt.colorbar(sm ,cax=cbaxes, ticks=[-2,-1,0,1, 2], orientation='horizontal',extend='both',shrink=0.5)
# cb.ax.tick_params(labelsize=12)
cb.set_label('Elevation bias $\mu_{dh}$ (m)')

cbaxes = ax0.inset_axes([0.4,0.085,0.2,0.02])

col_bounds3 = np.array([0, 0.1, 0.2])
cb = []
cb_val = np.linspace(0, 1, len(col_bounds2))
for j in range(len(cb_val)):
    cb.append(mpl.cm.Oranges(cb_val[j]))
cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cb2', list(zip((col_bounds2-min(col_bounds2))/(max(col_bounds2-min(col_bounds2))),cb)), N=1000, gamma=1.0)
norm = mpl.colors.Normalize(vmin=0,vmax=0.2)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = plt.colorbar(sm ,cax=cbaxes, ticks=[0,0.1,0.2], orientation='horizontal',extend='max',shrink=0.5)
# cb.ax.tick_params(labelsize=12)
cb.set_label('Elevation change bias |$\\beta_{dh}$| (m yr$^{-1}$)')

legax = ax0.inset_axes([0.065,0.9,0.1,0.08])

legax.fill_between([0,1],[0,0],[1,1],color=plt.cm.Greys(0.7))
legax.fill_between([0,1],[1.5,1.5],[2.5,2.5],color=plt.cm.Greys(0.3))
# legax.text(1.5,0.5,'IceBridge points: '+str(df[df.type=='all'].ns_ib.values[0]),va='center',ha='left')
# legax.text(1.5,2,'ICESat points: '+str(df[df.type=='all'].ns_ics.values[0]),va='center',ha='left')
legax.text(1.5,0.5,'IceBridge density: 21,064,071 points',va='center',ha='left')
legax.text(1.5,2,'ICESat density: 3,840,188 points',va='center',ha='left')
legax.set_xlim((0,5))
legax.set_ylim((0,3))
legax.set_xticks([])
legax.set_yticks([])
legax.spines['top'].set_visible(False)
legax.spines['left'].set_visible(False)
legax.spines['right'].set_visible(False)
legax.spines['bottom'].set_visible(False)

plt.savefig('/home/atom/ongoing/work_worldwide/figures/revised/ED_Figure_4.png',dpi=400)
