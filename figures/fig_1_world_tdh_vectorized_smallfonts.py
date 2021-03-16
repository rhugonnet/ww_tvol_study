
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
from pyddem.vector_tools import SRTMGL1_naming_to_latlon
import pyddem.vector_tools as ot
from pybob.image_tools import create_mask_from_shapefile
from pybob.GeoImg import GeoImg
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import scipy.interpolate

# shp_dir = '/home/atom/data/inventory_products/RGI/00_rgi60/00_rgi60_regions/nice_regions'
# list_shp = [os.path.join(shp_dir,fn) for fn in os.listdir(shp_dir) if fn.endswith('.shp')]
# list_shp = sorted(list_shp,key=(lambda x: int(os.path.basename(x)[:-4])))
plt.rcParams.update({'font.size': 5})
plt.rcParams.update({'lines.linewidth':0.5})
plt.rcParams.update({'axes.linewidth':0.5})
# plt.rcParams.update({'ticks.linewidth':0.5})
plt.rcParams.update({'pdf.fonttype':42})


fs_moyen = 5.5
fs_max = 6

shp_buff = '/home/atom/data/inventory_products/RGI/00_rgi60/rgi60_buff_diss.shp'
fn_hs = '/home/atom/documents/paper/Hugonnet_2020/figures/world_robin_rs.tif'
fn_land = '/home/atom/data/inventory_products/NaturalEarth/ne_50m_land/ne_50m_land.shp'

fn_subreg = '/home/atom/ongoing/work_worldwide/vol/final/subreg_fig1.csv'
# fn_subreg = '/home/atom/ongoing/work_worldwide/vol/tile/subreg_multann.csv'

df = pd.read_csv(fn_subreg)

# ind_as=np.logical_and(df.reg==10,df.lon_reg==-174.5)
#
# df = df[~ind_as]

regions_shp='/home/atom/data/inventory_products/RGI/00_rgi60/00_rgi60_regions/regions_split_adaptedHMA.shp'

fig_width_inch=7.2
fig = plt.figure(figsize=(fig_width_inch,fig_width_inch/1.9716))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

ax.set_global()
ax.outline_patch.set_linewidth(0)

bounds = [-179.99,179.99,-89.99,89.99]

def poly_from_extent(ext):

    poly = np.array([(ext[0],ext[2]),(ext[1],ext[2]),(ext[1],ext[3]),(ext[0],ext[3]),(ext[0],ext[2])])

    return poly
polygon = poly_from_extent(bounds)

sub_ax = fig.add_axes([0.0875,0,0.825,0.825],
                      projection=ccrs.Robinson(), label='world')
sub_ax.set_extent([-179.99,179.99,-89.99,89.99], ccrs.Geodetic())

sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor='gainsboro'))
sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor='dimgrey'))

sub_ax.outline_patch.set_edgecolor('lightgrey')


def coordXform(orig_crs, target_crs, x, y):
    return target_crs.transform_points( orig_crs, x, y )

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

    return np.array(list(zip(all_lon_interp, all_lat_interp)))


def rect_units_to_verts(rect_u):

    return np.array([[rect_u[0],rect_u[1]],[rect_u[0]+rect_u[2],rect_u[1]],[rect_u[0]+rect_u[2],rect_u[1] +rect_u[3]],[rect_u[0],rect_u[1]+rect_u[3]],[rect_u[0],rect_u[1]]])

def cerc_units_to_verts(cerc_u):

    xy, rad = cerc_u

    theta = np.linspace(0, 2 * np.pi, 100)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    return verts * rad + xy


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
hs_tmp = hs_land.copy()
hs_tmp_nl = hs_notland.copy()
mask = out_of_poly_mask(img, polygon)

hs_tmp[~mask] = 0
hs_tmp_nl[~mask] = 0

color1 = mpl.colors.to_rgba('black')
color2 = mpl.colors.to_rgba('white')
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2', [color1, color2], 256)
cmap2._init()
cmap2._lut[0:1, -1] = 0.0  # We made transparent de 10 first levels of hillshade,
cmap2._lut[1:, -1] = 0.30

cmap22 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap22', [color1, color2], 256)
cmap22._init()
cmap22._lut[0:1, -1] = 0.0  # We made transparent de 10 first levels of hillshade,
cmap22._lut[1:, -1] = 0.15

# sub_ax.imshow(np.flip(hs_tmp[:, :], axis=0), extent=ext, transform=ccrs.Robinson(), cmap=cmap2, zorder=2)
# sub_ax.imshow(np.flip(hs_tmp_nl[:, :], axis=0), extent=ext, transform=ccrs.Robinson(), cmap=cmap22, zorder=2)

shape_feature = ShapelyFeature(Reader(regions_shp).geometries(), ccrs.PlateCarree(), edgecolor='None', alpha=0.1,
                               facecolor='indigo')
sub_ax.add_feature(shape_feature)

shape_feature = ShapelyFeature(Reader(shp_buff).geometries(), ccrs.PlateCarree(), edgecolor='None', alpha=1,
                               facecolor='indigo', linewidth=0)
sub_ax.add_feature(shape_feature)

shape_feature = ShapelyFeature(Reader(regions_shp).geometries(), ccrs.PlateCarree(), edgecolor='white', alpha=1,
                               facecolor='None', linewidth=0.35)
sub_ax.add_feature(shape_feature)


verts = mpath.Path(latlon_extent_to_robinson_axes_verts(polygon))
sub_ax.set_boundary(verts, transform=sub_ax.transAxes)


sub_ax = fig.add_axes([0.0875,0,0.825,0.825],
                      projection=ccrs.Robinson(), label='world2')
sub_ax.set_extent([-179.99,179.99,-89.99,89.99], ccrs.Geodetic())

sub_ax.imshow(hs_tmp[:, :], extent=ext, transform=ccrs.Robinson(), cmap=cmap2, zorder=2, interpolation='nearest',rasterized=True)
sub_ax.imshow(hs_tmp_nl[:, :], extent=ext, transform=ccrs.Robinson(), cmap=cmap22, zorder=2,interpolation='nearest',rasterized=True)
# sub_ax.set_rasterized(True)

sub_ax.outline_patch.set_edgecolor('lightgrey')

verts = mpath.Path(latlon_extent_to_robinson_axes_verts(polygon))
sub_ax.set_boundary(verts, transform=sub_ax.transAxes)


def line_main_to_inset(main_pos,inset_pos,inset_verts):

    center_x = main_pos[0] + main_pos[2]/2
    center_y = main_pos[1] + main_pos[3]/2

    left_x = center_x - inset_pos[2]/2
    left_y = center_y - inset_pos[3]/2


    #first, get the limits of the manually positionned exploded polygon in projection coordinates
    limits_robin = coordXform(ccrs.PlateCarree(), ccrs.Robinson(), np.array([-179.99, 179.99, 0, 0]),
                              np.array([0, 0, -89.99, 89.99]))

    ext_robin_x = limits_robin[1][0] - limits_robin[0][0]
    ext_robin_y = limits_robin[3][1] - limits_robin[2][1]


    unit_vert_x = (inset_verts[0] + limits_robin[1][0]) / ext_robin_x
    unit_vert_y = (inset_verts[1] + limits_robin[3][1]) / ext_robin_y

    # inset_mod_x = inset_verts[:,0] +  (inset_pos[0]-left_x)/inset_pos[2]
    # inset_mod_y = inset_verts[:,1] +  (inset_pos[1]-left_y)/inset_pos[3]

    #then, get the limits of the polygon in the manually positionned center map
    main_mod_x = (unit_vert_x*main_pos[2] - left_x + main_pos[0])/inset_pos[2]
    main_mod_y = (unit_vert_y*main_pos[3] + main_pos[1])/inset_pos[3]

      #
    # shade_ax.plot(main_mod_x*ext_robin_x - limits_robin[1][0],main_mod_y*ext_robin_y - limits_robin[3][1],color=col_contour,linestyle='dashed',linewidth=0.35.5,dashes=(3,1.5))
    # shade_ax.fill(chull_robin_x, chull_robin_y, transform=ccrs.Robinson(), color='lightgrey', alpha=0.2, zorder=1)
    # verts = mpath.Path(np.column_stack((chull_robin_x,chull_robin_y)))
    # shade_ax.set_boundary(verts, transform=shade_ax.transAxes)

    return main_mod_x*ext_robin_x - limits_robin[1][0], main_mod_y*ext_robin_y - limits_robin[3][1]


def color_ribbon(col,position,rw,rl,label,ticks=None):

    #draw rectangles
    nb_rect = len(col)

    if nb_rect % 2 == 0:
        vect_rob_x = [position[0] - nb_rect/2 * rw + i* rw for i in range(nb_rect+1)]
    else:
        vect_rob_x = [position[0] - rw/2 - nb_rect/2 * rw + i* rw for i in range(nb_rect+1)]

    rob_y = position[1] - rl

    sub_ax_rect = fig.add_axes([0,0,1,1],
                               projection=ccrs.Robinson(), label=label + '_rect')

    for i in range(nb_rect):
        sub_ax_rect.add_patch(
            mpatches.Rectangle((vect_rob_x[i], rob_y), rw, rl, linewidth=0.35, edgecolor=col[i], facecolor=col[i],
                               transform=ccrs.Robinson(),zorder=30))

    sub_ax_rect.add_patch(
            mpatches.Rectangle((vect_rob_x[0], rob_y), nb_rect*rw, rl, linewidth=0.35, edgecolor='black', facecolor='None',
                               transform=ccrs.Robinson(),zorder=30))

    # draw ticks

    if ticks is not None:
        tick_rat = 0.5
        if ticks == 'top':
            y_rect = rob_y+rl*tick_rat
        else:
            y_rect = rob_y

        for i in np.arange(0, nb_rect + 1, 4):
            sub_ax_rect.add_patch(mpatches.Arrow(vect_rob_x[i], y_rect, 0, rl * tick_rat, color='black', linewidth=0.35,
                                                 transform=ccrs.Robinson(), zorder=30))
            # nax.text(vect_rob_x[i], rob_y-rl*tick_rat, str(i), transform=ccrs.Robinson(), horizontalalignment='center',
            #                 verticalalignment='top', fontsize=10)


    sub_ax_rect.set_extent([-179.999,179.999,-89.999,89.999], ccrs.Geodetic())
    verts = mpath.Path(rect_units_to_verts([vect_rob_x[0], rob_y, nb_rect*rw, rl]))

    sub_ax_rect.set_boundary(verts, transform=sub_ax_rect.transAxes)

def world_ribbon(col,tdh,position,rw,rl):
    # draw rectangles
    nb_rect = len(col)

    if nb_rect % 2 == 0:
        vect_rob_x = [position[0] - nb_rect / 2 * rw + i * rw for i in range(nb_rect + 1)]
    else:
        vect_rob_x = [position[0] - rw / 2 - nb_rect / 2 * rw + i * rw for i in range(nb_rect + 1)]

    rob_y = position[1]

    sub_ax_rect = fig.add_axes([0, 0, 1, 1],
                               projection=ccrs.Robinson(), label='world')

    for i in range(nb_rect):
        vect = rl * np.abs(tdh[i])
        sub_ax_rect.add_patch(
            mpatches.Rectangle((vect_rob_x[i], rob_y-vect), rw, vect, linewidth=0.35, edgecolor='black', facecolor=col[i],
                               transform=ccrs.Robinson(), zorder=30))

    # sub_ax_rect.add_patch(
    #     mpatches.Rectangle((vect_rob_x[0], rob_y), nb_rect * rw, rl, linewidth=0.35, edgecolor='black', facecolor='None',
    #                        transform=ccrs.Robinson(), zorder=30))

    sub_ax_rect.add_patch(
        mpatches.FancyArrow(vect_rob_x[0], rob_y, 0, -5/4*rl*np.max(np.abs(tdh)),linewidth=0.35, head_width=125000,length_includes_head=True,color='black',
                       transform=ccrs.Robinson(), zorder=30))

    sub_ax_latlon = fig.add_axes([0, 0, 1, 1],
                               projection=ccrs.PlateCarree(), label='world2')
    sub_ax_latlon.text(-143.75,-72.5,'0',ha='right',va='center',transform=ccrs.PlateCarree(),color='black')
    sub_ax_latlon.text(-143.75,-77.5,'-0.25',ha='right',va='center',transform=ccrs.PlateCarree(),color='black')
    sub_ax_latlon.text(-143.75,-82.5,'-0.50',ha='right',va='center',transform=ccrs.PlateCarree(),color='black')
    sub_ax_latlon.text(-142.12,-89,'m yr$^{-1}$',ha='center',va='center',transform=ccrs.PlateCarree(),color='black')
    # sub_ax_latlon.annotate('', xy=(-142.12, -87), xytext=(-142.12, -84),
    #                      xycoords=ccrs.PlateCarree()._as_mpl_transform(sub_ax_latlon),
    #                      arrowprops=dict(facecolor='black', width=0.35,headwidth=1,headlength=2))

    verts = mpath.Path(rect_units_to_verts([-178, -88, 10, 10]))

    sub_ax_latlon.set_boundary(verts, transform=sub_ax_latlon.transAxes)

    # sub_ax_rect.plot([vect_rob_x[0], vect_rob_x[-1]], [rob_y-0.56*rl,rob_y-0.56*rl],transform=ccrs.Robinson(),color='black', linewidth=0.35, linestyle='dashed',zorder=30)

    # sub_ax_rect.add_patch(
    #     mpatches.Arrow(vect_rob_x[0], rob_y, rw*0.5, 0, color='black', linewidth=0.35,
    #                    transform=ccrs.Robinson(), zorder=30))
    sub_ax_rect.add_patch(
        mpatches.Arrow(vect_rob_x[0] - rw * 0.5, rob_y, rw * 0.5, 0, color='black', linewidth=0.35,
                       transform=ccrs.Robinson(), zorder=30))
    sub_ax_rect.add_patch(
        mpatches.Arrow(vect_rob_x[0]-rw*0.5, rob_y-rl*0.25, rw*0.5, 0, color='black', linewidth=0.35,
                       transform=ccrs.Robinson(), zorder=30))
    sub_ax_rect.add_patch(
        mpatches.Arrow(vect_rob_x[0]-rw*0.5, rob_y-rl*0.5, rw*0.5, 0, color='black', linewidth=0.35,
                       transform=ccrs.Robinson(), zorder=30))
    # sub_ax_rect.add_patch(
    #     mpatches.Arrow(vect_rob_x[0], rob_y-rl*0.75, rw*0.5, 0, color='black', linewidth=0.35,
    #                    transform=ccrs.Robinson(), zorder=30))

    sub_ax_rect.plot([vect_rob_x[0], vect_rob_x[0]], [rob_y, rob_y+2150000],transform=ccrs.Robinson(),color='black', linewidth=0.35, linestyle='dashed',zorder=30)
    sub_ax_rect.plot((vect_rob_x[-1], vect_rob_x[-1]),(rob_y,rob_y+2150000),transform=ccrs.Robinson(), color='black', linewidth=0.35, linestyle='dashed', zorder=30)

    sub_ax_rect.text(vect_rob_x[0] + 100000, rob_y + 1300000, '-0.36',
                     horizontalalignment='left', verticalalignment='bottom',
                     transform=ccrs.Robinson(), color='black', fontsize=fs_max,
                     fontweight='bold')
    sub_ax_rect.text(vect_rob_x[-1] - 100000, rob_y + 1300000, '-0.56',
                     horizontalalignment='right', verticalalignment='bottom',
                     transform=ccrs.Robinson(), color='black', fontsize=fs_max,
                     fontweight='bold')
    sub_ax_rect.text(position[0], rob_y + 800000, 'm yr$^{-1}$',
                     horizontalalignment='center', verticalalignment='bottom',
                     transform=ccrs.Robinson(), color='black', fontsize=fs_max,
                     fontweight='bold')
    sub_ax_rect.text(vect_rob_x[0], rob_y + 2200000, '2000',
                     horizontalalignment='center', verticalalignment='bottom',
                     transform=ccrs.Robinson(), color='black', fontsize=fs_max,
                     fontweight='bold',bbox= dict(boxstyle='round',facecolor='white', alpha=1,linewidth=0.35),zorder=30)
    sub_ax_rect.text(vect_rob_x[-1], rob_y + 2200000, '2020',
                     horizontalalignment='center', verticalalignment='bottom',
                     transform=ccrs.Robinson(), color='black', fontsize=fs_max,
                     fontweight='bold',bbox= dict(boxstyle='round',facecolor='white', alpha=1,linewidth=0.35),zorder=30)


    sub_ax_rect.set_extent([-179.999, 179.999, -89.999, 89.999], ccrs.Geodetic())
    verts = mpath.Path(rect_units_to_verts([vect_rob_x[0]-3*rw, rob_y - 1.5*rl * np.max(np.abs(tdh)), (nb_rect+3) * rw, 3.5*rl * np.max(np.abs(tdh))]))

    sub_ax_rect.set_boundary(verts, transform=sub_ax_rect.transAxes)


def add_reg_label(rect_center,text,label):

    sub_ax_lab = fig.add_axes([0,0,1,1],
                            projection=ccrs.Robinson(), label=label + '_label')

    height = 500000
    width = 200000 * len(text)

    # sub_ax_lab.add_patch(
    #     mpatches.Rectangle((rect_center[0]-width/2, rect_center[1]-height), width, height, linewidth=0.35, edgecolor='grey', facecolor='white',
    #                        transform=ccrs.Robinson()))

    verts = mpath.Path(rect_units_to_verts([rect_center[0]-width/2, rect_center[1]-height, width, height]))
    sub_ax_lab.set_extent([-179.999,179.999,-89.999,89.999], ccrs.Geodetic())

    sub_ax_lab.set_boundary(verts, transform=sub_ax_lab.transAxes)

    if text == 'Global':
        fs = fs_max
    else:
        fs = 4.5

    sub_ax_lab.text(rect_center[0], rect_center[1]-60000, text,
                  horizontalalignment='center', verticalalignment='top',
                  transform=ccrs.Robinson(), color='black', fontsize=fs, bbox= dict(facecolor='white', alpha=0.9,pad=1.5,linewidth=0.35),fontweight='bold')



def add_color_ribbon_circ(position,tdh,dmdt,valid_obs_py,label,perc_dm_tw,perc_area_tw,text,offset=(0,0),width_ribbon=100000,height_ribbon=300000):

    orig_pos = np.copy(position)
    position = position + np.array(offset)
    #color scale
    col_bounds = np.array([-1.2, -0.6, 0, 0.05, 0.1])
    col_bounds = np.array(col_bounds)
    # cmap = plt.get_cmap('RdYlBu')
    cb = []
    cb_val = np.linspace(0, 1, len(col_bounds))
    fun = scipy.interpolate.interp1d(col_bounds,cb_val,fill_value=(0,1),bounds_error=False)
    for j in range(len(tdh)):
        cb.append(mpl.cm.RdYlBu(fun(tdh[j])))

    #add circ
    lon, lat = position
    xy = coordXform(ccrs.PlateCarree(), ccrs.Robinson(), np.array([lon]), np.array([lat]))[0][0:2]


    rad = 1000+6000*np.sqrt(np.abs(dmdt))

    theta1_dm = 90
    theta2_dm = perc_dm_tw/100*360 +90

    theta1_area = 90
    theta2_area = perc_area_tw/100*360+90

    width_area = 100000

    # cb_cerc = mpl.cm.RdYlBu(fun(np.mean(tdh)))

    # do an independent ax for line
    sub_ax_line = fig.add_axes([0, 0, 1, 1],
                               projection=ccrs.Robinson(), label=label + '_line')
    xy_orig = coordXform(ccrs.PlateCarree(), ccrs.Robinson(), np.array([orig_pos[0]]), np.array([orig_pos[1]]))[0][0:2]

    pos_x, pos_y = line_main_to_inset([0.0875, 0, 0.825, 0.825], [0, 0, 1, 1], np.array([xy_orig[0], xy_orig[1]]))

    sub_ax_line.add_patch(
        mpatches.Arrow(xy[0], xy[1] - rad - width_area, pos_x - xy[0], pos_y - (xy[1] - rad - width_area),
                       color='black', linewidth=0.35,
                       transform=ccrs.Robinson(), zorder=30))

    sub_ax_line.set_extent([-179.999, 179.999, -89.999, 89.999], ccrs.Geodetic())
    verts = mpath.Path(
        rect_units_to_verts([xy[0], xy[1] - rad - width_area, pos_x - xy[0], pos_y - (xy[1] - rad - width_area)]))

    sub_ax_line.set_boundary(verts, transform=sub_ax_line.transAxes)

    #inside wedge: volume
    sub_ax_cerc = fig.add_axes([0,0,1,1],
                            projection=ccrs.Robinson(), label=label + '_cerc')

    # derive position of center of region polygon for downscaled Earth

    if perc_area_tw>0:
        sub_ax_cerc.add_patch(
            mpatches.Wedge(center=xy, r=rad, theta1=theta1_dm, theta2=theta2_dm ,facecolor=plt.cm.Blues(0.3), alpha=1, transform=ccrs.Robinson(), zorder=30,edgecolor=plt.cm.Greys(0.4),linewidth=0.35))
        sub_ax_cerc.add_patch(
            mpatches.Wedge(center=xy, r=rad, theta1=theta2_dm, theta2=theta1_dm, facecolor='whitesmoke', alpha=1,
                           transform=ccrs.Robinson(), zorder=30, edgecolor=plt.cm.Greys(0.4), linewidth=0.35))
        sub_ax_cerc.add_patch(
            mpatches.Wedge(center=xy, r=rad+width_area, theta1=theta1_area, theta2=theta2_area, width=width_area,facecolor=plt.cm.Blues(0.9), alpha=1,
                           transform=ccrs.Robinson(), zorder=30, edgecolor=plt.cm.Greys(0.4), linewidth=0.35))
        sub_ax_cerc.add_patch(
            mpatches.Wedge(center=xy, r=rad+width_area, theta1=theta2_area, theta2=theta1_area, width=width_area,facecolor='silver', alpha=1,
                           transform=ccrs.Robinson(), zorder=30, edgecolor=plt.cm.Greys(0.4), linewidth=0.35))
        sub_ax_cerc.add_patch(
            mpatches.Circle(xy=xy, radius=rad+width_area, facecolor='None', alpha=1,
                           transform=ccrs.Robinson(), zorder=30, edgecolor=plt.cm.Greys(0.8), linewidth=0.35))
        sub_ax_cerc.add_patch(
            mpatches.Circle(xy=xy, radius=rad, facecolor='None', alpha=1,
                           transform=ccrs.Robinson(), zorder=30, edgecolor=plt.cm.Greys(0.8), linewidth=0.35))
    else:
        sub_ax_cerc.add_patch(
            mpatches.Circle(xy=xy, radius=rad+width_area, facecolor='silver', alpha=1,
                           transform=ccrs.Robinson(), zorder=30, edgecolor=plt.cm.Greys(0.8), linewidth=0.35))
        sub_ax_cerc.add_patch(
            mpatches.Circle(xy=xy, radius=rad, facecolor='whitesmoke', alpha=1,
                           transform=ccrs.Robinson(), zorder=30, edgecolor=plt.cm.Greys(0.8), linewidth=0.35))

    gt_loss = np.abs(dmdt)/1000
    if gt_loss > 3.:
        if text == 'Global':
            fs=fs_max
        else:
            fs=fs_moyen
        print('{:,.1f}'.format(-gt_loss) + ' Gt')
        sub_ax_cerc.text(xy[0], xy[1], '{:,.1f}'.format(-gt_loss),
                         horizontalalignment='center', verticalalignment='center',
                         transform=ccrs.Robinson(), color=plt.cm.Blues(0.99), fontsize=fs,
                         fontweight='bold',zorder=30)
        if text == 'Global':
            sub_ax_cerc.text(xy[0]+10000,xy[1]-500000,'Gt yr$^{-1}$',horizontalalignment='center', verticalalignment='center',
                         transform=ccrs.Robinson(), color=plt.cm.Blues(0.99), fontsize=fs,
                         fontweight='bold',zorder=30)

    sub_ax_cerc.set_extent([-179.999,179.999,-89.999,89.999], ccrs.Geodetic())

    verts = mpath.Path(cerc_units_to_verts([xy, rad+width_area]))
    sub_ax_cerc.set_boundary(verts, transform=sub_ax_cerc.transAxes)

    cb2 = [plt.cm.Blues(1-val) for val in valid_obs_py]

    #add color ribbons + region label
    if text != 'Global':
        color_ribbon(cb,(xy[0],xy[1]-rad-width_area-height_ribbon/2),width_ribbon,height_ribbon,label=label,ticks='bottom')
        color_ribbon(cb2, (xy[0], xy[1] - rad - width_area), width_ribbon, height_ribbon / 2, label=label + '_valid',
                     ticks=None)
        add_reg_label((xy[0], xy[1] - rad - width_area - height_ribbon * 3 / 2), text=text, label=label)

    else:
        world_ribbon(cb,tdh,(xy[0],xy[1]-rad-width_area-height_ribbon/5),width_ribbon,height_ribbon)
        color_ribbon(cb2, (xy[0], xy[1] - rad - width_area), width_ribbon, height_ribbon / 5, label=label + '_valid',
                     ticks=True)
        add_reg_label((xy[0], xy[1]+rad/2), text=text, label=label)



tlim=[np.datetime64(str(2000+i)+'-01-01') for i in range(21)]
list_reg = sorted(list(set(list(df.subreg))))

#              Alaska,  Antarc,  CA N,    CA S,   Asia C,   Asia N,  Asia W,  Asia E,  Caucas,  Europe,  Green E,  Green N, Green W, Ice,  Low lat, New Zeal,      Russ E,    Russ W,    Scand,     SA N,    SA S,  Svalb,   USA,  West CA
list_offsets = [(-6,-1),(7,22),(-22,-10.0),(8,-36.5),(40,-27),(30,-28),(8,-32),(-7,-40),(-7,-28),(-3,-28),(-8,-52),(5,-5.1),(-25,5.55),(0,-35),(-20,-15),(-20,-15),(65.5,-24.8),(65,-5.4),(41,4.85),(40,0),(-15,3),(5,-2.25),(19,-30),(-17.5,-39)]
list_text = ['Alaska (01)','Antarctic and\nSubantarctic (19)','Arctic Canada\nNorth (03)','Arctic Canada\nSouth (04)','Central Asia (13)', 'North Asia (10)', 'South Asia\nEast (15)','South Asia\nWest (14)','Caucasus and\nMiddle East (12)','Central\nEurope (11)','Greenland\nPeri., E (05 E)','Greenland\nPeri., N (05 N)','Greenland\nPeri., W (05 W)','Iceland (06)','Low Latitudes (16)','New Zealand (18)','Russian Arctic,\nE (09 E)','Russian Arctic,\nW (09 W)', 'Scandinavia (08)','Southern Andes,\n N (17 N)','Southern Andes,\nS (17 S)','Svalbard and\nJan Mayen (07)','USA (02 S)','Western\nCanada (02 N)']
for reg in list_reg:
    df_reg = df[df.subreg == reg]

    yearly_dh = []
    yearly_dh_tw = []
    valid_obs_py = []
    for i in range(len(tlim)-1):
        period = str(tlim[i])+'_'+str(tlim[i+1])
        df_period = df_reg[df_reg.period==period]
        if len(df_period[df_period.category=='all'])>0:
            yearly_dh.append(df_period[df_period.category=='all'].dhdt.values[0])
        else:
            yearly_dh.append(np.nan)
        if len(df_period[df_period.category == 'tw']) > 0:
            yearly_dh_tw.append(df_period[df_period.category=='tw'].dhdt.values[0])
        else:
            yearly_dh_tw.append(np.nan)

        valid_obs_py.append(df_period[df_period.category=='all'].valid_obs_py.values[0])

    if len(df_reg[df_reg.category=='tw'])>0:
        perc_area_tw = df_reg[df_reg.category=='tw'].area.values[-1]/df_reg[df_reg.category=='all'].area.values[-1]*100.
        perc_dm_tw = df_reg[np.logical_and(df_reg.category=='tw',df_reg.period=='2000-01-01_2020-01-01')].dmdt.values[-1]/df_reg[np.logical_and(df_reg.category=='all',df_reg.period=='2000-01-01_2020-01-01')].dmdt.values[-1]*100.
    else:
        perc_area_tw = 0.
        perc_dm_tw = 0.


    add_color_ribbon_circ(np.array([df_reg.lon_reg.values[-1],df_reg.lat_reg.values[-1]]),yearly_dh,df_reg[np.logical_and(df_reg.category=='all',df_reg.period=='2000-01-01_2020-01-01')].dmdt.values[-1]*1000,valid_obs_py,label=reg,perc_dm_tw=perc_dm_tw,perc_area_tw=perc_area_tw,text=list_text[list_reg.index(reg)],offset=list_offsets[list_reg.index(reg)],height_ribbon=350000)

df['area_valid_obs_py'] = df['valid_obs_py'] * df['area']
df_world = df.groupby(['period','category'])['dvoldt','dmdt','tarea','area','area_valid_obs_py'].sum()
df_world['valid_obs_py'] = df_world['area_valid_obs_py'] / df_world['area']
df_world['dhdt'] = df_world['dvoldt'] / df_world['tarea']
df_world['period'] = df_world.index.get_level_values(0)
df_world['category'] = df_world.index.get_level_values(1)

yearly_dh_world = []
yearly_dh_world_tw = []
valid_obs_py_world = []
for i in range(len(tlim)-1):
    period = str(tlim[i]) + '_' + str(tlim[i + 1])
    df_period = df_world[df_world.period==period]
    if len(df_period[df_period.category=='all'])>0:
        yearly_dh_world.append(df_period[df_period.category=='all'].dhdt.values[0])
    else:
        yearly_dh_world.append(np.nan)
    if len(df_period[df_period.category == 'tw']) > 0:
        yearly_dh_world_tw.append(df_period[df_period.category=='ntw'].dhdt.values[0])
    else:
        yearly_dh_world_tw.append(np.nan)

    valid_obs_py_world.append(df_period[df_period.category == 'all'].valid_obs_py.values[0])

if len(df_world[df_world.category=='tw'])>0:
    perc_area_tw = df_world[df_world.category=='tw'].area.values[-1]/df_world[df_world.category=='all'].area.values[-1]*100.
    perc_dm_tw = df_world[np.logical_and(df_world.category=='tw',df_world.period=='2000-01-01_2020-01-01')].dmdt.values[-1]/df_world[np.logical_and(df_world.category=='all',df_world.period=='2000-01-01_2020-01-01')].dmdt.values[-1]*100.
else:
    perc_area_tw = 0.
    perc_dm_tw = 0.

add_color_ribbon_circ(np.array([-126,-30]),yearly_dh_world,df_world[np.logical_and(df_world.category=='all',df_world.period=='2000-01-01_2020-01-01')].dmdt.values[-1]*1000,valid_obs_py_world,label='world',perc_dm_tw=perc_dm_tw,perc_area_tw=perc_area_tw,text='Global',width_ribbon=200000,height_ribbon=2000000)
# ax.set_rasterized(True)

plt.savefig('/home/atom/ongoing/work_worldwide/figures/final/Figure_1_main_type42.pdf',dpi=400,transparent=True)


#LEGEND
fig_width_inch=7.2
fig = plt.figure(figsize=(fig_width_inch,fig_width_inch/1.9716))

out_png = '/home/atom/ongoing/work_worldwide/figures/final/Figure_1_legend_type42.pdf'
axleg2 = fig.add_axes([0,0,1,1],projection=ccrs.Robinson(),label='legend2')
axleg2.outline_patch.set_linewidth(0)

col_bounds = np.array([-1.2, -0.6, 0, 0.05, 0.1])
col_bounds = np.array(col_bounds)
# cmap = plt.get_cmap('RdYlBu')
cb = []
cb_val = np.linspace(0, 1, len(col_bounds))
for j in range(len(cb_val)):
    cb.append(mpl.cm.RdYlBu(cb_val[j]))
cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(zip((col_bounds-min(col_bounds))/(max(col_bounds-min(col_bounds))),cb)), N=1000, gamma=1.0)
norm = mpl.colors.Normalize(vmin=-1.2,vmax=0.1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = plt.colorbar(sm ,ax=axleg2, ticks=[-1.2,-1,-0.8,-0.6,-0.4,-0.2,0,0.1], orientation='horizontal',extend='both',shrink=0.35)
cb.ax.tick_params(labelsize=5, width=0.5,length=2)

# cb.ax.tick_params(labelsize=12)
cb.set_label('Mean elevation change rate (m yr$^{-1}$)')


axleg = fig.add_axes([0.22,-0.36,1,1],projection=ccrs.Robinson(),label='legend')
axleg.set_extent([-179.99,179.99,-89.99,89.99], ccrs.Geodetic())
axleg.outline_patch.set_linewidth(0)

u=0
rad_tot = 0
for a in [-0.5, -5, -50]:
    rad = (1000+np.sqrt(1000*np.abs(a))*6000)
    if a != -50:
        axleg.add_patch(mpatches.Circle(xy=[-700000+rad_tot+rad+u*800000,0],radius=rad,edgecolor='k',label=str(a)+' km$^2$', transform = ccrs.Robinson(),fill=False, zorder=30,linewidth=0.35))
        axleg.text(-700000 + rad_tot + rad + u * 800000, -rad - 100000, str(a), transform=ccrs.Robinson(),
                   horizontalalignment='center',
                   verticalalignment='top')
    else:
        rad_tot += 250000
        axleg.add_patch(mpatches.Wedge(center=[-700000+rad_tot+rad+u*800000,0],r=rad,theta1=90,theta2=270,facecolor=plt.cm.Blues(0.3),edgecolor=plt.cm.Greys(0.4),label=str(a)+' km$^2$', transform = ccrs.Robinson(), zorder=30,linewidth=0.35))
        axleg.add_patch(mpatches.Wedge(center=[-700000+rad_tot+rad+u*800000,0],r=rad,theta1=270,theta2=90,facecolor='whitesmoke',edgecolor=plt.cm.Greys(0.4),label=str(a)+' km$^2$', transform = ccrs.Robinson(), zorder=30,linewidth=0.35))
        axleg.add_patch(mpatches.Wedge(center=[-700000+rad_tot+rad+u*800000,0],r=rad+100000,theta1=90,theta2=270,width=100000,facecolor=plt.cm.Blues(0.9),edgecolor=plt.cm.Greys(0.4),label=str(a)+' km$^2$', transform = ccrs.Robinson(), zorder=30,linewidth=0.35))
        axleg.add_patch(mpatches.Wedge(center=[-700000+rad_tot+rad+u*800000,0],r=rad+100000,theta1=270,theta2=90,width=100000,facecolor='silver',edgecolor=plt.cm.Greys(0.4),label=str(a)+' km$^2$', transform = ccrs.Robinson(), zorder=30,linewidth=0.35))
        axleg.add_patch(mpatches.Circle(xy=[-700000+rad_tot+rad+u*800000,0],radius=rad,edgecolor=plt.cm.Greys(0.8),label=str(a)+' km$^2$', transform = ccrs.Robinson(),fill=False, zorder=30,linewidth=0.35))
        axleg.add_patch(mpatches.Circle(xy=[-700000+rad_tot+rad+u*800000,0],radius=rad+100000,edgecolor=plt.cm.Greys(0.8),label=str(a)+' km$^2$', transform = ccrs.Robinson(),fill=False, zorder=30,linewidth=0.35))
        axleg.text(-700000 + rad_tot + rad + u * 800000, -rad - 100000 - 100000, str(a), transform=ccrs.Robinson(),
                   horizontalalignment='center',
                   verticalalignment='top')

    u=u+1
    rad_tot += rad

axleg.text(5, -14, 'Mass change rate\nfor 2000-2019 (Gt yr$^{-1}$)', transform=ccrs.Geodetic(),horizontalalignment='center',verticalalignment='center')
axleg.text(60,-13.5,'Surface ratio',transform=ccrs.Geodetic(),horizontalalignment='center',verticalalignment='center')
axleg.add_patch(mpatches.Arrow(4600000, -1400000,-1600000,0,transform=ccrs.Robinson(),edgecolor='black',linewidth=0.35,zorder=30))
axleg.text(60,-7,'Mass change\nratio',transform=ccrs.Geodetic(),horizontalalignment='center',verticalalignment='center')
axleg.add_patch(mpatches.Arrow(4600000,-700000,-1600000,0,transform=ccrs.Robinson(),edgecolor='black',linewidth=0.35,zorder=30))
axleg.text(60,10,'Land-\nterminating',color='dimgrey',transform=ccrs.Geodetic(),horizontalalignment='center',verticalalignment='center')
axleg.add_patch(mpatches.Arrow(4600000,1000000,-500000,-100000,transform=ccrs.Robinson(),edgecolor='black',linewidth=0.35,zorder=30))
axleg.add_patch(mpatches.Arrow(4600000,1000000,-1000000,-500000,transform=ccrs.Robinson(),edgecolor='black',linewidth=0.35,zorder=30))
axleg.text(60,2.5,'Marine-\nterminating',color=plt.cm.Blues(0.9),transform=ccrs.Geodetic(),horizontalalignment='center',verticalalignment='center')
axleg.add_patch(mpatches.Arrow(4600000,250000,-2400000,-200000,transform=ccrs.Robinson(),edgecolor='black',linewidth=0.35,zorder=30))
axleg.add_patch(mpatches.Arrow(4600000,250000,-2850000,-700000,transform=ccrs.Robinson(),edgecolor='black',linewidth=0.35,zorder=30))

verts = mpath.Path(rect_units_to_verts([-8000000,-4000000,16000000,8000000]))
axleg.set_boundary(verts, transform=axleg.transAxes)


axleg3 = fig.add_axes([-0.27,-0.35,1,1],projection=ccrs.Robinson(),label='legend')
axleg3.set_extent([-179.99,179.99,-89.99,89.99], ccrs.Geodetic())
axleg3.outline_patch.set_linewidth(0)
col = [plt.cm.Blues(1-v) for v in np.linspace(0,1,20)]
position = (0,-500000)
rw = 225000
rl = 400000
label = 'legend3'
# draw rectangles
nb_rect = len(col)

if nb_rect % 2 == 0:
    vect_rob_x = [position[0] - nb_rect / 2 * rw + i * rw for i in range(nb_rect + 1)]
else:
    vect_rob_x = [position[0] - rw / 2 - nb_rect / 2 * rw + i * rw for i in range(nb_rect + 1)]

rob_y = position[1] -rl/2

for i in range(nb_rect):
    axleg3.add_patch(
        mpatches.Rectangle((vect_rob_x[i], rob_y), rw, rl/2, linewidth=0.35, edgecolor=col[i], facecolor=col[i],
                           transform=ccrs.Robinson(), zorder=30))

axleg3.add_patch(
    mpatches.Rectangle((vect_rob_x[0], rob_y), nb_rect * rw, rl/2, linewidth=0.35, edgecolor='black', facecolor='None',
                       transform=ccrs.Robinson(), zorder=30))


rob_y = position[1] - 3/2*rl

col = ['None']*20

for i in range(nb_rect):
    axleg3.add_patch(
        mpatches.Rectangle((vect_rob_x[i], rob_y), rw, rl, linewidth=0.35, edgecolor='None', facecolor=col[i],
                           transform=ccrs.Robinson(), zorder=30))

axleg3.add_patch(
    mpatches.Rectangle((vect_rob_x[0], rob_y), nb_rect * rw, rl, linewidth=0.35, edgecolor='black', facecolor='None',
                       transform=ccrs.Robinson(), zorder=30))

# draw ticks
tick_rat = 0.5

for i in np.arange(0, nb_rect + 1, 4):
    axleg3.add_patch(mpatches.Arrow(vect_rob_x[i], rob_y, 0, rl * tick_rat, color='black', linewidth=0.35,
                                         transform=ccrs.Robinson(), zorder=30))
    if i % 8 ==0:
        y_low = 100000
    else:
        y_low = 350000
    axleg3.text(vect_rob_x[i], rob_y-y_low, str(2000+i), transform=ccrs.Robinson(), horizontalalignment='center',
                    verticalalignment='top')

axleg3.text(position[0],rob_y-800000,'Annual time series',horizontalalignment='center',verticalalignment='top')
axleg3.text(position[0],position[1]+200000,'Percentage of area observed\nat least once a year',horizontalalignment='center',verticalalignment='bottom')
axleg3.text(vect_rob_x[0],position[1]+50000,'0%',horizontalalignment='center',verticalalignment='bottom')
axleg3.text(vect_rob_x[-1],position[1]+50000,'100%',horizontalalignment='center',verticalalignment='bottom')

verts = mpath.Path(rect_units_to_verts([vect_rob_x[0], rob_y, nb_rect*rw, 3*rl]))
axleg3.set_boundary(verts, transform=axleg3.transAxes)

axleg5 = fig.add_axes([-0.40,-0.365,1,1],projection=ccrs.Robinson(),label='legend5')
axleg5.set_extent([-179.99,179.99,-89.99,89.99], ccrs.Geodetic())
axleg5.outline_patch.set_linewidth(0)
bounds = [-10,10,-10,10]
polygon = poly_from_extent(bounds)
verts = mpath.Path(latlon_extent_to_robinson_axes_verts(polygon))
axleg5.set_boundary(verts, transform=axleg5.transAxes)

axleg5.add_patch(mpatches.Rectangle((-250000,-250000), 500000, 500000, edgecolor='black',
                                linewidth=0.35,
                                transform=ccrs.Robinson(), facecolor='indigo', alpha=1))

axleg5.text(0,-1000000,'Glacier contours\n(shaded: regions)',horizontalalignment='center',verticalalignment='center')

plt.savefig(out_png,dpi=400,transparent=True)


# cb_val = np.linspace(0, 1, len(col_bounds))
# for j in range(len(tdh)):
#     ind = np.argmin(np.abs(col_bounds - tdh[j]))
#     cb.append(mpl.cm.RdYlBu(cb_val[ind]))
#
# cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(
#     zip((date_bounds - min(date_bounds)) / (max(date_bounds - min(date_bounds))), cb)), N=20, gamma=1.0)
#
#
# ax1 = fig.add_axes([0,0.7,1,0.025],
#                       projection=ccrs.Robinson(), label='test1',aspect=0.2)
#
# cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, orientation='horizontal',ticks=np.arange(0,21,4))

# ax2 = fig.add_axes([0,0,1,1],projection=ccrs.Robinson(), label='test2')

# sub_ax.add_patch(mpatches.Circle(xy=[0,0], radius=1000000, color='red', alpha=1, transform=ccrs.Robinson(), zorder=30))

# plt.figure()
#
# sub_ax.imshow([[0., 1.], [0., 1.]],
#        cmap=cmap,interpolation='bicubic',extent=[0,1000000,0,100000],transform=ccrs.Robinson())
#
