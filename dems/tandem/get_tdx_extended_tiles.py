"""
@author: hugonnet
to retrieve ref DEM tiles (TDX): create extended tiles based on L1A width and latitude distorsion and calculates 1x1 tiles intersecting this new shapefile
"""

import csv, os
import pandas as pd
import numpy as np
from tiledivlib import stack_tile_polygon, area_intersect_geom_listpoly
from vectlib import extent_shp, read_geom_from_shp, union_cascaded_multipoly, polygon_list_to_multipoly, intersect_poly, geomcol_to_valid_multipoly, write_poly_to_shp, clip_shp_to_extent, convhull_of_geomcol
from misclib import SRTMGL1_naming_to_latlon, latlon_to_UTM

def write_csv(list_tiles,list_area,list_tot,out_csv):

    with open(out_csv, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Tile_name', 'UTM zone', 'Tot_area_intersecting [km2]', 'Tot_area_tile [km2]'])

        for i in range(len(list_tiles)):
            lat, lon = SRTMGL1_naming_to_latlon(list_tiles[i])
            _, utm_zone = latlon_to_UTM(lat+0.5,lon+0.5)

            writer.writerow([str(list_tiles[i]), utm_zone, str(list_area[i] / 1000000), str(list_tot[i] / 1000000)])


def main(in_csv,out_csv,inters_out,min_inters_area):

    print('Recovering tiles ID with intersecting area superior than ' + str(min_inters_area) + '...')

    # recovering list of intersecting tile with area superior than criteria
    df = pd.read_csv(in_csv)  # fetching tile list as Series object
    # tilelist = df['Tile_name']
    # chk = df['Tot_area_intersecting [km2]']
    # ind = chk[chk > min_inters_area].index  # sorting Series according to criterium
    # list_tiles = tilelist[ind].tolist()
    list_tiles = df['Tile_name']

    print('Calculating union of extended tiles...')

    # create stack of tiles polygon
    _, list_extended_poly = stack_tile_polygon(list_tiles, True) #HERE WE EXTEND WITH "TRUE"
    extended_multipoly = polygon_list_to_multipoly(list_extended_poly)
    extended_union = union_cascaded_multipoly(extended_multipoly)

    print('Writing intersection to shapefile...')

    write_poly_to_shp(extended_union,inters_out,os.path.splitext(os.path.basename(inters_out))[0], True)

    print('Calculating extent...')

    # calculate 'ConvexHull' extent
    geomcol = read_geom_from_shp(inters_out)
    extent_geom=convhull_of_geomcol(geomcol)

    print('Calculating intersection of shapefile extent with all tiles globally...')

    #stack polygon of all tiles globally
    list_all_tiles,list_poly = stack_tile_polygon(None, False)
    list_inters_extent,_,_=area_intersect_geom_listpoly(list_all_tiles,list_poly,extent_geom,'Polygon')

    print('Calculating intersection of tiles in extent with all geometries and deriving area...')

    #calculate intersection with full geometry only for tiles intersecting preceding extent
    _,list_poly_inters_extent=stack_tile_polygon(list_inters_extent,None)
    list_tile_inters,list_area_inters,list_tot_inters=area_intersect_geom_listpoly(list_inters_extent,list_poly_inters_extent,geomcol,'GeometryCollection')

    print('Writing to csv file: ' + out_csv)

    #write out to csv file
    write_csv(list_tile_inters,list_area_inters,list_tot_inters,out_csv)

    print('Done.')

if __name__=="__main__":
    #in_shp='/home/atom/proj/aster_tdem/data/RGI/00_rgi60/06_rgi60_Iceland/06_rgi60_Iceland.shp'
    rgi_naming_txt = '/home/atom/proj/aster_tdem/worldwide/rgi_neighb_merged_naming_convention.txt'
    main_dir = '/home/atom/proj/aster_tdem/worldwide/'

    text_file = open(rgi_naming_txt, 'r')
    rgi_list = text_file.readlines()

    for rgi_counter in np.arange(len(rgi_list)):
        rgi_region = rgi_list[rgi_counter]

        in_csv = os.path.join(main_dir,rgi_region[:-1].split('rgi60')[0]+'rgi60','list_glacierized_tiles_' + rgi_region[:-1].split('rgi60')[0] + 'rgi60' +'.csv')
        out_csv = os.path.join(main_dir,rgi_region[:-1].split('rgi60')[0]+'rgi60','list_ref_DEM_tiles_' + rgi_region[:-1].split('rgi60')[0] + 'rgi60' +'.csv')

        inters_out= os.path.join(main_dir,rgi_region[:-1].split('rgi60')[0]+'rgi60','ref_DEM_extended_'+rgi_region[:-1].split('rgi60')[0]+'rgi60')
        min_inters_area=0.

        main(in_csv, out_csv, inters_out, min_inters_area)
