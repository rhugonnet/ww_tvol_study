"""
@author: hugonnet
derive list 1x1 tiles with glaciers and their glacierized area from RGI shapefile
"""

import numpy as np
import csv, os
from tiledivlib import stack_tile_polygon, area_intersect_geom_listpoly
from vectlib import extent_shp, read_geom_from_shp, convhull_of_geomcol
from misclib import SRTMGL1_naming_to_latlon, latlon_to_UTM

def write_csv(list_tiles,list_area,list_tot,out_csv):

    with open(out_csv, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Tile_name', 'UTM zone', 'Tot_area_intersecting [km2]', 'Tot_area_tile [km2]'])

        for i in range(len(list_tiles)):
            lat, lon = SRTMGL1_naming_to_latlon(list_tiles[i])
            _, utm_zone = latlon_to_UTM(lat+0.5,lon+0.5)

            writer.writerow([str(list_tiles[i]), utm_zone, str(list_area[i] / 1000000), str(list_tot[i] / 1000000)])


def main(in_shp,out_csv):

    geomcol = read_geom_from_shp(in_shp)
    extent_geom = convhull_of_geomcol(geomcol)

    print('Calculating intersection of shapefile extent with all tiles globally...')

    #stack polygon of all tiles globally
    list_tiles,list_poly=stack_tile_polygon(None,False)
    list_inters_extent,_,_=area_intersect_geom_listpoly(list_tiles,list_poly,extent_geom,'Polygon')

    print('Calculating intersection of tiles in extent with all geometries and deriving area...')

    #calculate intersection with full geometry only for tiles intersecting preceding extent
    _,list_poly_inters_extent=stack_tile_polygon(list_inters_extent,False)
    list_tile_inters,list_area_inters,list_tot_inters=area_intersect_geom_listpoly(list_inters_extent,list_poly_inters_extent,geomcol,'GeometryCollection')

    print('Writing to csv file: ' + out_csv)

    #write out to csv file
    write_csv(list_tile_inters,list_area_inters,list_tot_inters,out_csv)

    print('Done.')

if __name__=="__main__":

    rgi_dir = '/home/atom/data/inventory_products/RGI/00_rgi60_neighb_merged/'
    rgi_naming_txt = '/home/atom/proj/aster_tdem/worldwide/rgi_neighb_merged_naming_convention.txt'
    main_dir = '/home/atom/proj/aster_tdem/worldwide/'

    text_file = open(rgi_naming_txt, 'r')
    rgi_list = text_file.readlines()

    for rgi_counter in np.arange(len(rgi_list)):
    # for rgi_counter in [0,11]:
        rgi_region = rgi_list[rgi_counter]

        in_rgi = os.path.join(rgi_dir,rgi_region[:-1],rgi_region[:-1] + '.shp')  # .shp RGI or glacier outlines file
        if not os.path.exists(os.path.join(main_dir,rgi_region[:-1].split('rgi60')[0]+'rgi60')):
            os.mkdir(os.path.join(main_dir,rgi_region[:-1].split('rgi60')[0]+'rgi60'))

        out_csv = os.path.join(main_dir,rgi_region[:-1].split('rgi60')[0]+'rgi60','list_glacierized_tiles_' + rgi_region[:-1].split('rgi60')[0] + 'rgi60' +'.csv')

        main(in_rgi,out_csv)
