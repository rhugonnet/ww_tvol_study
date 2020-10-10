"""
@author: hugonnet
create shapefiles of union-cascaded tiles to download ASTER L1A data on EarthData Search
"""
import numpy as np
import os
import pandas as pd
from vectlib import write_poly_to_shp, polygon_list_to_multipoly, union_cascaded_multipoly
from tiledivlib import stack_tile_polygon

def main(in_tile_list,out_shp,min_inters_area):

    print('Recovering tiles ID with intersecting area superior than ' + str(min_inters_area) + '...')

    #recovering list of intersecting tile with area superior than criteria
    df=pd.read_csv(in_tile_list) #fetching tile list as Series object
    tilelist=df['Tile_name']
    chk=df['Tot_area_intersecting [km2]']
    ind=chk[chk>min_inters_area].index #sorting Series according to criterium
    list_tiles=tilelist[ind].tolist()

    #create stack of tiles polygon
    _,list_poly = stack_tile_polygon(list_tiles,True)

    #calculate multipolygon
    multipoly=polygon_list_to_multipoly(list_poly)

    #derive cascaded union
    cascpoly=union_cascaded_multipoly(multipoly)

    #write ESRI files to folder, zip folder
    write_poly_to_shp(cascpoly,out_shp,os.path.basename(out_shp),True)


if __name__ == '__main__':
    rgi_naming_txt = '/home/atom/proj/aster_tdem/worldwide/rgi_neighb_merged_naming_convention.txt'
    main_dir = '/home/atom/proj/aster_tdem/worldwide/'

    text_file = open(rgi_naming_txt, 'r')
    rgi_list = text_file.readlines()

    for rgi_counter in np.arange(len(rgi_list)):
        rgi_region = rgi_list[rgi_counter]

        in_csv = os.path.join(main_dir, rgi_region[:-1].split('rgi60')[0] + 'rgi60',
                              'list_glacierized_tiles_' + rgi_region[:-1].split('rgi60')[0] + 'rgi60' + '.csv')

        out_shp = os.path.join(main_dir, rgi_region[:-1].split('rgi60')[0] + 'rgi60',
                                  'L1A_glacierized_extended_' + rgi_region[:-1].split('rgi60')[0] + 'rgi60')
        min_glacier_area = 0.

        main(in_csv, out_shp, min_glacier_area)

