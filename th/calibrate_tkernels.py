"""
@author: hugonnet
draw empirical temporal variograms with categories of slope, elevation change rate, etc... (see fit_tools)
"""

import pyddem.fit_tools as ft
import os
from glob import glob
import numpy as np
import gdal
from pybob.bob_tools import mkdir_p
from pyddem.vector_tools import latlon_to_UTM, SRTMGL1_naming_to_latlon

# list_regions = ['05_rgi60','06_rgi60','07_rgi60','08_rgi60','09_rgi60','11_rgi60','12_rgi60','17_rgi60','16_rgi60','18_rgi60','19_rgi60']
# list_tiles = [['N69W026','N60W044','N69W054','N82W045'],['N64W018'],['N77E015'],['N66E013'],['N74E056'],['N46E008'],['N43E042'],['S48W074','S34W070'],['S10W078'],['S44E170'],['S65W059']]
list_regions = ['17_rgi60','16_rgi60','18_rgi60','19_rgi60']
list_tiles = [['S48W074','S34W070'],['S10W078'],['S44E170'],['S65W059']]

world_data_dir = '/calcul/santo/hugonnet/worldwide/'
nproc=64
main_out_dir = '/data/icesat/travail_en_cours/romain/results/tvar/'
gla_mask = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/regions/rgi60_merge.shp'
inc_mask = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/buffered/rgi60_buff_10.shp'


for region in list_regions:

    dir_stack = os.path.join(world_data_dir, region, 'stacks')
    #
    # utm_dirs = os.listdir(dir_stack)
    #
    # for utm in utm_dirs:
    # tile = 'S48W074'

    for tile in list_tiles[list_regions.index(region)]:
        lat, lon = SRTMGL1_naming_to_latlon(tile)
        _, utm = latlon_to_UTM(lat,lon)

        ref_utm_dir = os.path.join(world_data_dir,region,'ref',utm)
        fn_stack = glob(os.path.join(dir_stack,utm, tile+'_final.nc'), recursive=True)[0]

        fn_orig = os.path.join(os.path.dirname(fn_stack),os.path.basename(fn_stack).split('_')[0]+'.nc')
        ref_vrt = os.path.join(ref_utm_dir, 'tmp_' + utm + '.vrt')
        ref_list = glob(os.path.join(ref_utm_dir, '**/*.tif'), recursive=True)
        if not os.path.exists(ref_vrt):
            gdal.BuildVRT(ref_vrt, ref_list, resampleAlg='bilinear')

        out_dir = os.path.join(main_out_dir,region)
        mkdir_p(out_dir)

        ft.manual_refine_sampl_temporal_vgm(fn_orig, ref_vrt, out_dir, filt_ref='both', time_filt_thresh=[-30, 5]
                                            ,ref_dem_date=np.datetime64('2015-01-01'), inc_mask=inc_mask, gla_mask=gla_mask, nproc=nproc)