"""
@author: hugonnet
download, extract, reproject and pairwise-coregister all REMA 8m strips intersecting glaciers
"""
from __future__ import print_function
import os, sys, shutil
from subprocess import Popen
sys.path.append('/home/echos/hugonnet/code/devel/rh_pygeotools')
import pandas as pd
import multiprocessing as mp
from demstripproducts import REMA_strip_r1_1
from shlib import merged_stderr_stdout, stdout_redirected
import matplotlib
import time
import traceback
matplotlib.use('Agg')

os.environ["OMP_NUM_THREADS"] = "8" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "8" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "8" # export NUMEXPR_NUM_THREADS=6

main_dir = '/data/icesat/travail_en_cours/romain/ww_tvol_study/worldwide'
tmp_dir = '/calcul/santo/hugonnet/setsm/'
final_dir = '/data/icesat/travail_en_cours/romain/data/dems/rema_8m/'
arcdem_directory_listing = '/data/icesat/travail_en_cours/romain/ww_tvol_study/worldwide/global/SETSM_directory_listing/REMA_v1-1_directory_listing_8m.csv'
rgi_naming_txt = '/data/icesat/travail_en_cours/romain/ww_tvol_study/worldwide/rgi_neighb_merged_naming_convention.txt'
nb_tasks = 8

# read csv list of tiles with more than 0km2 of ice for each RGI region
text_file = open(rgi_naming_txt, 'r')
rgi_list = text_file.readlines()
tile_list_csv = [os.path.join(main_dir, rgi[:-1].split('rgi60')[0] + 'rgi60', 'cov', 'list_glacierized_tiles_' + rgi[:-1].split('rgi60')[0] + 'rgi60' + '.csv') for rgi in rgi_list]

tiles_per_rgi = []
for list_csv in tile_list_csv:
    df = pd.read_csv(list_csv)
    tiles = df['Tile_name'].tolist()
    tiles_per_rgi.append(tiles)

all_tiles = []
for tiles in tiles_per_rgi:
    all_tiles = all_tiles + tiles

#list of REMA tiles: directory listing
with open(arcdem_directory_listing) as f:
    list_arcdem_dir= f.readlines()

list_arc_tiles = [os.path.split(arcdem_dir)[-1][:-1].upper() for arcdem_dir in list_arcdem_dir]

list_common_tiles=[]
#loop over all tiles of RGI regions
for tile in all_tiles:
    #if ArcticDEM tile is common with tile covering glacier
    if tile in list_arc_tiles:
        list_common_tiles.append(tile)

list_common_tiles = list(set(list_common_tiles))  # remove duplicates

print('Found '+str(len(list_common_tiles))+' REMA r1.1 8m tiles intersecting glaciers.')

list_mosaic_log = [os.path.join(tmp_dir,mosaic_log) for mosaic_log in os.listdir(tmp_dir) if 'mosaic' in mosaic_log and mosaic_log.endswith('log')]

list_remaining_tiles = list_common_tiles.copy()
list_tiles_to_move = []
for mosaic_log in list_mosaic_log:
    log = open(mosaic_log,'r')
    lines = log.readlines()
    log.close()

    if 'Fin' in lines[-1]:
        tile_name = os.path.splitext(os.path.basename(mosaic_log))[0].split('_')[1]
        list_remaining_tiles.remove(tile_name.upper())
        list_tiles_to_move.append(tile_name.lower())

for tile_name in list_tiles_to_move:
    processed_dir = os.path.join(tmp_dir,'8m','processed_'+tile_name)
    if os.path.exists(processed_dir):
        tgt_dir = os.path.join(final_dir,'processed_'+tile_name)
        if os.path.exists(tgt_dir):
            shutil.rmtree(tgt_dir)
        shutil.copytree(processed_dir,tgt_dir)

print('Found '+str(len(list_remaining_tiles))+' REMA r1.1 8m tiles not yet processed.')

def get_process_tile(tmp_dir,tile):

    def check_wget(log_file):

        chk=0
        with open(log_file) as s:
            text = s.readlines()
            for line in text:
                # if 'error' in line or 'Error' in line or 'ERROR' in line:
                if ('ERROR 404' not in line and 'robots.txt' not in line) and (
                        'error' in line or 'Error' in line or 'ERROR' in line or 'fail' in line or 'Fail' in line or 'FAIL' in line):
                    print(text.index(line))
                    print(line)
                    chk=1

        return chk==0

    #log files
    wget_log_file = os.path.join(tmp_dir,'wget_'+tile+'.log')
    mosaic_log_file = os.path.join(tmp_dir,'mosaic_'+tile+'.log')
    wget_fail = os.path.join(tmp_dir,'wget_fails.log')

    #get REMA tile naming
    arc_tile_name = tile.lower()
    print('Downloading '+arc_tile_name+'... writing to: '+wget_log_file)
    check = True
    u=0
    while check and u<5:
        u=u+1
        #if download already failed, sleep 10min before trying again
        if u>1:
            time.sleep(600)
        #download strip data
        cmd = 'wget -r -N -nH -np -R index.html* --cut-dirs=6 http://data.pgc.umn.edu/elev/dem/setsm/REMA/geocell/v1.0/8m/'+arc_tile_name+'/ -P '+tmp_dir
        log = open(wget_log_file,'w')
        p = Popen(cmd,stdout=log, stderr=log,shell=True)
        p.wait()
        log.close()
        check = not check_wget(wget_log_file)

    #if download keeps failing for some reason, abandon and append to specific log of failed downloads
    if u==5:
        print('Downloading for '+arc_tile_name+' failed.')
        with open(wget_fail,'a') as fail:
            fail.write(arc_tile_name+'\n')
    #otherwise, process
    else:
        print('Processing '+arc_tile_name+'... writing to: '+mosaic_log_file)
        tmp_dir_out = os.path.join(tmp_dir,'8m','processed_'+arc_tile_name)
        if not os.path.exists(tmp_dir_out):
            os.mkdir(tmp_dir_out)
        else:
            shutil.rmtree(tmp_dir_out)
            os.mkdir(tmp_dir_out)

        with open(mosaic_log_file,'w') as mos:
            with stdout_redirected(to=mos), merged_stderr_stdout():
                try:
                    REMA_strip_r1_1(os.path.join(tmp_dir,'8m'),arc_tile_name,tmp_dir_out,mosaic_coreg_segm=True,tgt_EPSG=None,tgt_res=[30,-30],nodata_out=-9999,interp_method='bilinear',geoid=False,rm_tar=True,downsample=False)
                except Exception:
                    print(traceback.format_exc())

def batch_wrapper(arg_dict):
    return get_process_tile(**arg_dict)

pool = mp.Pool(nb_tasks)

arg_dict = {'tmp_dir': tmp_dir}
u_args = [{'tile': t} for t in list_remaining_tiles]
for t in u_args:
    t.update(arg_dict)

pool.map(batch_wrapper,u_args)
pool.close()

#final sorting
# for rgi in rgi_list:
#     if tile in tiles_per_rgi[rgi_list.index(rgi)]:
#         dir_out = os.path.join(main_dir,(str(rgi)).zfill(2) + '_rgi60','tile_process',tile,'corr_dem_arcDEM')
#         shutil.copytree(tmp_dir_out,dir_out)
#
# shutil.rmtree(tmp_dir_out)