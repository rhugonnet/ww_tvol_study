"""
@author: hugonnet
postprocessing for bias-correction and co-registration of ASTER DEMs after MicMac stereo
"""
from __future__ import print_function
import os, sys, shutil
from subprocess import Popen
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

#to run corrections for an entire region
aster_dir = '/data/lidar2/RGI_paper/worldwide_out/17_rgi60/aster_dem'
#directory where reference DEM is located (by region by UTM zone and by tile), see "tandem" scripts for more details
ref_dir = '/data/lidar2/RGI_paper/worldwide/17_rgi60/ref'
#glacier mask
fn_excl_mask = '/data/lidar2/RGI_paper/worldwide/inventories/rgi60_merge.shp'
proc_aster_dir = '/tmp/'
proc_ref_dir = '/data/localdata/hugonnet/tandem/'
out_dir = '/data/lidar2/RGI_paper/worldwide_out/17_rgi60/aster_corr/'

utm_dir = [d for d in os.listdir(aster_dir)]

for utm in utm_dir:

    print('Working on UTM zone: ' +utm)

    ref_utm_dir = os.path.join(ref_dir,utm)
    aster_utm_dir = os.path.join(aster_dir,utm)

    tmp_aster_utm_dir = os.path.join(proc_aster_dir,utm)
    proc_ref_utm_dir = os.path.join(proc_ref_dir,utm)
    out_utm_dir = os.path.join(out_dir,utm)

    if not os.path.exists(proc_ref_utm_dir):
        shutil.copytree(ref_utm_dir,proc_ref_utm_dir)

    fn_ref_shp = os.path.join(proc_ref_utm_dir, 'ref.shp')

    print('Getting image footprint...')

    os.chdir(proc_ref_utm_dir)
    cmd = 'image_footprint.py *.tif -o ' + fn_ref_shp
    p = Popen(cmd,stderr=sys.stderr,stdout=sys.stdout,shell=True)
    p.wait()

    print('Bias correcting DEMs...')

    os.chdir(aster_utm_dir)
    cmd2 = 'bias_correct_tiles.py '+fn_ref_shp+' AST* -a '+fn_excl_mask+' -n 20 -t '+tmp_aster_utm_dir+' -o '+out_utm_dir+' -z'
    p = Popen(cmd2,stderr=sys.stderr,stdout=sys.stdout,shell=True)
    p.wait()

print('Fin.')



