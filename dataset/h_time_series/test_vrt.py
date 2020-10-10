from __future__ import print_function
from pyddem.fit_tools import reproj_build_vrt
from glob import glob
import os
import shutil

region_list = ['06_rgi60','11_rgi60','18_rgi60']
utms = ['27N','32N','59S']
out_dir = '/calcul/santo/hugonnet/worldwide/'
res=100
res_dir = '/data/icesat/travail_en_cours/romain/results/'

for region in region_list:

    region_out_dir = os.path.join(out_dir,region,'stacks')
    list_dh = glob(os.path.join(region_out_dir, '**/*_err.tif'),recursive=True)

    dh_dir = os.path.join(out_dir, region, 'err')
    if os.path.exists(dh_dir):
        shutil.rmtree(dh_dir)
    os.mkdir(dh_dir)

    for fn_dh in list_dh:
        shutil.copy(fn_dh, dh_dir)

    utm_vrt = utms[region_list.index(region)]

    new_list = [os.path.join(dh_dir,f) for f in os.listdir(dh_dir)]

    reproj_build_vrt(new_list, utm_vrt, os.path.join(dh_dir, region + '_' + str(res) + 'm.vrt'))

    region_res_dir= os.path.join(res_dir,region,'err')
    if os.path.exists(region_res_dir):
        shutil.rmtree(region_res_dir)
    shutil.copytree(dh_dir,region_res_dir)