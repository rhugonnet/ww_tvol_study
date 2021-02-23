"""
@author: hugonnet
transform all ILAKS1B LAS into DEM using the Ames Stereo Pipeline
"""

import os
import sys
from glob import glob
from xml.dom import minidom
from pybob.GeoImg import GeoImg

ilaks1b_dir = '/data/icesat/travail_en_cours/romain/data/dems/icebridge/ilaks1b/las'
output_dir = '/data/icesat/travail_en_cours/romain/data/dems/icebridge/ilaks1b/dem'
# ilaks1b_dir = '/home/atom/ongoing/work_worldwide/icebridge/5000000500383/54568742'
# output_dir = '/home/atom/ongoing/work_worldwide/icebridge/5000000500383/54568742'

# asp_path = '/home/atom/opt/asp-2.6.0/bin'
asp_path = '/home/h/hugonnet/asp_2.6.0/bin'

res=15

list_las = glob(os.path.join(ilaks1b_dir,'*.las'),recursive=True)

for las in list_las:

    print('Working on '+las)

    xmldoc = minidom.parse(las+'.xml')
    itemlist = xmldoc.getElementsByTagName('RangeBeginningDate')
    date = itemlist[0].childNodes[0].nodeValue

    fn_out = os.path.join(output_dir,'ILAKS1B_'+''.join(date.split('-'))+'_'+'_'.join(os.path.splitext(os.path.basename(las))[0].split('_')[2:]))


    if not os.path.exists(fn_out+'-DEM.tif'):
        os.system(os.path.join(asp_path,'point2dem')+' --dem-spacing '+str(res)+' -o '+fn_out+' '+las)
    else:
        img = GeoImg(fn_out + '-DEM.tif')
        #try again, something failed
        if img.npix_x < 5:
            print('Wrong output raster; reprocessing...')
            os.remove(fn_out + '-DEM.tif')
            os.system(os.path.join(asp_path, 'point2dem') + ' --dem-spacing ' + str(res) + ' -o ' + fn_out + ' ' + las)
        else:
            print('Already processed.')