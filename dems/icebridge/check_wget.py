"""
@author: hugonnet
check for wget errors in IODEM3 bulk download and write list of failed links for re-download if necessary
"""

import os
import pandas as pd
import csv

dl_dir = '/calcul/santo/hugonnet/icebridge/03_rgi60'
# input_csv = '/data/icesat/travail_en_cours/romain/ww_tvol_study/worldwide/global/IceBridge/list_icebridge_03_rgi60.csv'
input_csv = '/home/atom/proj/ww_tvol_study/worldwide/global/IceBridge/list_icebridge_03_rgi60.csv'
out_csv = '/calcul/santo/hugonnet/icebridge/03_missing.csv'

list_dl = os.listdir(dl_dir)

list_csv = [os.path.basename(f) for f in pd.read_csv(input_csv,header=None).values.tolist()]

list_missing = [f for f in list_dl if f not in list_csv]

if len(list_missing) > 1:
    with open(out_csv, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        for url in list_missing:
            writer.writerow([url])