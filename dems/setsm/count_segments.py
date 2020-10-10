"""
@author: hugonnet
count original segments from output summary files of pairwise co-registration for Table S1
"""
from __future__ import print_function
import pandas as pd
import os
from glob import glob

setsm_dir = '/data/icesat/travail_en_cours/romain/data/dems/arctic_dem'

list_summary_csv = glob(os.path.join(setsm_dir,'**/mosaic*.csv'),recursive=True)

nb_dem = 0
for summary_csv in list_summary_csv:

    df = pd.read_csv(summary_csv)
    nb_dem += len(df.index)

print(nb_dem)
