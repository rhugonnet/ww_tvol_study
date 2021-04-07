import os
from pyddem.tdem_tools import aggregate_int_to_shp
import numpy as np
import pandas as pd

# examples to integrate per-glacier estimates over regional polygons of a shapefile

nproc=64

# for HiMAP (only need HMA RGI regions)
# fn_gla = '/data/icesat/travail_en_cours/romain/results/vol_final/dh_13_14_15_rgi60_int_base.csv'
# df_gla = pd.read_csv(fn_gla)
# fn_shp='/data/icesat/travail_en_cours/romain/data/outlines/00.HIMAP_regions/boundary_mountain_regions_hma_v3.shp'
# field_name = 'Primary_ID'

# for global, concatenate all per-glacier results (using 01-02 and 13-15 as regions 20 and 21 to propagate correlated uncertainties rigorously)
int_dir = '/data/icesat/travail_en_cours/romain/results/vol_final'
list_fn_base= [os.path.join(int_dir,'dh_'+str(i).zfill(2)+'_rgi60_int_base.csv') for i in [3,4,6,7,8,9,10,11,12,16,17,18]]
list_fn_base = list_fn_base + [os.path.join(int_dir,'dh_01_02_rgi60_int_base.csv'),os.path.join(int_dir,'dh_13_14_15_rgi60_int_base.csv')]
list_reg = [3,4,6,7,8,9,10,11,12,16,17,18,20,21]

list_df = []
for fn_base in list_fn_base:
    df = pd.read_csv(fn_base)
    list_df.append(df)
df_gla = pd.concat(list_df)

# RGI first-order regions
# fn_shp = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/00_rgi60_O1Regions.shp'

# RGI second-order regions
fn_shp = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/00_rgi60_O2Regions.shp'
field_name = 'RGI_CODE'
# file containing RGI 6.0 (updated for this study) metadata; only necessary for sort_tw (sort tidewater/non-tidewater) = True
fn_base = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/base_rgi.csv'

df_cumul, df_rates, _, _ = aggregate_int_to_shp(df_gla,fn_shp,field_name=field_name,nproc=nproc,sort_tw=True,fn_base=fn_base)

df_cumul.to_csv('/data/icesat/travail_en_cours/romain/results/vol_final/subreg_O2_cumul.csv')
df_rates.to_csv('/data/icesat/travail_en_cours/romain/results/vol_final/subreg_O2_rates.csv')
