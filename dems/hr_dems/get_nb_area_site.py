"""
@author: hugonnet
get table of glacier number and areas for Table S2
"""

import os
import pandas as pd
import numpy as np

df=pd.read_csv('/home/atom/data/validation/Hugonnet_2020/dhdt_int_HR.csv')

list_sites = list(set(list(df.site)))

nb_gla = []
area_gla = []

for site in list_sites:

    nb_gla.append(len(df[df.site==site]))
    area_gla.append(np.nansum(df[df.site==site].area.values/1000000))

df_out = pd.DataFrame()
df_out['site']=list_sites
df_out['nb_gla']=nb_gla
df_out['area']=area_gla
df_out.to_csv('/home/atom/ongoing/work_worldwide/tables/table_hr_dem_nb.csv')