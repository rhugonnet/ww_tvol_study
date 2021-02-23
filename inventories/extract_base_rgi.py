"""
extract relevant data (connectivity level, nominal glaciers, etc...) from RGI shapefiles to integrate in the rest of calculations
"""
import pyddem.tdem_tools as tt

dir_shp = '/home/atom/data/inventory_products/RGI/00_rgi60_neighb_renamed'

outfile = '/home/atom/data/inventory_products/RGI/base_rgi.csv'

tt.get_base_df_inventory(dir_shp,outfile)
