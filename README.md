# ww_tvol_study

Code and results of **Hugonnet et al. (2021), *Accelerated global glacier mass loss in the early twenty-first century***. :earth_americas: :snowflake:

Below a short guide to: manipulate the dataset, reproduce the processing steps, and reproduce the figures and tables.

## Manipulate the dataset

### Retrieve the data

The dataset consists of:
1. **Cumulative time series of volume and mass change** (.csv, *~500MB*) at an annual time step and **rates of volume 
and mass changes** (.csv, *~3GB*) for successive 1-, 2-, 4-, 5-, 10- and 20-year periods of 2000-2019, both 
for global, regional (RGI O1 and O2), per-tile (0.5x0.5°, 1x1° and 2x2°) and per-glacier, available at [https://doi.org/10.5281/zenodo.4530314]().
2. **Elevation change rasters** (.tif, *~15GB*) at 100 m posting for successive 5-, 10- and 20-year periods of 2000-2019,
 available at [https://doi.org/10.5281/zenodo.4530314]().
3. **Elevation time series** (.nc, *~3TB*) at 100 m posting and monthly time step (in time, should be available through NSIDC): [TBC]().
4. **Bias-corrected ASTER DEMs** (.tif, ~*10TB*) at 30 m posting, available at [TBC]().

*Notes:*
* *Global and regional series of specific (area-scaled) change presented throughout the article (e.g., mean elevation change) use
time-varying areas (see Methods). Due to the lack of such estimates per individual glaciers, specific change per glacier is computed based on fixed areas. 
Thus, only **global and regional direct mass change/volume change are consistent with the individual glacier contributions of a given region**.
Regional/global rates must therefore be manipulated with caution by using direct mass change/volume change, and later choosing an area 
to scale into specific rates, fixed or time-varying.*
* *Rates uncertainties over a specific period (e.g., 2004-2011) **need to be derived from the cumulative volume change
time series**, due to the varying spatial correlation at each point in time (volume time series) and temporal correlation at the regional 
scale assumed for certain uncertainties (density conversion).*
* *Mass change **cumulative series and rates for periods shorter than 5 years** are provided, but **do not respect assumptions of density conversion** 
of [Huss (2013)](https://tc.copernicus.org/articles/7/877/2013/), possibly resulting in too small uncertainties.*
* ***Tile mass changes** (e.g., 1x1° grid) currently rely on per-glacier integrated volumes, aggregated according to 
glacier outline centroids. Therefore, those changes **are not necessarily representative of mass change within the exact 
spatial boundaries of the tile**. Deriving such changes is more complex and is not available yet (contact me for more details).* 

### Setup environment

Most scripts rely on code assembled in the packages [pyddem](https://github.com/iamdonovan/pybob) (DEM time series) and 
[pymmaster](https://github.com/luc-girod/MMASTER-workflows) (ASTER processing), which themselves are based on 
[pybob](https://github.com/iamdonovan/pybob) (GDAL/OGR-based georeferenced processing).

You can rapidly install a working environment containing all those packages and their dependencies with the 
*ww_tvol_env.yml* file, located at the root of the repository, using 
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

```sh
conda env create -f ww_tvol_env.yml
```

Further details on setup and functions present in these packages are available through **[pyddem documentation](https://pyddem.readthedocs.io/en/latest/)** and
 **[pymmaster documentation](https://mmaster-workflows.readthedocs.io/en/latest/index.html)**.

### How to use

Scripts for selecting or manipulating the dataset at various scales are located in *dataset/* and divide in three sections:
* *gla_vol_time_series/* for the **volume and mass change time series integrated over glaciers** (.csv)

* *h_time_series/* for the **elevation time series and elevation change rasters** (.nc and .tif)

* *raw_dems/* for the **bias-corrected ASTER DEMs**

Below a few examples:

**1. Aggregating volume change over a specific regional shapefile and deriving rates for specific time periods**

:exclamation: *Propagation of errors can be CPU intensive, and might require running in parallel*

```python
TBC
```

**2. Aggregating volume change over 4x4° tiles**

```python
TBC
```

**3. Deriving and displaying elevation change time series from a time stack**

```python
TBC
```

**4. Extracting 2-year elevation change rates**

```python
TBC
```

## Reproduce the processing steps

To generate the dataset, we sequentially use the scripts located in:

1. *inventories/*, to replace [RGI 6.0](https://www.glims.org/RGI/) outlines in RGI region 12 (Caucasus Middle East); 
and to derive a 10 km buffered zone around glaciers where to assess elevation changes,
2. *dems/*, to sort [TanDEM-X global DEM](https://geoservice.dlr.de/web/dataguide/tdm90/) tiles; to download, sort, 
generate, bias-correct and co-register ASTER DEMs based on [ASTER L1A data](https://lpdaac.usgs.gov/products/ast_l1av003/) 
(~5M CPU hours); to download and pairwise co-register [ArcticDEM](https://www.pgc.umn.edu/data/arcticdem/) and 
[REMA](https://www.pgc.umn.edu/data/rema/) DEMs (requires my old processing packages in [rh_pygeotools](https://github.com/rhugonnet/rh_pygeotools));
 and to download and process IceBridge [ILAKS1B](https://nsidc.org/data/ILAKS1B/versions/1) and [IODEM3](https://nsidc.org/data/IODEM3/versions/1) data,
3. *th/*, to stack DEMs; to filter extreme outliers; to estimate elevation measurement error; to refine the filtering of outliers;
 to estimate glacier elevation temporal covariances; and finally to filter remaining outliers and derive elevation time series
 through Gaussian Process regression,
4. *validation/*, to intersect our elevation time series to [ICESat](https://nsidc.org/data/glah14) and IceBridge; to compile results; and to estimate
 the spatial correlations of the time series depending on the time lag to the closest observation;
5. *tvol/*, to integrate our results into volume change; to aggregate time series at different scales with uncertainty 
propagation using the spatial correlation constrained with ICESat.

## Reproduce the Figures, Tables and main text values

To generate the Figures, Tables and values cited in the main text, we use the scripts located in *figures/* and *tables/*.

Additional data might be necessary to run some of these scripts, such as a world hillshade (we used 
[SRTM30_PLUS v8.0](https://researchdata.edu.au/global-hillshading-srtm30plus-source-ucsd/690579)), buffered RGI 6.0 
outlines (see *inventories/*), or auxiliary files of the data analysis not shared through the dataset (available upon request).

Additionally, we can also display other variables of the dataset. For example, using the equivalent of Extended Data Fig. 7 
colored by the number of valid observation by 5-year period.
Some examples are given in *dataset/examples_figs/*, with figures directly below:

**Mean elevation change rate of full period on 2x2° tiles (4x2° for latitude higher than 60°, 4x4° for latitudes higher than 74°)**
```shell script
./dataset/example_figs/alike_fig_2_2by2.py
```
![alt text](https://github.com/rhugonnet/ww_tvol_study/blob/main/dataset/example_figs/fig_2_2by2?raw=True)


**Number of valid observations per 100 m pixel per 5-year period on 1x1° tiles**
```shell script
./dataset/example_figs/alike_fig_ed7_datacov.py
```
![alt text](https://github.com/rhugonnet/ww_tvol_study/blob/main/dataset/example_figs/fig_ed7_alike_datacov.png?raw=True)

**Mean elevation change rate per 5-year period on 0.5x0.5° tiles**
```shell script
./dataset/example_figs/alike_fig_ed7_05by05.py
```
![alt text](https://github.com/rhugonnet/ww_tvol_study/blob/main/dataset/example_figs/fig_ed7_alike_05by05.png?raw=true)

**Mean elevation change rate per 2-year period on 1x1° tiles as a GIF**
```shell script
./dataset/example_figs/alike_fig_ed7_biennial_gif.py
```
![alt text](https://github.com/rhugonnet/ww_tvol_study/blob/main/dataset/example_figs/fig_ed7_biennial_gif.gif?raw=True)


As further guide, you will find comments directly present in the structure of the code, or in the
 documentation of our packages ([pyddem](https://pyddem.readthedocs.io/en/latest/),
[pymmaster](https://mmaster-workflows.readthedocs.io/en/latest/index.html)).

**Enjoy !** :snowman: