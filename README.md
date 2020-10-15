# ww_tvol_study

Code and results of Hugonnet et al. (202X), Accelerated global glacier mass loss in the early twenty-first century.

Below a short guide to: manipulate the dataset, reproduce the processing steps, and reproduce the figures and tables.

## Manipulate the dataset

### Retrieve the data

The dataset consists of:
1. **Time series of volume and mass change** (.csv) for global, regional (RGI O1 and O2), per-tile (0.5x0.5°, 1x1° and 
2x2°) and per-glacier at an annual time step, available at [TBC]()
2. **Elevation change rasters** (.tif) at 100 m posting for successive 5-year, 10-year and 20-year periods of 2000-2019,
 available at [TBC]()
3. **Elevation time series** (.nc) at 100 m posting and monthly time step, available at [TBD]()
4. **Bias-corrected ASTER DEMs** at 30 m posting, available at [TBD]()

*Note: tile mass changes (e.g., 1x1°) currently rely on per-glacier integrated volumes later aggregated according to 
glacier outline centroids. Therefore, those changes are not necessarily representative of mass change within the exact 
spatial boundaries of a tile. Deriving those specific changes is more complex and is not available yet (contact me for more details).* 

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

To generate Figures, Tables and values cited in the main text, we use the scripts located in *figures/* and *tables/*.

Additional data might be necessary to run some of these scripts, such as a world hillshade (we used 
[SRTM30_PLUS v8.0](https://researchdata.edu.au/global-hillshading-srtm30plus-source-ucsd/690579)), buffered RGI 6.0 
outlines (see *inventories/*), or auxiliary files of the data analysis not shared through the dataset (available upon request).

For example, the equivalent of Extended Data Fig. 7 colored by the number of valid observation by 5-year period 
(instead of elevation change) and scaled to glacier area (instead of inversely scaled to uncertainties):

```python
TBC
```

As further guide, you will find comments directly present in the structure of the code, and detailed documentation of our packages ([pyddem](https://pyddem.readthedocs.io/en/latest/),
[pymmaster](https://mmaster-workflows.readthedocs.io/en/latest/index.html)).

**Enjoy !** :snowflake: