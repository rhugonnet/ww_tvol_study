# ww_tvol_study

This repository contains all scripts used to process DEMs into globally resolved time series of glacier mass change (see #provide link when available).

Routines rely on the following packages (todo: provide environment file):
- pybob: for manipulating georeferenced data, co-registering DEMs (Nuth and Kääb (2012)).
- MMASTER-workflows: for generating and correcting ASTER DEMs
- pyddem: for creating DEM stacks, interpolating time series and deriving volume changes with uncertainty propagation
- rh_pygeotools: old routines for manipulating georeferenced data and process various DEM products (ArcticDEM, REMA, TanDEM-X, ...)
