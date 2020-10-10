#!/usr/bin/env bash

#wrapper for processing one or several RGI regions  (or any shapefile grouping that went through sort_l1a.py) containing UTM zones created by the sorting algorithm

#parameters
region_dir=/data/lidar2/RGI_paper/worldwide/03_rgi60/aster_l1a/
out_dir=/data/lidar2/RGI_paper/worldwide_out/03_rgi60/aster_dem/
proc_dir=/tmp/
log_dir=/home/hugonnet/code/MMASTER-workflows-chris_updates/

#wrapper
cd ${region_dir}

utm_dirs=$(ls -d ${region_dir}*)

cd ${log_dir}
echo "Queueing: wrapper_utm_mmaster_sbatch.sh ${out_dir} ${proc_dir} ${utm_dirs}"
sbatch sbatch_utm_mmaster.sh ${out_dir} ${proc_dir} ${utm_dirs} &

echo 'Fin.'

