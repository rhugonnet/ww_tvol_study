#!/bin/bash
#@author: hugonnet
#(inspired from Bob's process_mmaster.sh)
#wrapper for mass processing of AST L1A with MicMac/MMASTER (Luc Girod, Chris Nuth, Bob McNabb)
#we assume the following structure for L1A data after sorting (sort_l1a.py): region/utm/strips/

#runs as follows: process_l1a_mmaster.sh datadir tdir procdir

#datadir: path to parent directory where l1a strips are located (utm zone directory): usually on storage disks
datadir=$1
#tdir: path to target directory to write final results: usually on storage disks
tdir=$2
#procdir: path to directory where processing is done: usually on a HPC node or a SSD
procdir=$3

#if using this standalone, need to get environment variables ready for MicMac + MMASTER workflows
#source $HOME/.bashrc
#export PATH=/home/hugonnet/code/MMASTER-workflows-chris_updates:$PATH

#we assume parent directory name of the form 09N or 56S, and get corresponding utm zone code: "9 +north" or "56 +south"
get_utmzone () {
    l1adir=$1

    path=${l1adir%/*}
    utm_dirn=${path##*/}

    utm_z_nb=`echo ${utm_dirn:0:2} | sed 's/^0*//'`
    utm_z_hemi=`echo ${utm_dirn:2:1}`

    if [ "$utm_z_hemi" == "S" ]; then
        utm_str2="+south"
    elif [ "$utm_z_hemi" == "N" ]; then
        utm_str2="+north"
    else
        return [n]
    fi

    echo "$utm_z_nb $utm_str2"
}
utm=`get_utmzone $datadir`
echo $utm

strip_id=${datadir##*/}

#go to processing directory
cd ${procdir}

#copy l1a data to temp strip directory
rm -rf proc_${strip_id}
mkdir -p proc_${strip_id}
cp -r ${datadir} proc_${strip_id}

# go to temp strip directory
cd proc_${strip_id}

#process + postproc + clean after each single strip, to avoid filling up the nodes temporary disks
WorkFlowASTER.sh -s ${strip_id} -z "$utm" -a -i 2 > ${strip_id}.log
PostProcessMicMac.sh -z "$utm"
CleanMicMac.sh

#move data to target directory
#mv -v PROCESSED_INITIAL/* $tdir/PROCESSED_INITIAL
cd PROCESSED_FINAL

mkdir -p ${tdir}/no_dem

dem=${strip_id}/${strip_id}_Z.tif
#if we find the DEM file (AST_L1A_..._Z.tif), we copy to target directory ; metadata is left outside the zip
if [ -f "$dem" ]; then
    mv ${strip_id}/*.zip.met ./
    zip -r ${strip_id}.zip ${strip_id}/
    rm -f ${strip_id}/*
    mv -v *.zip ${strip_id}/
    mv -v *.zip.met ${strip_id}/
    mv -v ../*.log ${strip_id}/
    mv -v ./* ${tdir}
#if we don't, an error occured ; we copy the l1a data to a "no_dem" folder if we want to do a rerun in the future
else
    mv -v ../*.log ${strip_id}/
    mv -v ../PROCESSED_INITIAL/${strip_id}/zips/* ${strip_id}/
    mv -v ./* ${tdir}/no_dem
fi

#remove temp strip directory
cd ${procdir}
rm -rf proc_${strip_id}

echo "Strip " $datadir " has finished MMASTER processing"