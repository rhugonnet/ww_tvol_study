#!/bin/bash
#bash script to retrieve aster l1a recursively from a csv file of HTTPS directories

#filename of csv list with LPDAAC PullDirs links
fn_csv=/home/atom/proj/aster_tdem/worldwide/coverage/L1A_retrieval/list_PullDirs_ww_1.csv #if end of string displays with Windows carrier (i.e %OD, resulting in Error 404 for wget: correct csv file with: dos2unix(list.csv))
#download folder output
out_dir=/home/atom/tmp/filezilla/Test_L1A_retrieval
#number of simultaneous downloads
nb_paral=10

cd ~
touch .netrc
echo "machine urs.earthdata.nasa.gov login <user> password <password>" >> .netrc
chmod 0600 .netrc
cd ~
touch .urs_cookies
#touch $logpath

# while IFS=',' read -r var
# do
# 	echo "Downloading from folder $var..." | tee -a $logpath
# 	wget --no-check-certificate --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r -np -l 1 -A *.zip $var -P $output 2>&1 | grep -i "failed\|error" | tee -a $logpath
# 	echo "Downloading from folder $var is finished." | tee -a $logfile
# done < $input

# download granules zip and met recursively
cat $fn_csv | parallel --verbose --delay 1 -j $nb_paral --results $out_dir wget --no-check-certificate --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r -np -l 1 -A *.zip,*.met {} -P $out_dir

# download zip archive of ~100 granules >> easier to check for errors but zip will be corrupted if file size > 4Go
#cat $input | parallel --verbose --delay 1 -j $nb_paral --results $output wget --no-check-certificate --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies {} -P $output
