#!/bin/bash
#bash script to retrieve IODEM3 recursively from a csv file of HTTPS links

#csv list of IODEM3 links
input=/data/icesat/travail_en_cours/romain/ww_tvol_study/worldwide/global/IceBridge/list_icebridge_01_02_rgi60.csv #if end of string displays with Windows carrier (i.e %OD, resulting in Error 404 for wget: correct csv file with: dos2unix(list.csv))
#download folder output
output=/calcul/santo/hugonnet/icebridge/01_02_rgi60/
#number of simultaneous downloads
nb_paral=10

cd ~
touch .netrc
echo "machine urs.earthdata.nasa.gov login <username> password <password>" >> .netrc
chmod 0600 .netrc
cd ~
touch .urs_cookies

# download granules tifs and mets
cat $input | parallel --verbose --delay 1 -j $nb_paral wget --no-check-certificate --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies {} -P $output
