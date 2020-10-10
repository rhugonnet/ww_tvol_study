"""
@author: NSIDC/hugonnet
use region polygons to select download links for IODEM3 data and write to csv file
"""
from __future__ import print_function
import gdal, ogr
import os, sys, shutil
import shapely.wkt
import base64
import itertools
import json
import netrc
import ssl
import sys
import gdal, ogr
from getpass import getpass
from urllib.parse import urlparse
from urllib.request import urlopen, Request, build_opener, HTTPCookieProcessor
from urllib.error import HTTPError, URLError
import csv

CMR_URL = 'https://cmr.earthdata.nasa.gov'
URS_URL = 'https://urs.earthdata.nasa.gov'
CMR_PAGE_SIZE = 2000
CMR_FILE_URL = ('{0}/search/granules.json?provider=NSIDC_ECS'
                '&sort_key[]=start_date&sort_key[]=producer_granule_id'
                '&scroll=true&page_size={1}'.format(CMR_URL, CMR_PAGE_SIZE))


def cmr_filter_urls(search_results):
    """Select only the desired data files from CMR response."""
    if 'feed' not in search_results or 'entry' not in search_results['feed']:
        return []

    entries = [e['links']
               for e in search_results['feed']['entry']
               if 'links' in e]
    # Flatten "entries" to a simple list of links
    links = list(itertools.chain(*entries))

    urls = []
    unique_filenames = set()
    for link in links:
        if 'href' not in link:
            # Exclude links with nothing to download
            continue
        if 'inherited' in link and link['inherited'] is True:
            # Why are we excluding these links?
            continue
        if 'rel' in link and 'data#' not in link['rel']:
            # Exclude links which are not classified by CMR as "data" or "metadata"
            continue

        if 'title' in link and 'opendap' in link['title'].lower():
            # Exclude OPeNDAP links--they are responsible for many duplicates
            # This is a hack; when the metadata is updated to properly identify
            # non-datapool links, we should be able to do this in a non-hack way
            continue

        filename = link['href'].split('/')[-1]
        if filename in unique_filenames:
            # Exclude links with duplicate filenames (they would overwrite)
            continue
        unique_filenames.add(filename)

        urls.append(link['href'])

    return urls


def cmr_search(short_name, version, time_start, time_end,
               polygon='', filename_filter=''):
    """Perform a scrolling CMR query for files matching input criteria."""
    cmr_query_url = build_cmr_query_url(short_name=short_name, version=version,
                                        time_start=time_start, time_end=time_end,
                                        polygon=polygon, filename_filter=filename_filter)
    print('Querying for data:\n\t{0}\n'.format(cmr_query_url))

    cmr_scroll_id = None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        urls = []
        while True:
            req = Request(cmr_query_url)
            if cmr_scroll_id:
                req.add_header('cmr-scroll-id', cmr_scroll_id)
            response = urlopen(req, context=ctx)
            if not cmr_scroll_id:
                # Python 2 and 3 have different case for the http headers
                headers = {k.lower(): v for k, v in dict(response.info()).items()}
                cmr_scroll_id = headers['cmr-scroll-id']
                hits = int(headers['cmr-hits'])
                if hits > 0:
                    print('Found {0} matches.'.format(hits))
                else:
                    print('Found no matches.')
            search_page = response.read()
            search_page = json.loads(search_page.decode('utf-8'))
            url_scroll_results = cmr_filter_urls(search_page)
            if not url_scroll_results:
                break
            if hits > CMR_PAGE_SIZE:
                print('.', end='')
                sys.stdout.flush()
            urls += url_scroll_results

        if hits > CMR_PAGE_SIZE:
            print()
        return urls
    except KeyboardInterrupt:
        quit()

def get_username():
    username = 'username'

    # For Python 2/3 compatibility:
    try:
        do_input = raw_input  # noqa
    except NameError:
        do_input = input

    while not username:
        try:
            username = do_input('Earthdata username: ')
        except KeyboardInterrupt:
            quit()
    return username


def get_password():
    password = 'password'
    while not password:
        try:
            password = getpass('password: ')
        except KeyboardInterrupt:
            quit()
    return password


def get_credentials(url):
    """Get user credentials from .netrc or prompt for input."""
    credentials = None
    try:
        info = netrc.netrc()
        username, account, password = info.authenticators(urlparse(URS_URL).hostname)
    except Exception:
        try:
            username, account, password = info.authenticators(urlparse(CMR_URL).hostname)
        except Exception:
            username = None
            password = None

    while not credentials:
        if not username:
            username = get_username()
            password = get_password()
        credentials = '{0}:{1}'.format(username, password)
        credentials = base64.b64encode(credentials.encode('ascii')).decode('ascii')

        if url:
            try:
                req = Request(url)
                req.add_header('Authorization', 'Basic {0}'.format(credentials))
                opener = build_opener(HTTPCookieProcessor())
                opener.open(req)
            except HTTPError:
                print('Incorrect username or password')
                credentials = None
                username = None
                password = None

    return credentials


def build_version_query_params(version):
    desired_pad_length = 3
    if len(version) > desired_pad_length:
        print('Version string too long: "{0}"'.format(version))
        quit()

    version = str(int(version))  # Strip off any leading zeros
    query_params = ''

    while len(version) <= desired_pad_length:
        padded_version = version.zfill(desired_pad_length)
        query_params += '&version={0}'.format(padded_version)
        desired_pad_length -= 1
    return query_params


def build_cmr_query_url(short_name, version, time_start, time_end, polygon=None, filename_filter=None):
    params = '&short_name={0}'.format(short_name)
    params += build_version_query_params(version)
    params += '&temporal[]={0},{1}'.format(time_start, time_end)
    if polygon:
        params += '&polygon={0}'.format(polygon)
    if filename_filter:
        params += '&producer_granule_id[]={0}&options[producer_granule_id][pattern]=true'.format(filename_filter)
    return CMR_FILE_URL + params

if __name__ == '__main__':

    short_name = 'IODEM3'
    version = '1'
    time_start = '2009-10-16T00:00:00Z'
    time_end = '2017-07-25T23:59:59Z'
    # polygon = '-44.886630719305,60.831437082012386,-45.220102452166344,59.79435280656104,-42.96579765102328,60.15679679420265,-42.92126967842339,61.416673255520834,-44.177410009763875,61.32415503452134,-44.886630719305,60.831437082012386'
    filename_filter = '*'

    rgi_naming_txt = '/home/atom/proj/ww_tvol_study/worldwide/rgi_neighb_merged_naming_convention.txt'
    ww_dir = '/home/atom/proj/ww_tvol_study/worldwide/'
    text_file = open(rgi_naming_txt, 'r')
    rgi_list = text_file.readlines()

    list_tile_shp = [os.path.join(ww_dir, rgi[:-1].split('rgi60')[0] + 'rgi60', 'cov',
                                  'tile_glacierized_' + rgi[:-1].split('rgi60')[0] + 'rgi60',
                                  'tile_glacierized_' + rgi[:-1].split('rgi60')[0] + 'rgi60.shp') for rgi in rgi_list]

    list_tile_shp = list_tile_shp[-2:]
    rgi_list = rgi_list[-2:]

    for tile_shp in list_tile_shp:

        list_urls = []
        rgi = rgi_list[list_tile_shp.index(tile_shp)][:-1].split('rgi60')[0] + 'rgi60'
        out_list = '/home/atom/ongoing/list_icebridge_'+rgi+'.csv'

        ds_shp = ogr.GetDriverByName("ESRI Shapefile").Open(tile_shp, 0)
        layer = ds_shp.GetLayer()
        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom.GetGeometryName() == 'MULTIPOLYGON':
                list_geom = []
                for geom_part in geom:
                    list_geom.append(geom_part)
            else:
                list_geom = [geom]

            for test in list_geom:
                poly = shapely.wkt.loads(test.ExportToWkt())
                x, y = poly.exterior.coords.xy

                poly_list = [None] * (len(list(x)) + len(list(y)))
                in_x = list(x)
                in_x.reverse()
                in_y = list(y)
                in_y.reverse()

                poly_list[::2] = in_x
                poly_list[1::2] = in_y

                print(poly_list)
                # str_poly_list = ["{0:.12f}".format(x) for x in poly_list]
                polygon = ','.join(map(str,poly_list))

                try:
                    urls = cmr_search(short_name, version, time_start, time_end,
                                      polygon=polygon, filename_filter=filename_filter)

                    list_urls += urls
                except HTTPError:
                    print('No intersecting IceBridge data, skipping...')


        if len(list_urls)>1:
            with open(out_list, 'w') as file:
                writer = csv.writer(file, delimiter=',')
                for url in list_urls:
                    writer.writerow([url])
