# -*- coding: utf-8 -*-
"""
@author: hugonnet
Python routines to sort downloaded ASTER L1A granules by:
- shapefile used to download the data (EarthData Search): if no shapefile was used, remove this step in the main
- remove duplicates (due to adjacent polygons in space, overlapping filters in time, etc...)
- strips (less than 12s apart: Bob's script) within a region (corresponding to each shapefile, can be a multipolygon)
- utm zone (based on the centroid of each strip, reading the metadata)
for a final architecture of: /datadir/region/aster_l1a/utm_zone/strips/l1a.zip + l1a.met

l1a data with no stereo bands is sorted in a separate folders
"""
from __future__ import print_function
import sys
from osgeo import ogr
import os
import shutil
import errno
from datetime import datetime, timedelta
from glob import glob
import numpy as np
import pandas as pd


def extract_odl_astL1A(fn):
    f = open(fn, 'r')
    body = f.read()

    def get_odl_parenth_value(text_odl, obj_name):
        posobj = str.find(text_odl, obj_name)
        posval = str.find(text_odl[posobj + 1:len(text_odl)], 'VALUE')
        posparenthesis = str.find(text_odl[posobj + 1 + posval:len(text_odl)], '(')
        posendval = str.find(text_odl[posobj + 1 + posval + posparenthesis:len(text_odl)], ')')

        val = text_odl[posobj + posval + posparenthesis + 2:posobj + posval + posparenthesis + posendval + 1]

        return val

    def get_odl_quot_value(text_odl, obj_name):
        posobj = str.find(text_odl, obj_name)
        posval = str.find(text_odl[posobj + 1:len(text_odl)], 'VALUE')
        posquote = str.find(text_odl[posobj + 1 + posval:len(text_odl)], '"')
        posendval = str.find(text_odl[posobj + posval + posquote + 2:len(text_odl)], '"')

        val = text_odl[posobj + posval + posquote + 2:posobj + posval + +posquote + posendval + 2]

        return val

    # get latitude
    lat_val = get_odl_parenth_value(body, 'GRingPointLatitude')
    lat_tuple = [float(lat_val.split(',')[0]), float(lat_val.split(',')[1]), float(lat_val.split(',')[2]),
                 float(lat_val.split(',')[3])]

    # get longitude
    lon_val = get_odl_parenth_value(body, 'GRingPointLongitude')
    lon_tuple = [float(lon_val.split(',')[0]), float(lon_val.split(',')[1]), float(lon_val.split(',')[2]),
                 float(lon_val.split(',')[3])]

    # get calendar date + time of day
    caldat_val = get_odl_quot_value(body, 'CalendarDate')
    timeday_val = get_odl_quot_value(body, 'TimeofDay')
    caldat = datetime(year=int(caldat_val.split('-')[0]), month=int(caldat_val.split('-')[1]),
                      day=int(caldat_val.split('-')[2]),
                      hour=int(timeday_val.split(':')[0]), minute=int(timeday_val.split(':')[1]),
                      second=int(timeday_val.split(':')[2][0:2]),
                      microsecond=int(timeday_val.split(':')[2][3:6]) * 1000)

    # get cloud cover
    cloudcov_val = get_odl_quot_value(body, 'SceneCloudCoverage')
    cloudcov_perc = int(cloudcov_val)

    # get flag if bands acquired or not: band 1,2,3N,3B,4,5,6,7,8,9,10,11,12,13,14
    list_band = []
    band_attr = get_odl_quot_value(body, 'Band3N_Available')
    band_avail = band_attr[0:3] == 'Yes'
    list_band.append(band_avail)
    band_attr = get_odl_quot_value(body, 'Band3B_Available')
    band_avail = band_attr[0:3] == 'Yes'
    list_band.append(band_avail)

    range_band = list(range(1, 15))
    range_band.remove(3)
    for i in range_band:
        band_attr = get_odl_quot_value(body, 'Band' + str(i) + '_Available')
        band_avail = band_attr[0:3] == 'Yes'
        list_band.append(band_avail)

    band_tags = pd.DataFrame(data=list_band,
                             index=['band_3N', 'band_3B', 'band_1', 'band_2', 'band_4', 'band_5', 'band_6', 'band_7',
                                    'band_8', 'band_9', 'band_10', 'band_11', 'band_12', 'band_13', 'band_14'])

    # get scene orientation angle
    orient_attr = get_odl_quot_value(body, 'ASTERSceneOrientationAngle')
    # orient_angl = float(orient_attr)
    # some .met files are in fact somehow incomplete for angles... let's forget it!
    orient_angl = float(15.)

    return lat_tuple, lon_tuple, caldat, cloudcov_perc, band_tags, orient_angl


def latlon_to_UTM(lat, lon):
    # utm module excludes regions south of 80°S and north of 84°N, unpractical for global vector manipulation
    # utm_all = utm.from_latlon(lat,lon)
    # utm_nb=utm_all[2]

    # utm zone from longitude without exclusions
    if -180 <= lon < 180:
        utm_nb = int(
            np.floor((lon + 180) / 6)) + 1  # lon=-180 refers to UTM zone 1 towards East (West corner convention)
    else:
        sys.exit('Longitude value is out of range.')

    if 0 <= lat < 90:  # lat=0 refers to North (South corner convention)
        epsg = '326' + str(utm_nb).zfill(2)
        utm_zone = str(utm_nb).zfill(2) + 'N'
    elif -90 <= lat < 0:
        epsg = '327' + str(utm_nb).zfill(2)
        utm_zone = str(utm_nb).zfill(2) + 'S'
    else:
        sys.exit('Latitude value is out of range.')

    return epsg, utm_zone


def read_geom_from_shp(in_shp):
    print('Reading shapefile: ' + in_shp + '...')

    # get layer from ESRI shapefile
    inShapefile = in_shp
    inDriver = ogr.GetDriverByName("ESRI Shapefile")
    inDataSource = inDriver.Open(inShapefile, 0)
    inLayer = inDataSource.GetLayer()
    proj = inLayer.GetSpatialRef()

    # collect all geometry in a geometry collection
    geomcol = ogr.Geometry(ogr.wkbGeometryCollection)
    for feature in inLayer:
        geomcol.AddGeometry(feature.GetGeometryRef())

    return geomcol, proj.ExportToWkt()


def polygon_list_to_multipoly(list_poly):
    print('Creating multipolygon from polygon list...')

    # create empty multipolygon
    multipoly = ogr.Geometry(ogr.wkbMultiPolygon)

    for i in range(len(list_poly)):
        # stacking polygons in multipolygon
        multipoly.AddGeometry(list_poly[i])

    return multipoly


def union_cascaded_multipoly(multipoly):
    print('Calculating cascaded union of multipolygon...')
    # calculating cascaded union of multipolygon
    cascadedpoly = multipoly.UnionCascaded()

    return cascadedpoly


def get_poly_centroid(poly):
    centroid = poly.Centroid()

    center_lon, center_lat, _ = centroid.GetPoint()

    return center_lon, center_lat


def poly_from_coords(list_coord):
    # creating granule polygon
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for coord in list_coord:
        ring.AddPoint(coord[0], coord[1])

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    return poly


def sort_l1a_by_region(list_pulldir, list_shp, out_dir):
    # first, load shapefiles of region polygons
    list_poly_shp = []
    for shp in list_shp:
        poly_shpf = read_geom_from_shp(shp)
        list_poly_shp.append(poly_shpf)

    for pulldir in list_pulldir:

        # list l1a files in temporary directory
        list_l1af = [os.path.join(pulldir, fil) for fil in os.listdir(pulldir) if fil.endswith('.met')]

        for l1af in list_l1af:

            print('Calculating intersections for granule ' + str(list_l1af.index(l1af) + 1) + ' out of ' + str(
                len(list_l1af)) + ' of archive ' + os.path.basename(pulldir) + ':' + os.path.basename(l1af))
            lat_tup, lon_tup, _, _, df, _ = extract_odl_astL1A(l1af)

            max_lon = np.max(lon_tup)
            min_lon = np.min(lon_tup)

            if min_lon < -160 and max_lon > 160:
                # dateline exception...

                # we need the intersection, so let's do two split polygons on each side of the dateline...
                lon_rightside = np.array(lon_tup, dtype=float)
                lon_rightside[lon_rightside < -160] += 360

                lon_leftside = np.array(lon_tup, dtype=float)
                lon_leftside[lon_leftside > 160] -= 360

                rightside_coord = list(zip(list(lon_rightside) + [lon_rightside[0]], lat_tup + [lat_tup[0]]))
                rightside_poly = poly_from_coords(rightside_coord)

                leftside_coord = list(zip(list(lon_leftside) + [lon_leftside[0]], lat_tup + [lat_tup[0]]))
                leftside_poly = poly_from_coords(leftside_coord)

                # create a world polygon and get intersection
                world_coord = [(-180, -90), (-180, 90), (180, 90), (180, -90), (-180, -90)]
                world_poly = poly_from_coords(world_coord)

                leftside_inters = world_poly.Intersection(leftside_poly)
                rightside_inters = world_poly.Intersection(rightside_poly)

                poly = polygon_list_to_multipoly([leftside_inters,rightside_inters])
            else:
                list_coord = list(zip(lon_tup + [lon_tup[0]], lat_tup + [lat_tup[0]]))
                poly = poly_from_coords(list_coord)

            poly2 = poly.Buffer(0.5)
            # find all non-empty rgi intersections
            rgi_reg_to_copy = []
            for poly_shp in list_poly_shp:
                polyintersect = poly2.Intersection(poly_shp)  # polygon intersection

                if not polyintersect.IsEmpty():
                    rgi_reg_to_copy.append(
                        os.path.basename(rgi_list[list_poly_shp.index(poly_shp)]).split('rgi60')[0] + 'rgi60')

            # copy or move to proper directories
            no_inters_dir = os.path.join(out_dir, 'no_inters')

            if len(rgi_reg_to_copy) == 0:

                print('Found no intersection.')

                if not os.path.exists(no_inters_dir):
                    os.mkdir(no_inters_dir)

                if df.loc['band_3N'].values[0] and df.loc['band_3B'].values[0]:
                    shutil.move(l1af, os.path.join(no_inters_dir, os.path.basename(l1af)))
                    shutil.move(l1af[:-4], os.path.join(no_inters_dir, os.path.basename(l1af[:-4])))

                else:
                    no_stereo_dir = os.path.join(no_inters_dir, 'no_stereoband')
                    if not os.path.exists(no_stereo_dir):
                        os.mkdir(no_stereo_dir)
                    shutil.move(l1af, os.path.join(no_stereo_dir, os.path.basename(l1af)))
                    shutil.move(l1af[:-4], os.path.join(no_stereo_dir, os.path.basename(l1af[:-4])))

            elif len(rgi_reg_to_copy) == 1:

                print('Found 1 intersection.')
                print('Moving to ' + rgi_reg_to_copy[0] + '...')

                rgi_reg_dir = os.path.join(out_dir, rgi_reg_to_copy[0], 'aster_l1a')
                if not os.path.exists(os.path.join(out_dir, rgi_reg_to_copy[0])):
                    os.mkdir(os.path.join(out_dir, rgi_reg_to_copy[0]))
                if not os.path.exists(rgi_reg_dir):
                    os.mkdir(rgi_reg_dir)

                if df.loc['band_3N'].values[0] and df.loc['band_3B'].values[0]:
                    shutil.move(l1af, os.path.join(rgi_reg_dir, os.path.basename(l1af)))
                    shutil.move(l1af[:-4], os.path.join(rgi_reg_dir, os.path.basename(l1af[:-4])))
                else:
                    no_stereo_dir = os.path.join(rgi_reg_dir, 'no_stereoband')
                    if not os.path.exists(no_stereo_dir):
                        os.mkdir(no_stereo_dir)
                    shutil.move(l1af, os.path.join(no_stereo_dir, os.path.basename(l1af)))
                    shutil.move(l1af[:-4], os.path.join(no_stereo_dir, os.path.basename(l1af[:-4])))

            elif len(rgi_reg_to_copy) >= 2:

                print('Found ' + str(len(rgi_reg_to_copy)) + ' intersections.')

                for rgi_reg in rgi_reg_to_copy:
                    print('Copying to ' + rgi_reg + '...')
                    rgi_reg_dir = os.path.join(out_dir, rgi_reg, 'aster_l1a')
                    if not os.path.exists(os.path.join(out_dir, rgi_reg)):
                        os.mkdir(os.path.join(out_dir, rgi_reg))
                    if not os.path.exists(rgi_reg_dir):
                        os.mkdir(rgi_reg_dir)

                    if df.loc['band_3N'].values[0] and df.loc['band_3B'].values[0]:
                        shutil.copy(l1af, os.path.join(rgi_reg_dir, os.path.basename(l1af)))
                        shutil.copy(l1af[:-4], os.path.join(rgi_reg_dir, os.path.basename(l1af[:-4])))
                    else:
                        no_stereo_dir = os.path.join(rgi_reg_dir, 'no_stereoband')
                        if not os.path.exists(no_stereo_dir):
                            os.mkdir(no_stereo_dir)
                        shutil.copy(l1af, os.path.join(no_stereo_dir, os.path.basename(l1af)))
                        shutil.copy(l1af[:-4], os.path.join(no_stereo_dir, os.path.basename(l1af[:-4])))

                os.remove(l1af)
                os.remove(l1af[:-4])

            else:
                sys.exit('Problem with the intersections. Exiting...')


def rm_duplic_l1a(dir_l1a):
    list_l1a = [os.path.join(dir_l1a, l1a) for l1a in os.listdir(dir_l1a) if l1a.endswith('.met')]
    list_final_l1a = []
    list_l1a_id = []

    for l1a in list_l1a:
        l1a_id = os.path.basename(l1a)[0:25]
        if l1a_id not in list_l1a_id:
            list_l1a_id.append(l1a_id)
            list_final_l1a.append(l1a)

    list_l1a_dupli = [l1a for l1a in list_l1a if l1a not in list_final_l1a]
    nb_dupli = len(list_l1a_dupli)
    print('Found ' + str(nb_dupli) + ' duplicates in ' + str(len(list_l1a)) + ' granules.')
    # tag_list = input('List the ' + str(nb_dupli) + ' duplicates in ' + dir_l1a + '? [y/n]')
    tag_list = 'y'
    if tag_list == 'y':
        print(list_l1a_dupli)
    # tag_rm = input('Remove the ' + str(nb_dupli) + ' duplicates in ' + dir_l1a + '? [y/n]')
    tag_rm = 'y'
    if tag_rm == 'y':
        for l1a_dupli in list_l1a_dupli:
            print('Removing ' + l1a_dupli + '...')
            os.remove(l1a_dupli)
            os.remove(l1a_dupli[:-4])


def sort_l1a_by_strip(dir_l1a, strip_length):
    def parse_aster_filename(fname):
        return datetime.strptime(fname[11:25], '%m%d%Y%H%M%S')

    def sliding_window(a, size, step):
        for i in range(0, len(a) - 1, step):
            yield (a[i:i + size])

    def mkdir_p(outdir):
        try:
            os.makedirs(outdir)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(outdir):
                pass
            else:
                raise

    print('Looking in folder {}'.format(dir_l1a))

    os.chdir(dir_l1a)

    flist = glob('*.zip.met')
    filenames = np.array([f.rsplit('.zip', 1)[0] for f in flist])
    filenames.sort()
    dates = [parse_aster_filename(f) for f in filenames]

    striplist = []
    # loop through the dates
    for i, s in enumerate(dates):
        # get a list of all the scenes we're currently using
        current_striplist = [item for sublist in striplist for item in sublist]
        # if the current filename is already in the sorted list, move on.
        if filenames[i] in current_striplist:
            continue
        else:
            td_list = np.array([d - s for d in dates])
            # because we sorted the filelist, we don't have to consider timedeltas
            # less than zero (i.e., scenes within a single day are chronologically ordered)
            matched_inds = np.where(np.logical_and(td_list >= timedelta(0),
                                                   td_list < timedelta(0, 600)))[0]
            # if we only get one index back, it's the scene itself.
            if len(matched_inds) == 1:
                striplist.append(filenames[matched_inds])
                continue
            # now, check that we have continuity (if we have a difference of more than 12 seconds,
            # then the scenes aren't continuous even if they come from the same day)
            matched_diff = np.diff(np.array(td_list)[matched_inds])
            break_inds = np.where(matched_diff > timedelta(0, 12))[0]
            if len(break_inds) == 0:
                pass
            else:
                # we only need the first index, add 1 because of diff
                break_ind = break_inds[0] + 1
                matched_inds = matched_inds[0:break_ind]
            # here, we make sure that we only return strips that are at most max_length long.
            for strip in sliding_window(matched_inds, strip_length, strip_length - 1):
                strip = list(strip)
                if len(matched_inds) > strip_length and len(strip) == strip_length - 1:
                    strip.insert(0, strip[0] - 1)
                striplist.append(filenames[strip])
    print('Found {} strips, out of {} individual scenes'.format(len(striplist), len(filenames)))
    # now that the individual scenes are sorted into "strips",
    # we can create "strip" and "single" folders
    print('Moving strips to individual folders.')
    # mkdir_p('strips')
    # mkdir_p('singles')
    mkdir_p('sorted')

    for s in striplist:
        mkdir_p(os.path.join(dir_l1a, 'sorted', s[0][0:25]))
        if len(s) == 1:
            shutil.move(s[0] + '.zip', os.path.join(dir_l1a, 'sorted', s[0][0:25]))
            shutil.move(s[0] + '.zip.met', os.path.join(dir_l1a, 'sorted', s[0][0:25]))
        else:
            for ss in s:
                shutil.copy(ss + '.zip', os.path.join(dir_l1a, 'sorted', s[0][0:25]))
                shutil.copy(ss + '.zip.met', os.path.join(dir_l1a, 'sorted', s[0][0:25]))
                # now, clean up the current folder.
    for f in glob('*.zip*'):
        os.remove(f)
    print('Fin.')


def sort_strip_by_utm(dir_l1a):
    sorted_dir = os.path.join(dir_l1a, 'sorted')
    list_subdir = [os.path.join(sorted_dir, subdir) for subdir in os.listdir(sorted_dir)]

    for subdir in list_subdir:

        # number of l1a granules
        strip_l1a = [os.path.join(subdir, l1a) for l1a in os.listdir(subdir) if l1a.endswith('.met')]

        list_poly = []
        for l1a in strip_l1a:
            lat_tup, lon_tup, _, _, _, _ = extract_odl_astL1A(l1a)

            max_lon = np.max(lon_tup)
            min_lon = np.min(lon_tup)

            if min_lon < -160:
                # if this is happening, ladies and gentlemen, bad news, we definitely have an image on the dateline

                # here we need the right centroid, so let's do polygons on only one side of the dateline: the right one
                lon_rightside = np.array(lon_tup, dtype=float)
                lon_rightside[lon_rightside < -160] += 360

                rightside_coord = list(zip(list(lon_rightside) + [lon_rightside[0]], lat_tup + [lat_tup[0]]))
                rightside_poly = poly_from_coords(rightside_coord)

                # add to list
                list_poly.append(rightside_poly)
            else:
                list_coord = list(zip(lon_tup + [lon_tup[0]], lat_tup + [lat_tup[0]]))
                poly = poly_from_coords(list_coord)
                list_poly.append(poly)

        multipoly = polygon_list_to_multipoly(list_poly)
        union = union_cascaded_multipoly(multipoly)
        centroid = get_poly_centroid(union)

        #now we get a centroid with longitude in a -180/180 longitude
        final_lon = ((centroid[0] + 180) % 360 ) - 180
        _, utm = latlon_to_UTM(centroid[1], final_lon)

        print('Centroid of strip is ' + str(centroid[0]) + ',' + str(
            centroid[1]) + ' [lon/lat] corresponding to UTM zone ' + utm)

        utm_dir = os.path.join(dir_l1a, utm)
        if not os.path.exists(utm_dir):
            os.mkdir(utm_dir)

        shutil.move(subdir, utm_dir)


if __name__ == '__main__':

    # list of directories directly containing l1a .zips and .met
    # here a typical structure once mass downloaded from LPDAAC recursively with wget
    pulldir_dir = '/mnt/lidar2/RGI_paper/ww_L1A/ww_t1/e4ftl01.cr.usgs.gov/PullDir/'
    list_pulldir = [os.path.join(pulldir_dir, subdir) for subdir in os.listdir(pulldir_dir)]

    # list of region shapefiles used for ordering data
    # here we use extended polygons derived from the RGI
    rgi_naming_txt = '/mnt/lidar2/RGI_paper/worldwide/rgi_neighb_merged_naming_convention.txt'
    text_file = open(rgi_naming_txt, 'r')
    rgi_list = text_file.readlines()
    ww_dir = '/mnt/lidar2/RGI_paper/worldwide/'
    shp_list = [os.path.join(ww_dir, rgi[:-1].split('rgi60')[0] + 'rgi60', 'cov',
                             'L1A_glacierized_extended_' + rgi[:-1].split('rgi60')[0] + 'rgi60',
                             'L1A_glacierized_extended_' + rgi[:-1].split('rgi60')[0] + 'rgi60.shp') for rgi in
                rgi_list]

    # parent directory where data will be sorted
    main_dir = '/mnt/lidar2/RGI_paper/test/'

    # maximum strip length
    strip_length = 3

    # sort by region first, rgi regions here
    sort_l1a_by_region(list_pulldir, shp_list, main_dir)

    # then loop for each region
    for rgi in rgi_list:
        rgi_dir = os.path.join(main_dir, rgi[:-1].split('rgi60')[0] + 'rgi60', 'aster_l1a')
        if os.path.exists(rgi_dir):
            # remove duplicates
            rm_duplic_l1a(rgi_dir)
            if os.path.exists(os.path.join(rgi_dir, 'no_stereoband')):
                rm_duplic_l1a(os.path.join(rgi_dir, 'no_stereoband'))

            # sort by strips
            sort_l1a_by_strip(rgi_dir, strip_length)

            # sort by utm zone
            sort_strip_by_utm(rgi_dir)
