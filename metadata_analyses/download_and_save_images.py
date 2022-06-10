# NOTE: this file needs updated directories!


import collections
from io import StringIO

import pandas as pd
from PIL import Image
import urllib
import requests
import datetime
import sys
import csv
import os

### 1. Download results from respective APIs

## For eBird, there is not possiblity (that i know of) to download inside a bounding box, so we first download for th whole of england
# Unfortunately, downloading more than 31 entries requires login/auth. So just download  the csv manually by logging in
# Browser URL:
# https://ebird.org/media/catalog?taxonCode=redkit1&view=List&mediaType=p&sort=rating_rank_desc&region=England,%20United%20Kingdom%20(GB)&regionCode=GB-ENG
with requests.Session() as s:
    # result = s.get("https://ebird.org/media/catalog.csv?taxonCode=redkit1&view=List&mediaType=p&sort=rating_rank_desc&regionCode=GB-ENG")

    # decoded_csv = result.content.decode("utf-8")
    bb = [[51.47025, -1.152104], [51.890047, -0.432355]]
    # df = pd.read_csv(StringIO(decoded_csv), delimiter=",")
    df = pd.read_csv("../../general_data_insights/raw_data/ML_2021-11-29T09-06_redkit1_photo_GB-ENG.csv")
    df['Date'] = pd.to_datetime(df['Date'])

    ebird_data = df.loc[(df['Latitude'] >= bb[0][0]) & (df['Latitude'] <= bb[1][0]) & (df['Longitude'] >= bb[0][1])
                         & (df['Longitude'] <= bb[1][1]) & (df['Date'] <= pd.Timestamp(2021, 7, 31))]
    print("save ebird data")
    ebird_data.to_csv("ebird_observations_.csv")

exit(0)


## For Flickr, i use the metadata_platform file that contains the output of our workflow
flickr_data = pd.read_csv("../data/metadata_platform/flickr_observations.csv", delimiter=";")

## For other purposes, you might be interested e.g. in querying the Chiltern Hills Area based on tag etc. which is displayed here
try:
    import flickrapi
    from metadata_analyses.general_data_insights import api_creds
except ImportError:
    print("Did not found Flickrapi or an API credentials file")
    api_creds = collections.namedtuple(api_key="", api_secret="")

try:
    flickr = flickrapi.FlickrAPI(api_creds.api_key, api_creds.api_secret, cache=True, format='parsed-json')
    bbox = [[46.88041, 8.3924], [47.24894, 8.80186]]
    results = flickr.photos.search(tags="Red Kite", has_geo=1,
                                   bbox=",".join([str(bbox[0][1]), str(bbox[0][0]), str(bbox[1][1]), str(bbox[1][0])]))[
        "photos"]["total"]
except:
    print("Something wrong with the Flickr API call")

## For iNaturalist, i have not found a fully automated workflow:
# My current query for the chiltern hills area in iNaturalist (just type into browser)
# https://www.inaturalist.org/observations?d1=2000-01-01&has%5B%5D=photos&nelat=51.890047&nelng=-0.432355&photos&place_id=any&subview=map&swlat=51.47025&swlng=-1.152104&taxon_id=5267
# You can then access the data download via the Filter > Herunterladen/Download > Prepare Data

# ebird_data = pd.read_csv("old_ebird_observations_full.csv")
inat_data = pd.read_csv("../data/metadata_platform/old_inaturalist_observations.csv", delimiter=";")

flickr_data = pd.read_csv("flickr_observations.csv", delimiter=";")

os.makedirs("ebird", exist_ok=True)
for i, eb in ebird_data.iterrows():
    catalog_num = eb["ML Catalog Number"]

    url = "https://cdn.download.ams.birds.cornell.edu/api/v1/asset/" + str(catalog_num)
    if isinstance(url, str):
        try:
            urllib.request.urlretrieve(url, os.path.join("ebird", "photo_" + str(catalog_num) + ".png"))
        except:
            print("that didnt work: ", url)

os.makedirs("flickr", exist_ok=True)
for i, fd in flickr_data.iterrows():
    img_url = fd["img_url_l"]

    if isinstance(img_url, str):
        try:
            urllib.request.urlretrieve(img_url, os.path.join("flickr", "photo_" + str(i) + ".png"))
        except:
            print("that didnt work: ", img_url)

os.makedirs("inaturalist", exist_ok=True)
for i, id in inat_data.iterrows():
    img_url = id["image_url"]

    if isinstance(img_url, str):
        try:
            urllib.request.urlretrieve(img_url, os.path.join("inaturalist", "photo_" + str(i) + ".png"))
        except:
            print("that didnt work: ", img_url)
