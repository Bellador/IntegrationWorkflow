import datetime
import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

files = ['data/metadata_platform/inaturalist_observations_processed.csv',
         'data/metadata_platform/ebird_observations_processed.csv',
         'data/metadata_platform/flickr_observations_processed.csv']

transformer = Transformer.from_crs('epsg:4326', 'epsg:32630', always_xy=True)

for file in files:
    images = gpd.read_file(file,
                           X_POSSIBLE_NAMES='long',
                           Y_POSSIBLE_NAMES='lat')
    # transform coordinates to projected CRS
    images = pd.concat([images,
                        images.apply(
                            lambda row: dict(
                                zip(["long_trans", "lat_trans"], transformer.transform(xx=row.long, yy=row.lat))),
                            axis=1,
                            result_type='expand')], axis='columns')
    # map ts string to datetime object
    images["observe_timestamp"] = pd.to_datetime(images["observe_timestamp"])
    # only flickr contains timestamps, all others only feature dates: round to date
    images["observe_timestamp"] = images["observe_timestamp"].dt.date

    # cluster by observe_timestamp
    images = images.sort_values(by=['user_id',
                                    'observe_timestamp'])
    images["ts_diff"] = images.groupby(by='user_id')["observe_timestamp"].diff().fillna(datetime.timedelta())
    images["new_ts_session"] = images["ts_diff"] >= datetime.timedelta(days=1)
    images["user_ts_session"] = images.groupby(by='user_id')["new_ts_session"].cumsum()

    # geo-cluster
    # Note: Clusters may spatially propagate (e.g. one phone every 500m creates one long cluster). The number of
    # duplicates is correct but the number of clusters may be an underestimation (and therefore the session length
    # an over estimation).
    # A spatial join using buffer would be more exact by assigning images to multiple clusters. This would also be
    # better suited for visualisation.
    images["geo_clust"] = images.groupby(by=['user_id', 'user_ts_session'], as_index=False, group_keys=False).apply(
        lambda session: pd.Series(DBSCAN(eps=500, min_samples=2).fit_predict(session[["long_trans", "lat_trans"]]),
                                  index=session.index))
    images["geo_clust"] = images["geo_clust"].where(images["geo_clust"] != -1, -images.index)

    images = images.sort_values(by=['user_id',
                                    'user_ts_session',
                                    'geo_clust'])
    images["geo_diff"] = images.groupby(by=['user_id', 'user_ts_session'])["geo_clust"].diff().fillna(0)
    images["new_geo_loc"] = images["geo_diff"] != 0
    images["user_geo_loc"] = images.groupby(by=['user_id', 'user_ts_session'])["new_geo_loc"].cumsum()

    duplicates = images.groupby(by=["user_id",
                                    "user_ts_session",
                                    "user_geo_loc"]).filter(lambda x: len(x) > 1)

    duplicates["session_id"] = "d_" + duplicates["user_id"].astype(str) + "_" + duplicates["user_ts_session"].astype(
        str) + "_" + duplicates["user_geo_loc"].astype(str)

    # write this tag to image files and verify manually
    duplicates[["filename", "session_id"]].to_csv(file + '_sessions.csv')