import collections

import pandas as pd

df1 = pd.read_csv("data/metadata_platform/ebird_observations.csv", delimiter=",")
df2 = pd.read_csv("data/metadata_platform/TP_red_kite_chilterns_big_text_plus_visual_metadata.csv", delimiter=",")
df3 = pd.read_csv("data/metadata_platform/inaturalist_observations.csv", delimiter=",")

df_ebird_old = pd.read_csv("data/metadata_platform/old_ebird_observations_full.csv", delimiter=";")
df_ebird_old = df_ebird_old.drop('file_name', axis=1)

a = df1["ML Catalog Number"].tolist()
b = df_ebird_old["ML Catalog Number"].tolist()

df1 = df1.reset_index(drop=True)
df_ebird_old = df_ebird_old.reset_index(drop=True)

missing_entries = set(b) - set(a)
for entry in missing_entries:
    pd.concat([df1, df_ebird_old[df_ebird_old["ML Catalog Number"] == entry]], axis=1)
    print(df_ebird_old[df_ebird_old["ML Catalog Number"] == entry][["ML Catalog Number", "Recordist", "Latitude", "Longitude", "Date"]])

df1 = df1.set_index("ML Catalog Number")
df1 = df1.reindex(index=df_ebird_old["ML Catalog Number"])
df1 = df1.reset_index()
df1["user_id"] = df1["Recordist"].astype(str)
df2["user_id"] = df2["UserID"].astype(str)
df3["user_id"] = df3["user_id"].astype(str)

df1["lat"] = df1["Latitude"].astype(float)
df2["lat"] = df2["Latitude"].astype(float)
df3["lat"] = df3["latitude"].astype(float)

df1["long"] = df1["Longitude"].astype(float)
df2["long"] = df2["Longitude"].astype(float)
df3["long"] = df3["longitude"].astype(float)

df1['observe_timestamp'] = pd.to_datetime(df1['Date'])
df2['observe_timestamp'] = pd.to_datetime(df2['DateTaken'])
df3['observe_timestamp'] = pd.to_datetime(df3['observed_on'])

df1['filename'] = "photo_" + df1.index.astype(str) + ".png"
df2['filename'] =  df2["PhotoID"].astype(str) + ".jpg"
df3['filename'] = "photo_" + df3.index.astype(str) + ".png"

print(df1.head())
print(df2.head())
print(df3.head())

df1.to_csv("data/metadata_platform/ebird_observations_processed.csv", sep=",", index=False)
df2.to_csv("data/metadata_platform/flickr_observations_processed.csv", sep=",", index=False)
df3.to_csv("data/metadata_platform/inaturalist_observations_processed.csv", sep=",", index=False)