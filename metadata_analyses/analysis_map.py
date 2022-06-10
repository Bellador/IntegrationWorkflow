import folium
from folium.plugins import HeatMap
import pandas as pd
from folium import FeatureGroup
import time

m = folium.Map(location=[51.68014, -0.7922], zoom_start=11, attr='mapbox', tiles="https://api.mapbox.com/styles/v1/ymetz/cku2mktnt506j18lhmsq28aj7/tiles/256/{z}/{x}/{y}?access_token=pk.eyJ1IjoieW1ldHoiLCJhIjoiY2t0MWVhMDhsMDlpeDJ1bzhvYW82YmtseSJ9.NzfDheBhnCzCcXuw1Sq_Lg")
ebird_data = pd.read_csv("data/metadata_platform/ebird_observations.csv", delimiter=",")
print("#Elements Ebird data", len(ebird_data))

m.add_child(folium.features.Choropleth("data/map_data/chilltern_hills.geojson", fill_opacity=0.35, line_color="#46b03a", fill_color="#adf542"))

flickr_data = pd.read_csv("data/metadata_platform/flickr_observations_processed.csv", delimiter=",")
dates_as_string = flickr_data['DateTaken']
dates_list = []

inat_data = pd.read_csv("data/metadata_platform/inaturalist_observations_processed.csv", delimiter=",")
inat_dates_as_string = inat_data['observed_on_string']
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(flickr_data.sample(n=5))


folium.Rectangle(
    bounds=[[51.47025, -1.152104], [51.890047, -0.432355]],
    popup="The Waterfront",
    color="#c9c9c9",
    fill=False,
).add_to(m)
print("#Elements Flickr data", len(flickr_data))

ebird = FeatureGroup(name='EBird')
flickr = FeatureGroup(name='Flickr')
inat = FeatureGroup(name="inaturalist")

for _, row in ebird_data.iterrows():
    folium.Marker(
        location=(row['Latitude'], row['Longitude']),
        popup=row['Scientific Name']+" "+str(row['Year'])+"-"+str(row["Month"])+"-"+str(row["Day"]),
        icon=folium.Icon(color="orange", icon="fa-picture-o"),
        radius=15,
        # fill_color="#1f77b4"
    ).add_to(ebird)
for _, row in inat_data.iterrows():
    folium.Marker(
        location=(row['latitude'], row['longitude']),
        popup=str(row['scientific_name'])+"\n"+row['observed_on_string']+"\n<a href="+row['image_url']+">Image</a>",
        icon=folium.Icon(color="darkblue", icon="fa-picture-o"),
        radius=15,
        # fill_color="#f97306"
    ).add_to(inat)
# HeatMap(
#     data=list(zip(ebird_data['Latitude'], ebird_data['Longitude'])),
#     radius=70,
#     blur=30,
#     gradient={0.0: '#2171b5', 0.1: '#1e6aad', 0.2: '#1c64a6', 0.3: '#195d9e', 0.4: '#175797', 0.5: '#145090', 0.6: '#124a88', 0.7: '#0f4381', 0.8: '#0d3d79', 0.9: '#0a3672', 1.0: '#08306b'}
# ).add_to(ebird)

for _, row in flickr_data.iterrows():
    folium.Marker(
        location=(row['Latitude'], row['Longitude']),
        popup=str(row['Title'])+"\n"+row['DateTaken']+"\n<a href="+str(row['LargestPhotoURL'])+">Image</a>",
        icon=folium.Icon(color="green", icon="fa-photo-o"),
        radius=15,
        # fill_color="#f97306"
    ).add_to(flickr)

# HeatMap(
#     data=[[float(".".join(x.split(".")[:2])), float(".".join(y.split(".")[:2]))] for x,y in  list(zip(flickr_data['lat'], flickr_data['lng']))],
#     radius=60,
#     blur=30,
#     gradient={0.0: '#fee6ce', 0.1: '#f1d2b9', 0.2: '#e4bfa5', 0.3: '#d7ac91', 0.4: '#cb997d', 0.5: '#be8669', 0.6: '#b17354', 0.7: '#a56040', 0.8: '#984d2c', 0.9: '#8b3a18', 1.0: '#7f2704'}
# ).add_to(flickr)

m.add_child(ebird)
m.add_child(flickr)
m.add_child(inat)

m.add_child(folium.map.LayerControl())

m.save("figures/map_overview.html")