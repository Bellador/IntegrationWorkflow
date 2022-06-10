import numpy as np
import pandas as pd

df1 = pd.read_csv("data/metadata_platform/ebird_observations_processed.csv", delimiter=",")
print("Ebird", df1.shape)
df1["Recordist"] = pd.factorize(df1["Recordist"])[0]
df2 = pd.read_csv("data/metadata_platform/flickr_observations_processed.csv", delimiter=",")
print("Flickr", df2.shape)
df2["UserID"] = pd.factorize(df2["UserID"])[0]
df3 = pd.read_csv("data/metadata_platform/inaturalist_observations_processed.csv", delimiter=",")
print("iNaturalist", df3.shape)
df3["user_id"] = pd.factorize(df3["user_id"])[0]

# Give a sorted list of observers and their respective counts
vc1 = df1['Recordist'].value_counts()
print("ebird")
print(vc1)

# Give a sorted list of observers and their respective counts
vc2 = df2['UserID'].value_counts()
print("Flickr")
print(vc2)

vc3 = df3['user_id'].value_counts()
print("iNaturalist")
print(vc3)

print(df1['Recordist'].nunique())
print(df2['UserID'].nunique())
print(df3['user_id'].nunique())

# Plot both distributions into a shared histogram via matplotlib. The histograms are sorted from highest to lowest number of contributions.
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3)

axs[1].bar(range(len(vc3)), vc3, color="#2ca02c", width=1.0, linewidth=0.1)
axs[0].bar(range(len(vc1)), vc1, color='#1f77b4', width=1.0, linewidth=0.1)
axs[2].bar(range(len(vc2)), vc2, color="#ff7f0e", width=1.0, linewidth=0.1)
#axs.bar(range(len(vc2)), vc2, color='#ff7f0e', width=1.0, linewidth=0.1)
#zero_array = np.zeros_like(vc2)
#zero_array[:len(vc1)] = vc1
#axs.bar(range(len(vc2)), zero_array, color="#1f77b4", width=1.0, linewidth=0.1, bottom=vc2)
#added_counts = zero_array + vc2
#zero_array = np.zeros_like(vc2)
#zero_array[:len(vc3)] = vc3
#axs.bar(range(len(vc2)), zero_array, color="#2ca02c", width=1.0, linewidth=0.1, bottom=added_counts)
# plt.xticks([])
fig.supxlabel("Unique users per platform")
fig.supylabel("# of contrubutions per user")
# axs.legend(["Extracted from Flickr (2260 total images)", "eBird (271 total images)", "iNaturalist (199 total images)"], loc="upper right")
axs[1].legend(["iNaturalist (199 total images)"], loc="upper right")
axs[0].legend(["eBird (271 total images)"], loc="upper right")
axs[2].legend(["Extracted from Flickr (2260 total images)"], loc="upper right")
fig.subplots_adjust(hspace=0.25)
plt.savefig("figures/contrib_hist_stacked.png")