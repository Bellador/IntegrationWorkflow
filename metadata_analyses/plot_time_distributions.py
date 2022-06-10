# Python 3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
    Parse dates from a csv file, the dates are given as colums "Year", "Month" and "Day". Plot the dates on a timeline with the earliest and latest date as bounds
"""

# Read the csv file
df = pd.read_csv("data/metadata_platform/ebird_observations_processed.csv", parse_dates=["Year", "Month", "Day"])

print(len(df))
df2 = pd.read_csv('data/metadata_platform/flickr_observations_processed.csv', delimiter=",")
print(len(df2))
df3 = pd.read_csv('data/metadata_platform/inaturalist_observations_processed.csv', delimiter=",")
print(len(df3))

# Parse the dates
df['Date'] = pd.to_datetime(df['Date'])
df2['DateTaken'] = pd.to_datetime(df2['DateTaken'])
df3['observed_on'] = pd.to_datetime(df3['observed_on'])

# Get the earliest and latest date
earliest_date = df['Date'].min()
latest_date = df['Date'].max()

# Create a color scale
color_scale = np.linspace(0, 1, 12)

# Create a color map
cmap = plt.cm.get_cmap('hsv')

# Plot the dates
for i in range(len(df)):
    plt.scatter(df.iloc[i]['Date'], 0, color=plt.cm.viridis(color_scale[df.iloc[i]['Date'].month - 1]))

# Let the timeline start in 2012
plt.xlim(earliest_date, latest_date)

# Show the plot
plt.show()

plt.hist(df2['DateTaken'].dt.year, bins=np.arange(min(df['Date'].dt.year), max(df['Date'].dt.year) + 2), color="orange", edgecolor='black', linewidth=0.8, alpha=0.7)
plt.hist(df['Date'].dt.year, bins=np.arange(min(df['Date'].dt.year), max(df['Date'].dt.year) + 2), edgecolor='black', linewidth=0.8)
plt.hist(df3['observed_on'].dt.year, bins=np.arange(min(df3['observed_on'].dt.year), max(df3['observed_on'].dt.year) + 2), color="green", edgecolor='black', linewidth=0.8, alpha=0.7)
plt.xticks(np.arange(min(df['Date'].dt.year) + 1, max(df['Date'].dt.year) + 1, 2) + 0.5, np.arange(min(df['Date'].dt.year) + 1, max(df['Date'].dt.year) + 2, 2))
# plt.show()
plt.legend(["Flickr", "eBird", "iNaturalist"], loc="upper left")
plt.savefig("figures/year_distribution.png")

plt.xlim(1, 13)

plt.hist(df2['DateTaken'].dt.month, bins=12, range=(1, 13), color="orange", edgecolor='black', linewidth=0.8, alpha=0.7)

plt.hist(df['Date'].dt.month, bins=12, range=(1, 13), color='#1f77b4', edgecolor='black', linewidth=0.8)
plt.hist(df3['observed_on'].dt.month, bins=np.arange(min(df3['observed_on'].dt.month), max(df3['observed_on'].dt.month) + 2), color="green", edgecolor='black', linewidth=0.8, alpha=0.7)
plt.xticks(np.arange(1, 13) + 0.5, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(["Flickr", "eBird", "iNaturalist"], loc="upper right")
# plt.show()
plt.savefig("figures/month_distribution.png")