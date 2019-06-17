import matplotlib.pyplot as plt
import mplleaflet
import pandas as pd
import numpy as np

def leaflet_plot_stations(binsize, hashid):

    df = pd.read_csv('data/C2A2_data/BinSize_d{}.csv'.format(binsize))

    station_locations_by_hash = df[df['hash'] == hashid]

    lons = station_locations_by_hash['LONGITUDE'].tolist()
    lats = station_locations_by_hash['LATITUDE'].tolist()

    plt.figure(figsize=(8,8))

    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)

    return mplleaflet.display()

leaflet_plot_stations(400,'fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89')

df = pd.read_csv('data/C2A2_data/BinnedCsvs_d400/fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv')

## Watch out for leap days. It is reasonable to remove these points from
# the dataset for the purpose of this visualization.

# 1. Make sure date is between 2005 ~ 2015 and remove na
df = df[(df['Date'] > '2004') & (df['Date'] < '2016')]
df = df.dropna()

# 2. Remove leap days (2/29)
df = df[~((pd.DatetimeIndex(df['Date']).month == 2) & (pd.DatetimeIndex(df['Date']).day == 29))]

# 3. Make Month-Day Column
df['Year'] = pd.DatetimeIndex(df['Date']).year
df['Month-Day'] = pd.to_datetime(df['Date']).dt.strftime('%m-%d')

# 4.Find the broken point in 2015
df2015 = df[df['Year'] ==2015]
df2015_tmax = df2015.groupby('Month-Day')['Data_Value'].max()
df2015_tmax = df2015_tmax.reset_index()
df2015_tmin = df2015.groupby('Month-Day')['Data_Value'].min()
df2015_tmin = df2015_tmin.reset_index()

df_copy = df[df['Year'] < 2015]
df_tmax = df_copy.groupby('Month-Day')['Data_Value'].max()
df_tmax = df_tmax.reset_index()
df_tmin = df_copy.groupby('Month-Day')['Data_Value'].min()
df_tmin = df_tmin.reset_index()
max_broken = np.where(df2015_tmax['Data_Value'] > df_tmax['Data_Value'])[0]
min_broken = np.where(df2015_tmin['Data_Value'] < df_tmin['Data_Value'])[0]

df2015_tmax = df2015_tmax.set_index('Month-Day')
df2015_tmin = df2015_tmin.set_index('Month-Day')
df_tmax = df_tmax.set_index('Month-Day')
df_tmin = df_tmin.set_index('Month-Day')

#print(df_tmax)
# 5. Overlay a scatter of the 2015 data for any points 
# (highs and lows) for which the ten year record (2005-2014)
# record high or record low was broken in 2015.
# 5. Set the xlabel, ylabel, and title for graph
plt.figure()
plt.xlabel('Day')
plt.ylabel('Temperature in Celcius')
plt.title('Maximum and Minimum Temperature of Day during 2005-2015')

# 6. Set the x-axis values
plt.xticks(range(0, len(df_tmin), 20), df_tmin.index[range(0, len(df_tmin), 20)], rotation = '50')

# 7. Plot for the given value from 2005~2014
plt.plot(df_tmax.values/10, color = 'r')
plt.plot(df_tmin.values/10, color = 'b')

# 8. Scatter plot for value for 2015
plt.scatter(max_broken, df2015_tmax.iloc[max_broken]/10, s= 20, c='black', label= 'Broken Maximum')
plt.scatter(min_broken, df2015_tmin.iloc[min_broken]/10, s= 20, c='purple', label= 'Broken Minimum')

# 9. Now, create the legend
plt.legend(['Max (2005~2014)', 'Min (2005~2014)', "Max Broken in 2015", "Min Broken in 2015"])


# 10. Fill the area between min and max
plt.gca().fill_between(range(len(df_tmin)), df_tmin['Data_Value']/10,
                      df_tmax['Data_Value']/10, color="green", alpha= 0.4)

plt.show()
