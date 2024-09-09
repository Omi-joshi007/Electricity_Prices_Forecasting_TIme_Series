import numpy as np
import pandas as pd

# float display setting
pd.set_option("display.float_format", lambda x: "%.2f" % x)

# a list of filenames
files = ['tokyo_electricity_prices', 'tokyo_electricity_demand', 'tokyo_solar_generation', 'tokyo_wind_generation', 'tokyo_weather']
dataframes = {}

# load files into a dictionary for iteration and summary the data structure
for file in files:
    dataframes[file] = pd.read_csv(f"{file}.csv", header=0)

# merge data files together
df_final = dataframes['tokyo_electricity_prices']
for name, df in dataframes.items():
    df.replace(-99, np.nan, inplace=True)
    df = df.interpolate(method='linear', axis=0, limit_direction='both')
    if name != 'tokyo_electricity_prices':
        df_final = pd.merge(df_final, df, on=['date', 'time'], how='left')
        df_final.ffill(inplace = True)
        
df_final.columns = ['date', 'time', 'price_act', 'electricity_demand', 'solar_generation', 'wind_generation', 'tempc', 'cloud8', 'windmps', 'wdir', 'rainmm', 'humid', 'radjcm2']

# combine date and time
df_final['time'] = df_final['time'].apply(lambda x: f"{x:04d}")
df_final['datetime'] = pd.to_datetime(df_final['date'].astype(str) + ' ' + df_final['time'].str[:2] + ':' + df_final['time'].str[2:], format='%Y%m%d %H:%M')
df_final.set_index('datetime', inplace=True)
df_final.drop(columns=['time','date'], inplace=True)

# save dataset locally
df_final.to_csv('tokyo_electricity_final_dataset_half_hourly.csv')