import pandas as pd
import numpy as np


def preprocess_data(df, date_col, time_col, freq='30min'):
    """Preprocess the data by converting date and time columns to datetime and setting datetime as index"""
    df['datetime'] = pd.to_datetime(df[date_col].astype(
        str) + df[time_col].astype(str).str.zfill(4), format='%Y%m%d%H%M')
    df.set_index('datetime', inplace=True)
    df = df.resample(freq).asfreq().interpolate(method='linear')
    df.drop(columns=[date_col, time_col], inplace=True)
    return df


# load data
demand_data = pd.read_csv('tokyo_june_electricity_demand.csv')
solar_data = pd.read_csv('tokyo_june_solar_generation.csv')
weather_data = pd.read_csv('tokyo_june_weather.csv')
wind_data = pd.read_csv('tokyo_june_wind_generation.csv')

# preprocess data
demand_data = preprocess_data(demand_data, 'date', 'time')
solar_data = preprocess_data(solar_data, 'date', 'time')
weather_data = preprocess_data(weather_data, 'date', 'time')
wind_data = preprocess_data(wind_data, 'date', 'time')

# make sure name of columns are same as train data
demand_data.columns = ['electricity_demand']
solar_data.columns = ['solar_generation']
weather_data.columns = ['tempc', 'cloud8',
                        'windmps', 'wdir', 'rainmm', 'humid', 'radjcm2']
wind_data.columns = ['wind_generation']

# combine data
data = demand_data.join(solar_data, how='outer')
data = data.join(weather_data, how='outer')
data = data.join(wind_data, how='outer')

# print first few rows of data
print(data.head())

# change -99 to nan
data.replace(-99, np.nan, inplace=True)

# fill missing values using linear interpolation
data = data.interpolate(method='linear')

# print first few rows of data
print(data.head())

# save data
data.to_csv('tokyo_june_combined.csv')

print('Data was merged and preprocessed and saved as tokyo_june_combined.csv')
