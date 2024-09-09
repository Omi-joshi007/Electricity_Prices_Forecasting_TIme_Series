import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings

warnings.filterwarnings("ignore")


file_path = 'C:\\Users\\Administrator\\Desktop\\codes\\tokyo_electricity_final_dataset_half_hourly.csv'
data = pd.read_csv(file_path)


print(data.info())
print(data.head())


data['timestamp'] = pd.to_datetime(data['date'].astype(str) + data['time'].astype(str).str.zfill(4), format='%Y%m%d%H%M')


data.set_index('timestamp', inplace=True)


data.replace(-99, np.nan, inplace=True)


data.interpolate(method='linear', inplace=True)


print(data.isna().sum())


price_series = data['price_act']


train_size = int(len(price_series) * 0.8)
train, test = price_series[:train_size], price_series[train_size:]


model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()


print(model_fit.summary())


forecast = model_fit.forecast(steps=len(test))


forecast_series = pd.Series(forecast, index=test.index)


rmse = sqrt(mean_squared_error(test, forecast_series))
print(f'RMSE: {rmse}')
