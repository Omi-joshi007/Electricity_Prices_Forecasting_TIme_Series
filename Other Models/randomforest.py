import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

# Function to load data
@st.cache_data
def load_data(filepath):
    data = pd.read_csv(filepath)
    data['datetime'] = pd.to_datetime(data['date'].astype(str) + ' ' + data['time'].astype(str).str.zfill(4), format='%Y%m%d %H%M')
    data.set_index('datetime', inplace=True)
    return data[['price_act', 'electricity_demand', 'solar_generation', 'wind_generation', 'tempc']]

# Function to load actual price data
@st.cache_data
def load_actual_price_data(filepath):
    data = pd.read_csv(filepath)
    data['datetime'] = pd.to_datetime(data['date'].astype(str) + ' ' + data['time'].astype(str).str.zfill(4), format='%Y%m%d %H%M')
    data.set_index('datetime', inplace=True)
    return data[['price_act']]

# Function to prepare data for the model
@st.cache_data
def prepare_data_for_model(data, lag=24):
    X = pd.DataFrame()
    for i in range(lag):
        X[f'y_lag_{i+1}'] = data['price_act'].shift(i+1)
        X[f'electricity_demand_lag_{i+1}'] = data['electricity_demand'].shift(i+1)
        X[f'solar_generation_lag_{i+1}'] = data['solar_generation'].shift(i+1)
        X[f'wind_generation_lag_{i+1}'] = data['wind_generation'].shift(i+1)
        X[f'tempc_lag_{i+1}'] = data['tempc'].shift(i+1)
    
    # Adding rolling statistics
    X['rolling_mean_7'] = data['price_act'].rolling(window=7).mean().shift(1)
    X['rolling_mean_30'] = data['price_act'].rolling(window=30).mean().shift(1)
    X['rolling_std_7'] = data['price_act'].rolling(window=7).std().shift(1)
    X['rolling_std_30'] = data['price_act'].rolling(window=30).std().shift(1)
    X['rolling_min_7'] = data['price_act'].rolling(window=7).min().shift(1)
    X['rolling_min_30'] = data['price_act'].rolling(window=30).min().shift(1)
    X['rolling_max_7'] = data['price_act'].rolling(window=7).max().shift(1)
    X['rolling_max_30'] = data['price_act'].rolling(window=30).max().shift(1)
    X['rolling_median_7'] = data['price_act'].rolling(window=7).median().shift(1)
    X['rolling_median_30'] = data['price_act'].rolling(window=30).median().shift(1)
    X['rolling_skew_7'] = data['price_act'].rolling(window=7).skew().shift(1)
    X['rolling_skew_30'] = data['price_act'].rolling(window=30).skew().shift(1)
    X['rolling_kurt_7'] = data['price_act'].rolling(window=7).kurt().shift(1)
    X['rolling_kurt_30'] = data['price_act'].rolling(window=30).kurt().shift(1)

    # Adding seasonal components
    X['hour'] = data.index.hour
    X['day_of_week'] = data.index.dayofweek
    X['month'] = data.index.month
    X['quarter'] = data.index.quarter
    X['day_of_year'] = data.index.dayofyear
    
    X = X.dropna()
    
    y = data['price_act'][lag:]
    y = y.iloc[:X.shape[0]]  # Ensure y has the same length as X
    return X, y

# Function to generate future data for prediction
@st.cache_data
def generate_future_data(data, periods=3*30*24, lag=24):
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=periods+1, freq='H')[1:]
    
    future_data = pd.DataFrame(index=future_dates)
    future_data['electricity_demand'] = np.nan  # Fill with reasonable future values or use historical averages
    future_data['solar_generation'] = np.nan  # Fill with reasonable future values or use historical averages
    future_data['wind_generation'] = np.nan  # Fill with reasonable future values or use historical averages
    future_data['tempc'] = np.nan  # Fill with reasonable future values or use historical averages
    
    full_data = pd.concat([data, future_data])
    X_future, _ = prepare_data_for_model(full_data, lag=lag)
    
    X_future = X_future.iloc[-periods:]  # Select the future part of the data
    return X_future

# Function to standardize the features
@st.cache_data
def standardize_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# Function for feature selection
@st.cache_data
def feature_selection(X_scaled, y):
    tree_model = ExtraTreesRegressor(n_estimators=100)
    tree_model.fit(X_scaled, y)
    model = SelectFromModel(tree_model, prefit=True)
    X_selected_tree = model.transform(X_scaled)
    return X_selected_tree, model

# Function to train the model
@st.cache_resource
def train_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestRegressor(n_jobs=-1, random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Function to make parallel predictions
def parallel_predict(model, data, batch_size=1000):
    n_batches = (len(data) + batch_size - 1) // batch_size
    results = Parallel(n_jobs=-1)(delayed(model.predict)(data[i * batch_size:(i + 1) * batch_size]) for i in range(n_batches))
    return np.concatenate(results, axis=0)

# Streamlit application
st.title('Electricity Price Prediction')

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type='csv')

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write('Data Loaded Successfully')
    st.write(data.head())

    # Define lag period
    lag = 24

    # Prepare data
    X, y = prepare_data_for_model(data, lag=lag)
    
    # Standardize the features
    X_scaled, scaler = standardize_features(X)

    # Feature selection using tree-based model
    X_selected_tree, model = feature_selection(X_scaled, y)
    
    # Train/validation/test split
    train_size = int(len(X_selected_tree) * 0.7)
    val_size = int(len(X_selected_tree) * 0.1)
    test_size = len(X_selected_tree) - train_size - val_size
    
    X_train_tree = X_selected_tree[:train_size]
    y_train = y[:train_size]
    
    X_val_tree = X_selected_tree[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test_tree = X_selected_tree[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    data_index_val = data.index[lag + train_size:lag + train_size + val_size]
    data_index_test = data.index[lag + train_size + val_size:lag + train_size + val_size + test_size]

    # Train the model
    best_rf = train_model(X_train_tree, y_train)
    
    y_pred_train = best_rf.predict(X_train_tree)
    y_pred_val = best_rf.predict(X_val_tree)
    y_pred_test = best_rf.predict(X_test_tree)
    
    # RMSE
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    st.write(f'RMSE on Training Set: {rmse_train}')
    st.write(f'RMSE on Validation Set: {rmse_val}')
    st.write(f'RMSE on Test Set: {rmse_test}')

    # Cross-validation RMSE using TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(best_rf, X_selected_tree, y, cv=tscv, scoring='neg_mean_squared_error')
    cv_rmse_scores = np.sqrt(-cv_scores)
    mean_cv_rmse = np.mean(cv_rmse_scores)
    std_cv_rmse = np.std(cv_rmse_scores)
    
    st.write(f'Cross-Validation RMSE: {mean_cv_rmse}')
    st.write(f'Standard Deviation of Cross-Validation RMSE: {std_cv_rmse}')

    # Plot actual vs predicted values for RandomForest model
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data_index_test, y_test, label='Actual', color='b')
    ax.plot(data_index_test, y_pred_test, label='Predicted (RandomForest)', color='r')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title('Actual vs Predicted Electricity Prices (RandomForest with Grid Search)')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Residuals
    residuals = y_test - y_pred_test
    
    # Downsample data for plotting
    sample_size = 1000  # Adjust based on your data size and available memory
    if len(residuals) > sample_size:
        residuals_sample = residuals.sample(sample_size, random_state=42)
    else:
        residuals_sample = residuals

    # Plot residuals
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    sns.scatterplot(x=y_test, y=residuals, ax=ax[0])
    ax[0].set_xlabel('Actual Price')
    ax[0].set_ylabel('Residual')
    ax[0].set_title('Residuals vs Actual Price')

    sns.histplot(residuals_sample, kde=False, ax=ax[1])
    ax[1].set_xlabel('Residual')
    ax[1].set_ylabel('Count')
    ax[1].set_title('Residual Distribution')

    plt.tight_layout()
    st.pyplot(fig)
    
    # Predictions for the future
    future_days = st.number_input("How many days do you want to predict? ", min_value=1, max_value=365, value=90)
    future_periods = future_days * 24
    X_future = generate_future_data(data, periods=future_periods, lag=lag)
    X_future_scaled = scaler.transform(X_future)
    X_future_selected = model.transform(X_future_scaled)
    
    future_predictions = parallel_predict(best_rf, X_future_selected)
    
    future_dates = pd.date_range(start=data.index[-1], periods=future_periods+1, freq='H')[1:]
    future_df = pd.DataFrame({'datetime': future_dates, 'predicted_price': future_predictions})
    future_df.set_index('datetime', inplace=True)
    
    # Plot future predictions
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(future_df.index, future_df['predicted_price'], label='Predicted Price', color='r')
    ax.set_xlabel('Time')
    ax.set_ylabel('Predicted Price')
    ax.set_title('Future Predicted Electricity Prices')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Add download button for the predictions
    csv = future_df.to_csv().encode('utf-8')
    download_button_clicked = st.download_button(
    label="Download predictions as CSV",
    data=csv,
    file_name='future_predictions.csv',
    mime='text/csv',
    )

    # Upload CSV file for actual prices
    st.title('Upload Actual Prices CSV File')
    actual_price_file = st.file_uploader("Upload your actual prices CSV file", type='csv')
    
    if actual_price_file is not None:
        actual_price_data = load_actual_price_data(actual_price_file)
        st.write('Actual Price Data Loaded Successfully')
        st.write(actual_price_data.head())
        
        # Plot future predictions and actual prices
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(future_df.index, future_df['predicted_price'], label='Predicted Price', color='r')
        ax.plot(actual_price_data.index, actual_price_data['price_act'], label='Actual Price', color='b')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.set_title('Future Predicted vs Actual Electricity Prices')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
        actual_future_prices = actual_price_data['price_act'].reindex(future_df.index).dropna()
        predicted_future_prices = future_df['predicted_price'].loc[actual_future_prices.index]
        rmse_future = np.sqrt(mean_squared_error(actual_future_prices, predicted_future_prices))
        st.write(f'RMSE between predicted and actual future prices: {rmse_future}')
