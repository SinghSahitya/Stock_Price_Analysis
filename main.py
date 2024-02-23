import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from warnings import filterwarnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

filterwarnings('ignore')

# Load data and preprocess
df = pd.read_csv('BAJFINANCE.csv')
df.set_index('Date', inplace=True)
df.dropna(inplace=True)

# Create lagged features
lag_features = ['High', 'Low', 'Volume', 'Turnover', 'Trades']
window1 = 3
window2 = 7
for feature in lag_features:
    df[feature+'rolling_mean_3'] = df[feature].rolling(window=window1).mean()
    df[feature+'rolling_mean_7'] = df[feature].rolling(window=window2).mean()

for feature in lag_features:
    df[feature+'rolling_std_3'] = df[feature].rolling(window=window1).std()
    df[feature+'rolling_std_7'] = df[feature].rolling(window=window2).std()

df.dropna(inplace=True)

ind_features=['Highrolling_mean_3', 'Highrolling_mean_7',
       'Lowrolling_mean_3', 'Lowrolling_mean_7', 'Volumerolling_mean_3',
       'Volumerolling_mean_7', 'Turnoverrolling_mean_3',
       'Turnoverrolling_mean_7', 'Tradesrolling_mean_3',
       'Tradesrolling_mean_7', 'Highrolling_std_3', 'Highrolling_std_7',
       'Lowrolling_std_3', 'Lowrolling_std_7', 'Volumerolling_std_3',
       'Volumerolling_std_7', 'Turnoverrolling_std_3', 'Turnoverrolling_std_7',
       'Tradesrolling_std_3', 'Tradesrolling_std_7']

# Split into training and testing data
training_data=df[0:1800]
test_data=df[1800:]

# using ARIMA model for prediction
model=auto_arima(training_data['VWAP'], training_data[ind_features],trace=True)
# Fit model
model.fit(training_data['VWAP'],training_data[ind_features])

# dumping model in pickle file
with open('stock_arima.pickle', 'wb') as file:
    pickle.dump(model, file)

with open('stock_arima.pickle', 'rb') as model:
    arima_model = pickle.load(model)
    
    forecast=arima_model.predict(len(test_data),test_data[ind_features])

    # Assign forecast to testing data
    forecast.index = test_data.index
    test_data['Forecast_ARIMA']=forecast

    test_data[['VWAP', 'Forecast_ARIMA']].plot(figsize=(14,7))
    plt.show()

    print('Mean Squared Error: ',mean_squared_error(test_data['VWAP'],test_data['Forecast_ARIMA']))
    print('Mean Absolute Error: ', mean_absolute_error(test_data['VWAP'],test_data['Forecast_ARIMA']))