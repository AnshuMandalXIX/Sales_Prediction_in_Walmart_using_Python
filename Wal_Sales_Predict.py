import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf
import warnings
warnings.filterwarnings('ignore')
# Load data from CSV file
df = pd.read_csv('Walmart.csv', parse_dates=True,usecols=["Date","Weekly_Sales"])
df.dropna()

# Split the data into train and test sets
train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]

# Fit the ARIMA model
model = ARIMA(train['Weekly_Sales'], order=(2, 1, 2))
model_fit = model.fit()

# Make predictions
predictions = model_fit.forecast(steps=len(test))

# Compute accuracy metrics
mse = mean_squared_error(test['Weekly_Sales'], predictions)
rmse = mse ** 0.5
mape = abs((test['Weekly_Sales'] - predictions) / test['Weekly_Sales']).mean() * 100

# Plot actual vs predicted values
plt.plot(test.index, test['Weekly_Sales'], label='Actual')
plt.plot(test.index, predictions, label='Predicted')
plt.legend()
plt.show()

# Print accuracy metrics and model results
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAPE: {mape:.2f}%')

#Create ANOTHER Training and Test
train_2 = df.Weekly_Sales[:train_size]
test_2 = df.Weekly_Sales[train_size:]

# Build Model
model = ARIMA(train_2, order=(1, 1, 1)) 
fitted = model.fit() 
# Forecast
fc = fitted.predict()
# Make as pandas series
fc_series = pd.Series(fc, index=test_2.index)
lower_series = pd.Series(index=test_2.index)
upper_series = pd.Series(index=test_2.index)

# Ploting the Graphs
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train_2, label='training')
plt.plot(test_2, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()
