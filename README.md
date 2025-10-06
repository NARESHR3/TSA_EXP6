# Ex.No: 6               HOLT WINTERS METHOD
### Date: 



### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
# Importing necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
data = pd.read_csv('tsa.csv', parse_dates=['Date'], index_col='Date')

# Take the Close price as the time series
data_close = data[['Close']]

# Resample to month start (since stock data is daily, we take monthly average Close)
data_monthly = data_close.resample('MS').mean()
print(data_monthly.head())

# Plot monthly data
data_monthly.plot(title="Monthly Average Close Price")
plt.show()

# Scale the data
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)
scaled_data.plot(title="Scaled Monthly Close Price")
plt.show()

# Seasonal decomposition (additive model to inspect trend/seasonality)
decomposition = seasonal_decompose(data_monthly, model="additive")
decomposition.plot()
plt.show()

# Adjust for multiplicative seasonality (no zero/negative values)
scaled_data = scaled_data + 1

# Split into train/test
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Build Holt-Winters model
model = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()

# Forecast on test data
test_predictions = model.forecast(steps=len(test_data))

# Plot train, test, predictions
ax = train_data.plot(label="Train Data")
test_data.plot(ax=ax, label="Test Data")
test_predictions.plot(ax=ax, label="Holt-Winters Prediction")
ax.legend()
ax.set_title("Visual Evaluation - Holt Winters")
plt.show()

# Evaluate RMSE
rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
print("Test RMSE:", rmse)

# Build final model on full data and predict next 12 months
final_model = ExponentialSmoothing(scaled_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
final_predictions = final_model.forecast(steps=12)  # forecast next year

# Plot final predictions
ax = scaled_data.plot(label="Monthly Close Price")
final_predictions.plot(ax=ax, label="Forecast - Next 12 Months")
ax.set_xlabel('Time')
ax.set_ylabel('Scaled Close Price')
ax.set_title('Future Prediction - Holt Winters')
ax.legend()
plt.show()

```
### OUTPUT:



Scaled_data plot
<img width="719" height="550" alt="image" src="https://github.com/user-attachments/assets/6b0370a1-84d3-4870-89f2-bf54de779d3f" />
Decomposed plot:
<img width="742" height="478" alt="image" src="https://github.com/user-attachments/assets/c1432e44-f0f0-47fb-95ca-e87655c210b5" />


TEST_PREDICTION
<img width="675" height="465" alt="image" src="https://github.com/user-attachments/assets/65faf1f8-7651-478a-b9d7-255fbb24c636" />



FINAL_PREDICTION

<img width="662" height="442" alt="image" src="https://github.com/user-attachments/assets/097d5a0a-1fd5-4965-950a-8e7201725c62" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
