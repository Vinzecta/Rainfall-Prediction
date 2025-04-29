import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

scaled_30_min_df = pd.read_csv("../processed/scaled_30_min.csv", index_col=0)

scaled_12_hr_df = pd.read_csv("../processed/scaled_12_hr.csv", index_col=0)
scaled_day_df = pd.read_csv("../processed/scaled_day.csv", index_col=0)

scaled_1_hr_df = pd.read_csv("../processed/scaled_1_hr.csv", parse_dates=["Date Time"], index_col="Date Time")

print(scaled_1_hr_df.head())

input_hours = 168 # 24 * 7 for 7 days
output_hours = 3

X, y = [], []
timestamps = scaled_1_hr_df.index

for i in range(len(scaled_1_hr_df) - input_hours - output_hours):
    # Extract input (past 7 days)
    X.append(scaled_1_hr_df.iloc[i:i+input_hours].values)

    # Extract output (next 3 hours)
    y.append(scaled_1_hr_df.iloc[i+input_hours:i+input_hours+output_hours].values)

# Convert to NumPy arrays
X, y = np.array(X), np.array(y)

print("Input shape:", X.shape)  # Expected: (samples, 168, features)
print("Output shape:", y.shape)  # Expected: (samples, 3, features)


############################## Random forest ##############################

# Flatten time dimension for RandomForestRegressor
X_flat = X.reshape(X.shape[0], -1)  # (samples, 168 * features)
y_flat = y.reshape(y.shape[0], -1)  # (samples, 1 * features)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y_flat, test_size=0.2, random_state=42)

# Fit the model using all samples at once
rf = RandomForestRegressor(n_estimators=100, random_state=94)
rf.fit(X_train, y_train)

# Predict next 3 hour
y_pred = rf.predict(X_test)

# Evaluate performance
mse_rf = mean_squared_error(y_test, y_pred)
print(f'MSE - RandomForest: {mse_rf}')

mse_rf_per_feature = mean_squared_error(y_test, y_pred, multioutput="raw_values")
print("MSE per feature:", mse_rf_per_feature)  # Array of MSE values for each feature