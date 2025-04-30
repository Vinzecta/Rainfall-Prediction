import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

scaled_30_min_df = pd.read_csv("../processed/scaled_30_min.csv", index_col=0)

scaled_12_hr_df = pd.read_csv("../processed/scaled_12_hr.csv", index_col=0)
scaled_day_df = pd.read_csv("../processed/scaled_day.csv", index_col=0)

scaled_1_hr_df = pd.read_csv("../processed/scaled_1_hr.csv", parse_dates=["Date Time"], index_col="Date Time")

# print(scaled_1_hr_df.head())

# input_hours = 168 # 24 * 7 for 7 days
# output_hours = 3

# X, y = [], []
# timestamps = scaled_1_hr_df.index

# for i in range(len(scaled_1_hr_df) - input_hours - output_hours):
#     # Extract input (past 7 days)
#     X.append(scaled_1_hr_df.iloc[i:i+input_hours].values)

#     # Extract output (next 3 hours)
#     y.append(scaled_1_hr_df.iloc[i+input_hours:i+input_hours+output_hours].values)

# # Convert to NumPy arrays
# X, y = np.array(X), np.array(y)

# print("Input shape:", X.shape)  # Expected: (samples, 168, features)
# print("Output shape:", y.shape)  # Expected: (samples, 3, features)


############################## Random forest ##############################
from sklearn.model_selection import KFold
X = scaled_1_hr_df[:].values
y = scaled_1_hr_df[:].values

# Fit the model using all samples at once
num_trees = [100, 250, 500]
for num_tree in num_trees:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1,1000))
    rf = RandomForestRegressor(n_estimators=num_tree, random_state=random.randint(1,1000))
    kfold = KFold(n_splits=10)
    # Using 10-fold cross validation to evaluate performance
    mses_rf = []
    for train_idx, val_idx in kfold.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        rf.fit(X_train_fold, y_train_fold)
        mse_rf = mean_squared_error(y_val_fold, rf.predict(X_val_fold))
        mses_rf.append(mse_rf)

    mean_mse = np.mean(mses_rf)
    std_mse = np.std(mses_rf)
    # Evaluate performance with number of tree
    print(f'MSE - RandomForest - {num_tree} trees: {mean_mse} +/- {std_mse}')


############################## SVR ##############################
from sklearn.model_selection import GridSearchCV
# Tuning C and gamma via grid search and find the best model
svr = MultiOutputRegressor(SVR())

C_range = [0.01, 0.1, 1, 10, 100, 1000, 10000]
gamma_range = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9]

param_grid = [
    {'estimator__C': C_range, 'estimator__kernel': ['linear']},
    {'estimator__C': C_range, 'estimator__gamma': gamma_range, 'estimator__kernel': ['rbf']}
]


gs = GridSearchCV(estimator=svr, param_grid=param_grid, scoring='neg_mean_squared_error', cv=10, refit=True, n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1,1000))
gs = gs.fit(X_train, y_train)
print(f'MSE - SVR: {-gs.best_score_}')
print(f'Best model: {gs.best_params_}')

############################## LR ##############################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1,1000))
lr = LinearRegression()
kfold = KFold(n_splits=10)
# Using 10-fold cross validation to evaluate performance
mses_lr = []
for train_idx, val_idx in kfold.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    lr.fit(X_train_fold, y_train_fold)
    mse_lr = mean_squared_error(y_val_fold, lr.predict(X_val_fold))
    mses_lr.append(mse_lr)

mean_mse = np.mean(mses_lr)
std_mse = np.std(mses_lr)
# Evaluate performance with number of linear regression
print(mses_lr)
print(f'MSE - LinearRegression: {mean_mse} +/- {std_mse}')



import joblib

# Save the trained model
joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(gs, "svr_model.pkl")
joblib.dump(lr, "linear_regression_model.pkl")


# For whatever fucking reason LinearRegression outperform others ?????
# MSE - RandomForest - 100 trees: 0.07447493090282847 +/- 0.0678771238265784
# MSE - RandomForest - 250 trees: 0.07241612282924667 +/- 0.07542657798448549
# MSE - RandomForest - 500 trees: 0.04760942694992552 +/- 0.004098491748925844
# MSE - SVR: 0.00282000991060531
# Best model: {'estimator__C': 1, 'estimator__kernel': 'linear'}
# MSE - LinearRegression: 2.5782636268150114e-30 +/- 2.5832615415962112e-30

# It outperform again wtfffff ???????
# MSE - RandomForest - 100 trees: 0.04065811909936812 +/- 0.0062688328513773395
# MSE - RandomForest - 250 trees: 0.04561172430357175 +/- 0.02235332240480774
# MSE - RandomForest - 500 trees: 0.041799740125409754 +/- 0.017631189839590448
# MSE - SVR: 0.0025983334525475354
# Best model: {'estimator__C': 1, 'estimator__kernel': 'linear'}
# MSE - LinearRegression: 2.837795549961422e-29 +/- 1.242510008958933e-29