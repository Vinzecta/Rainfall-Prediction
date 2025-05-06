import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



is_rain_df = pd.read_csv("./processed/is_rain.csv", parse_dates=["Date Time"], index_col="Date Time")

is_rain_df['Is_Rain'] = is_rain_df['Is_Rain'].map({'Yes': 1, 'No': 0})
is_rain_df = is_rain_df.drop(columns=['Rain_Rate (mm/h)'])
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
from sklearn.model_selection import KFold, StratifiedKFold
X = is_rain_df.iloc[:, :-1].values
y = is_rain_df.iloc[:, -1].values


# Fit the model using all samples at once
num_trees = [100, 250, 500]
for num_tree in num_trees:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1,1000))
    rf = RandomForestClassifier(n_estimators=num_tree, random_state=random.randint(1,1000))
    # kfold = KFold(n_splits=10)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random.randint(1,1000))
    # Using 10-fold cross validation to evaluate performance
    accs_rf = []
    for train_idx, val_idx in kfold.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        rf.fit(X_train_fold, y_train_fold)
        acc_rf = accuracy_score(y_val_fold, rf.predict(X_val_fold))
        accs_rf.append(acc_rf)

    mean_acc = np.mean(accs_rf)
    std_acc = np.std(accs_rf)
    # Evaluate performance with number of tree
    # print(f'Accuracy array: {accs_rf}')
    print(f'Accuracy - RandomForest - {num_tree} trees: {mean_acc} +/- {std_acc}')


############################## SVC ##############################
from sklearn.model_selection import GridSearchCV
# Tuning C and gamma via grid search and find the best model
svc = SVC()

C_range = [0.01, 0.1, 1, 10, 100, 1000, 10000]
gamma_range = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9]

param_grid = [
    {'C': C_range, 'kernel': ['linear']},
    {'C': C_range, 'gamma': gamma_range, 'kernel': ['rbf']}
]

gs = GridSearchCV(estimator=svc, param_grid=param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1,1000))
gs = gs.fit(X_train, y_train)
print(f'Accuracy - SVC: {gs.best_score_}')
print(f'Best model: {gs.best_params_}')

############################## LR ##############################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1,1000))
lr = LinearRegression()
# kfold = KFold(n_splits=10)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random.randint(1,1000))
# Using 10-fold cross validation to evaluate performance
accs_lr = []
for train_idx, val_idx in kfold.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    lr.fit(X_train_fold, y_train_fold)

    y_pred_continuous = lr.predict(X_val_fold)

    y_pred = (y_pred_continuous >= 0.5).astype(int) 

    acc_lr = accuracy_score(y_val_fold, y_pred)
    accs_lr.append(acc_lr)

mean_acc = np.mean(accs_lr)
std_acc = np.std(accs_lr)
# Evaluate performance with number of linear regression
print(f'Accuracy - LinearRegression: {mean_acc} +/- {std_acc}')

