import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
######################################################## Multiple output ########################################################

rain_type_df = pd.read_csv("./processed/rain_type.csv", parse_dates=["Date Time"], index_col="Date Time")

rain_type_df = rain_type_df.drop(columns=['Rain_Rate (mm/h)'])
############################## Random forest ##############################
from sklearn.model_selection import KFold, StratifiedKFold
X = rain_type_df.drop(columns=[
    "Rain_Type_Cloudburst", "Rain_Type_Heavy_Rain", "Rain_Type_Moderate_Rain",
    "Rain_Type_No_Rain", "Rain_Type_Shower", "Rain_Type_Very_Heavy_Rain", "Rain_Type_Weak_Rain"
]).values

# Selecting Target (Rain_Type columns)
y = rain_type_df[[
    "Rain_Type_Cloudburst", "Rain_Type_Heavy_Rain", "Rain_Type_Moderate_Rain",
    "Rain_Type_No_Rain", "Rain_Type_Shower", "Rain_Type_Very_Heavy_Rain", "Rain_Type_Weak_Rain"
]].values

# num_trees = [100, 250, 500]
# for num_tree in num_trees:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1,1000))
#     rf = RandomForestClassifier(n_estimators=num_tree, random_state=random.randint(1,1000))
#     mo_rf = MultiOutputClassifier(rf)
#     kfold = KFold(n_splits=10)
#     # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random.randint(1,1000))
#     # Using 10-fold cross validation to evaluate performance
#     accs_rf = []
#     for train_idx, val_idx in kfold.split(X_train, y_train):
#         X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
#         y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
#         mo_rf.fit(X_train_fold, y_train_fold)
#         acc_rf = accuracy_score(y_val_fold, mo_rf.predict(X_val_fold))
#         accs_rf.append(acc_rf)

#     mean_acc = np.mean(accs_rf)
#     std_acc = np.std(accs_rf)
#     # Evaluate performance with number of tree
#     print(f'Accuracy - RandomForest - {num_tree} trees: {mean_acc} +/- {std_acc}')


############################## SVC ##############################
# from sklearn.model_selection import GridSearchCV
# # Tuning C and gamma via grid search and find the best model
# svc = MultiOutputClassifier(SVC())

# C_range = [0.01, 0.1, 1, 10, 100, 1000, 10000]
# gamma_range = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9]

# param_grid = [
#     {'estimator__C': C_range, 'estimator__kernel': ['linear']},
#     {'estimator__C': C_range, 'estimator__gamma': gamma_range, 'estimator__kernel': ['rbf']}
# ]

# gs = GridSearchCV(estimator=svc, param_grid=param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1,1000))
# gs = gs.fit(X, y)
# print(f'Accuracy - SVC: {gs.best_score_}')
# print(f'Best model: {gs.best_params_}')

############################## LR ##############################
mask = [1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]
selected_feature = [index for index, value in enumerate(mask) if value == 1]
# Select columns where chromosome is 1
new_X = rain_type_df.iloc[:, selected_feature].values

X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.2, random_state=random.randint(1,1000))
lr = LinearRegression()
kfold = KFold(n_splits=10)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random.randint(1,1000))
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

############################## KNN ##############################
mask = [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
selected_feature = [index for index, value in enumerate(mask) if value == 1]
# Select columns where chromosome is 1
new_X = rain_type_df.iloc[:, selected_feature].values

X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.2, random_state=random.randint(1,1000))
knn = KNeighborsClassifier()
kfold = KFold(n_splits=10)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random.randint(1,1000))
# Using 10-fold cross validation to evaluate performance
accs_knn = []
for train_idx, val_idx in kfold.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    knn.fit(X_train_fold, y_train_fold)

    y_pred_continuous = knn.predict(X_val_fold)

    y_pred = (y_pred_continuous >= 0.5).astype(int) 

    acc_knn = accuracy_score(y_val_fold, y_pred)
    accs_knn.append(acc_knn)

mean_acc = np.mean(accs_knn)
std_acc = np.std(accs_knn)
# Evaluate performance with accuracy of KNN
print(f'Accuracy - KNN: {mean_acc} +/- {std_acc}')

############################## Tree ##############################
mask = [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
selected_feature = [index for index, value in enumerate(mask) if value == 1]
# Select columns where chromosome is 1
new_X = rain_type_df.iloc[:, selected_feature].values

X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.2, random_state=random.randint(1,1000))
tree = DecisionTreeClassifier()
kfold = KFold(n_splits=10)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random.randint(1,1000))
# Using 10-fold cross validation to evaluate performance
accs_tree = []
for train_idx, val_idx in kfold.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    tree.fit(X_train_fold, y_train_fold)

    y_pred_continuous = tree.predict(X_val_fold)

    y_pred = (y_pred_continuous >= 0.5).astype(int) 

    acc_tree = accuracy_score(y_val_fold, y_pred)
    accs_tree.append(acc_tree)

mean_acc = np.mean(accs_tree)
std_acc = np.std(accs_tree)
# Evaluate performance with accuracy of Desicision tree classifier
print(f'Accuracy - Tree: {mean_acc} +/- {std_acc}')

# Accuracy - RandomForest - 100 trees: 0.8847819788497754 +/- 0.04248507157883245
# Accuracy - RandomForest - 250 trees: 0.9052947993625959 +/- 0.022410580084647497
# Accuracy - RandomForest - 500 trees: 0.8933362306243662 +/- 0.016378159672278226
# Accuracy - LinearRegression: 0.862588729537882 +/- 0.026209485641225605
# Accuracy - KNN: 0.8847457627118646 +/- 0.033250182760801716
# Accuracy - Tree: 0.9274880486744894 +/- 0.011524175397089536
# After filter: 
# Accuracy - LinearRegression: 0.8668767202665508 +/- 0.03186056064922991 => No improvements
# Accuracy - KNN: Accuracy - KNN: 0.9018542662610459 +/- 0.01892346402972425 => Slight improvements
# Accuracy - Tree: 0.9420107199768216 +/- 0.024310848629831617 => Slight improvements