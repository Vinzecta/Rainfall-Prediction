import numpy as np
import pandas as pd
import random
import math
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

rain_type_df = rain_type_df.drop(columns=['Rain_Rate (mm/h)','rain (mm)','raining (s)'])
############################## Random forest ##############################
from sklearn.model_selection import KFold, StratifiedKFold
X = rain_type_df.drop(columns=[
    "Rain_Type_Cloudburst", "Rain_Type_Heavy_Rain", "Rain_Type_Moderate_Rain",
    "Rain_Type_No_Rain", "Rain_Type_Shower", "Rain_Type_Very_Heavy_Rain", "Rain_Type_Weak_Rain"
])

# Selecting Target (Rain_Type columns)
y = rain_type_df[[
    "Rain_Type_Cloudburst", "Rain_Type_Heavy_Rain", "Rain_Type_Moderate_Rain",
    "Rain_Type_No_Rain", "Rain_Type_Shower", "Rain_Type_Very_Heavy_Rain", "Rain_Type_Weak_Rain"
]].values

mask = [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1]
selected_feature = [index for index, value in enumerate(mask) if value == 1]
# Select columns where chromosome is 1
new_X = X.iloc[:, selected_feature].values

# X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.2, random_state=69)
X_train, y_train = new_X, y
num_trees = [100, 250, 500]
best_acc = 0.0
best_rf = None
for num_tree in num_trees:
    
    rf = RandomForestClassifier(n_estimators=num_tree, random_state=69)
    # mo_rf = MultiOutputClassifier(rf)
    kfold = KFold(n_splits=10)
    # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=69)
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
    print(f'Accuracy - RandomForest - {num_tree} trees: {mean_acc} +/- {std_acc}')

    if mean_acc > best_acc:
        best_acc = mean_acc

############################## LR ##############################

lr = LinearRegression()
kfold = KFold(n_splits=10)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=69)
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
knn = KNeighborsClassifier()
kfold = KFold(n_splits=10)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=69)
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

tree = DecisionTreeClassifier()
kfold = KFold(n_splits=10)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=69)
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



######################################################### Remove 2 important feature ####################################################
## Before filter
# Accuracy - RandomForest - 100 trees: 0.878642027973999 +/- 0.02214886062442598
# Accuracy - RandomForest - 250 trees: 0.8800365840370585 +/- 0.022151713519450392
# Accuracy - RandomForest - 500 trees: 0.8803283917958531 +/- 0.02282211734067736
# Accuracy - LinearRegression: 0.887949491226941 +/- 0.022467729906366904
# Accuracy - KNN: 0.8741662785712874 +/- 0.02331564833175469
# Accuracy - Tree: 0.8276946440313454 +/- 0.02268214275508905



## No linear: 
## Take all voting: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
# Accuracy - RandomForest - 100 trees: 0.8570127257421458 +/- 0.01030668474428443
# Accuracy - RandomForest - 250 trees: 0.858245112627593 +/- 0.009469053185794944
# Accuracy - RandomForest - 500 trees: 0.8583100055406139 +/- 0.009794152841160116
# Accuracy - LinearRegression: 0.8907710498095703 +/- 0.02166997433434772
# Accuracy - KNN: 0.8785452039614757 +/- 0.021499969236063503
# Accuracy - Tree: 0.8023335575662192 +/- 0.0346430819330061

## Take or voting: [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1]
# Accuracy - RandomForest - 100 trees: 0.8794527371346907 +/- 0.02166677695951645
# Accuracy - RandomForest - 250 trees: 0.8801660122684416 +/- 0.022740897120325475
# Accuracy - RandomForest - 500 trees: 0.8806848505653992 +/- 0.023078546840871406
# Accuracy - LinearRegression: 0.8901225308614708 +/- 0.02074531669793074
# Accuracy - KNN: 0.8725772541189645 +/- 0.022902430012371904
# Accuracy - Tree: 0.8312614719521682 +/- 0.02157650881933212

## Major voting: [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1]
# Accuracy - RandomForest - 100 trees: 0.8749774925714968 +/- 0.021906890029406365
# Accuracy - RandomForest - 250 trees: 0.875658615738456 +/- 0.021623298208720054
# Accuracy - RandomForest - 500 trees: 0.8762748354749046 +/- 0.021814333016036684
# Accuracy - LinearRegression: 0.891225005710997 +/- 0.02201475890079133
# Accuracy - KNN: 0.8752687218683437 +/- 0.023951320924377642
# Accuracy - Tree: 0.8250673014182203 +/- 0.018854215936058513