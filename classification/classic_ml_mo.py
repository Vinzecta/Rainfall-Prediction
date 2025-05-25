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

mask = [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1]
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
# Accuracy - RandomForest - 100 trees: 0.878577166613448 +/- 0.022209822770644393
# Accuracy - RandomForest - 250 trees: 0.879614895794813 +/- 0.02285025107809665
# Accuracy - RandomForest - 500 trees: 0.879971606984118 +/- 0.023191043359977487
# Accuracy - LinearRegression: 0.8881440122036539 +/- 0.02307837997015312
# Accuracy - KNN: 0.8742311609668182 +/- 0.02443924685052174
# Accuracy - Tree: 0.8278566133766485 +/- 0.023069573395230556

## No linear: 
## Take all voting: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
# Accuracy - RandomForest - 100 trees: 0.8352168790568589 +/- 0.04357255395895333
# Accuracy - RandomForest - 250 trees: 0.8366114982248583 +/- 0.04309855379212968
# Accuracy - RandomForest - 500 trees: 0.8369358365800824 +/- 0.04288756007883909
# Accuracy - LinearRegression: 0.891225005710997 +/- 0.02201475890079133
# Accuracy - KNN: 0.8820796169782579 +/- 0.02261647808181005
# Accuracy - Tree: 0.8058660353648495 +/- 0.05794500828440858

## Take or voting: [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
# Accuracy - RandomForest - 100 trees: 0.8784149764008561 +/- 0.02236661076581743
# Accuracy - RandomForest - 250 trees: 0.879355534492529 +/- 0.02219650176305159
# Accuracy - RandomForest - 500 trees: 0.8805229548425257 +/- 0.022070967546886058
# Accuracy - LinearRegression: 0.888500775980409 +/- 0.023272178853097604
# Accuracy - KNN: 0.8726744252086565 +/- 0.025265145142976782
# Accuracy - Tree: 0.8262675363368761 +/- 0.02365667185127065

## Major voting: [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
# Accuracy - RandomForest - 100 trees: 0.8785446149820382 +/- 0.023324063374248447
# Accuracy - RandomForest - 250 trees: 0.8795176090127317 +/- 0.02290534066532216
# Accuracy - RandomForest - 500 trees: 0.8797122877517939 +/- 0.021972072204012365
# Accuracy - LinearRegression: 0.891225005710997 +/- 0.02201475890079133
# Accuracy - KNN: 0.8724474262229631 +/- 0.02447955261393022
# Accuracy - Tree: 0.8290242545939345 +/- 0.021666167242155115

# KNeighborsClassifier:     [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1]
# Accuracy - RandomForest - 100 trees: 0.8774746076240023 +/- 0.02134580238223546
# Accuracy - RandomForest - 250 trees: 0.8787716875901612 +/- 0.02187621210166064
# Accuracy - RandomForest - 500 trees: 0.8787069208870198 +/- 0.021550216426836487
# Accuracy - LinearRegression: 0.891225005710997 +/- 0.02201475890079133
# Accuracy - KNN: 0.8702748178055215 +/- 0.02456912161314162
# Accuracy - Tree: 0.8261051883619347 +/- 0.01960637730279816


# DecisionTreeClassifier:   [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1]
# Accuracy - RandomForest - 100 trees: 0.8777339899612662 +/- 0.0226977969959668
# Accuracy - RandomForest - 250 trees: 0.878512378875327 +/- 0.022263682542997026
# Accuracy - RandomForest - 500 trees: 0.8788690374771824 +/- 0.022648097550382152
# Accuracy - LinearRegression: 0.8900249601071606 +/- 0.023117923020787846
# Accuracy - KNN: 0.8716043652631706 +/- 0.02465071592170199
# Accuracy - Tree: 0.8266891509566919 +/- 0.02447133985712534

# RandomForestClassifier:   [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1]
# Accuracy - RandomForest - 100 trees: 0.8787070470968992 +/- 0.021835412933535168
# Accuracy - RandomForest - 250 trees: 0.8802636881976514 +/- 0.022098354402391156
# Accuracy - RandomForest - 500 trees: 0.8806203573169966 +/- 0.02241652456703205
# Accuracy - LinearRegression: 0.891225005710997 +/- 0.02201475890079133
# Accuracy - KNN: 0.8701122910332508 +/- 0.026344511248676112
# Accuracy - Tree: 0.8158577980667172 +/- 0.02103445970834434