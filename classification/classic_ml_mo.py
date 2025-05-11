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

mask = [1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
selected_feature = [index for index, value in enumerate(mask) if value == 1]
# Select columns where chromosome is 1
new_X = X.iloc[:, selected_feature].values

num_trees = [100, 250, 500]
for num_tree in num_trees:
    X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.2, random_state=69)
    rf = RandomForestClassifier(n_estimators=num_tree, random_state=69)
    mo_rf = MultiOutputClassifier(rf)
    kfold = KFold(n_splits=10)
    # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=69)
    # Using 10-fold cross validation to evaluate performance
    accs_rf = []
    for train_idx, val_idx in kfold.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        mo_rf.fit(X_train_fold, y_train_fold)
        acc_rf = accuracy_score(y_val_fold, mo_rf.predict(X_val_fold))
        accs_rf.append(acc_rf)

    mean_acc = np.mean(accs_rf)
    std_acc = np.std(accs_rf)
    # Evaluate performance with number of tree
    print(f'Accuracy - RandomForest - {num_tree} trees: {mean_acc} +/- {std_acc}')


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
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
# gs = gs.fit(X, y)
# print(f'Accuracy - SVC: {gs.best_score_}')
# print(f'Best model: {gs.best_params_}')

############################## LR ##############################
# mask = [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
# selected_feature = [index for index, value in enumerate(mask) if value == 1]
# # Select columns where chromosome is 1
# new_X = rain_type_df.iloc[:, selected_feature].values

X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.2, random_state=69)
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
# mask = [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
# selected_feature = [index for index, value in enumerate(mask) if value == 1]
# # Select columns where chromosome is 1
# new_X = rain_type_df.iloc[:, selected_feature].values

X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.2, random_state=69)
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
# mask = [0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
# selected_feature = [index for index, value in enumerate(mask) if value == 1]
# # Select columns where chromosome is 1
# new_X = rain_type_df.iloc[:, selected_feature].values

X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.2, random_state=69)
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

# Accuracy - RandomForest - 100 trees: 0.8847819788497754 +/- 0.04248507157883245
# Accuracy - RandomForest - 250 trees: 0.9052947993625959 +/- 0.022410580084647497
# Accuracy - RandomForest - 500 trees: 0.8933362306243662 +/- 0.016378159672278226
# Accuracy - LinearRegression: 0.862588729537882 +/- 0.026209485641225605
# Accuracy - KNN: 0.8847457627118646 +/- 0.033250182760801716
# Accuracy - Tree: 0.9274880486744894 +/- 0.011524175397089536


# After filter: 
# Accuracy - LinearRegression: 0.8668767202665508 +/- 0.03186056064922991 => No improvements
# Accuracy - KNN: 0.9018542662610459 +/- 0.01892346402972425 => Slight improvements
# Accuracy - Tree: 0.9420107199768216 +/- 0.024310848629831617 => Slight improvements

# Accuracy - RandomForest - 100 trees: 0.8993553527451834 +/- 0.02020487266001686
# Accuracy - RandomForest - 250 trees: 0.911241489207591 +/- 0.02073181292519888
# Accuracy - RandomForest - 500 trees: 0.912994350282486 +/- 0.027410536100320623 => slight improvement

# Take all voting [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0]
# Accuracy - RandomForest - 100 trees: 0.9342894393741851 +/- 0.01483754617835127
# Accuracy - RandomForest - 250 trees: 0.9334564682022309 +/- 0.021143378333895315
# Accuracy - RandomForest - 500 trees: 0.9325727944372012 +/- 0.018959124329052487
# Accuracy - LinearRegression: 0.8609952194697957 +/- 0.03400456523223149
# Accuracy - KNN: 0.9138418079096045 +/- 0.03132143218096468
# Accuracy - Tree: 0.9607344632768362 +/- 0.021699299101765074


######################################################### Remove 2 important feature ####################################################
## Before filter
# Accuracy - RandomForest - 100 trees: 0.8379110531652906 +/- 0.034646831159389054
# Accuracy - RandomForest - 250 trees: 0.837925539620455 +/- 0.039099948852384575
# Accuracy - RandomForest - 500 trees: 0.8404823989569753 +/- 0.03838538941005233
# Accuracy - LinearRegression: 0.829400260756193 +/- 0.042785593277700706
# Accuracy - KNN: 0.8430320150659135 +/- 0.03364016362027386
# Accuracy - Tree: 0.7986672461248732 +/- 0.03790754348331666


## Take all voting: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
# Accuracy - RandomForest - 100 trees: 0.7559684195277415 +/- 0.035670472036378166
# Accuracy - RandomForest - 250 trees: 0.7559684195277415 +/- 0.035670472036378166
# Accuracy - RandomForest - 500 trees: 0.7559684195277415 +/- 0.035670472036378166
# Accuracy - LinearRegression: 0.8464363320295524 +/- 0.04099278333455359
# Accuracy - KNN: 0.8122845139794294 +/- 0.06043015004802955
# Accuracy - Tree: 0.7551137186730407 +/- 0.036057139751591735

## Take or voting: [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
# Accuracy - RandomForest - 100 trees: 0.8379327828480372 +/- 0.03869588499167963
# Accuracy - RandomForest - 250 trees: 0.8413370998116761 +/- 0.038615890747275075
# Accuracy - RandomForest - 500 trees: 0.8413443430392583 +/- 0.035200419769688136
# Accuracy - LinearRegression: 0.8268361581920904 +/- 0.04247190562249235
# Accuracy - KNN: 0.8447559032304796 +/- 0.034931767758055884
# Accuracy - Tree: 0.8003766478342749 +/- 0.029772837410349452

## Major voting: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
# Accuracy - RandomForest - 100 trees: 0.8293857743010286 +/- 0.03810137717975042
# Accuracy - RandomForest - 250 trees: 0.8242575691728234 +/- 0.0385642394888533
# Accuracy - RandomForest - 500 trees: 0.8276763725916269 +/- 0.03826901491435314
# Accuracy - LinearRegression: 0.8455816311748515 +/- 0.0416088524520118
# Accuracy - KNN: 0.8259379979718963 +/- 0.03220366108384031
# Accuracy - Tree: 0.7721787628567289 +/- 0.03840897277171885

## More than 1: [1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
# Accuracy - RandomForest - 100 trees: 0.8413443430392584 +/- 0.03721790244180247
# Accuracy - RandomForest - 250 trees: 0.8421845574387948 +/- 0.03978134365282687
# Accuracy - RandomForest - 500 trees: 0.8438939591481965 +/- 0.03861258304368859
# Accuracy - LinearRegression: 0.8387657540199914 +/- 0.04275560119037604
# Accuracy - KNN: 0.8379182963928727 +/- 0.030835059569718284
# Accuracy - Tree: 0.8054831232797335 +/- 0.024451558247106923