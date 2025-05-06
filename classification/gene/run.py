import value
import gene
import time
import random
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, SVC
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier


if __name__=="__main__":
    total_generations = 0
    # Load the models back
    # rf_loaded = joblib.load("../classic-ml/random_forest_model.pkl")
    # svr_loaded = joblib.load("../classic-ml/svr_model.pkl")
    # lr_loaded = joblib.load("../classic-ml/linear_regression_model.pkl")
    # scaled_1_hr_df = pd.read_csv("../processed/scaled_1_hr.csv", parse_dates=["Date Time"], index_col="Date Time")

    rain_type_df = pd.read_csv("../processed/rain_type.csv", parse_dates=["Date Time"], index_col="Date Time")
    rain_type_df = rain_type_df.drop(columns=['Rain_Rate (mm/h)'])

    is_rain_df = pd.read_csv("../processed/is_rain.csv", parse_dates=["Date Time"], index_col="Date Time")
    is_rain_df['Is_Rain'] = is_rain_df['Is_Rain'].map({'Yes': 1, 'No': 0})
    is_rain_df = is_rain_df.drop(columns=['Rain_Rate (mm/h)'])

    genetic_time_performance = 0

    best_fitness = None
    best_chromosome = None
    for i in range(value.TESTS):
        gene_start = time.perf_counter()
        # model = RandomForestRegressor(n_estimators=500, random_state=random.randint(1,1000)) # model to test
        random.seed(i)
        chromosome, fitness, num_gen = gene.genetic_algorithm(rain_type_df, RandomForestClassifier())
        if (not (best_fitness and best_chromosome) or best_fitness < fitness):
            best_chromosome = chromosome
            best_fitness = fitness
        gene_end = time.perf_counter()
        print(f"Genetic algorithm runtime: {gene_end - gene_start:0.5f} seconds")
        genetic_time_performance += (gene_end - gene_start)
    
    
    print(f"Total genetic algorithm algorithm runtime: {genetic_time_performance:0.5f} seconds")
    print(f"Best chromosome: {best_chromosome}. Fitness: {best_fitness}")

# Best, set limit to 10:
# LinearRegression:         [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
# KNeighborsClassifier:     [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
# DecisionTreeClassifier:   [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
# RandomForestRegressor:    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1]
#                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0]