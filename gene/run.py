import value
import gene
import time
import random
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

if __name__=="__main__":
    total_generations = 0
    # Load the models back
    # rf_loaded = joblib.load("../classic-ml/random_forest_model.pkl")
    # svr_loaded = joblib.load("../classic-ml/svr_model.pkl")
    # lr_loaded = joblib.load("../classic-ml/linear_regression_model.pkl")
    scaled_1_hr_df = pd.read_csv("../processed/scaled_1_hr.csv", parse_dates=["Date Time"], index_col="Date Time")
    genetic_time_performance = 0

    best_fitness = None
    best_chromosome = None
    for i in range(value.TESTS):
        gene_start = time.perf_counter()
        # model = RandomForestRegressor(n_estimators=500, random_state=random.randint(1,1000)) # model to test
        chromosome, fitness, num_gen = gene.genetic_algorithm(scaled_1_hr_df, MLPRegressor())
        if (not (best_fitness and best_chromosome) or best_fitness > fitness):
            best_chromosome = chromosome
            best_fitness = fitness
        gene_end = time.perf_counter()
        print(f"Genetic algorithm runtime: {gene_end - gene_start:0.5f} seconds")
        genetic_time_performance += (gene_end - gene_start)
    
    
    print(f"Total genetic algorithm algorithm runtime: {genetic_time_performance:0.5f} seconds")
    print(f"Best chromosome: {best_chromosome}")

# Best, set limit to 10:
# linear regression:        [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
# svr linear kernel c = 1:  [1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]