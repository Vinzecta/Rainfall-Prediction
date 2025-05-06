import value
import random
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

## An individual will have a binary encoding chromosome for each features. 1 is selected, 0 is ignored
class Individual:
    ''' Class representing individual in the population '''
    def __init__(self, chromosome, dataset, model):
        self.chromosome = chromosome
        self.model = model
        self.fitness = self.calculate_fitness(dataset)

    @classmethod
    def create_chromosome(self):
        ''' Create a random chromosome of selected features '''
        while True:
            chromosome = [random.randint(0, 1) for _ in range(value.FEATURES)]
            if self.check_valid_chromosome(chromosome):
                break
        random.shuffle(chromosome)
        return chromosome
    
    def calculate_fitness(self, dataset):
        ''' Calculate fitness (MSE) of the selected model and features '''
        selected_feature = [index for index, value in enumerate(self.chromosome) if value == 1]
        # Select columns where chromosome is 1
        X = dataset.iloc[:, selected_feature]
        y = dataset[[
            "Rain_Type_Cloudburst", "Rain_Type_Heavy_Rain", "Rain_Type_Moderate_Rain",
            "Rain_Type_No_Rain", "Rain_Type_Shower", "Rain_Type_Very_Heavy_Rain", "Rain_Type_Weak_Rain"
        ]].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1,1000))
        self.model.fit(X_train, y_train)

        y_pred_continuous = self.model.predict(X_test)
        y_pred = (y_pred_continuous >= 0.5).astype(int) 
        return accuracy_score(y_test, y_pred)
    
    @staticmethod
    def check_valid_chromosome(chromosome):
        gnome_lst = []
        for gnome in chromosome:
            if gnome == 1:
                gnome_lst.append(gnome)

        return len(gnome_lst) >= 10