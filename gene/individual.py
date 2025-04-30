import value
import random
import pandas as pd
from sklearn.metrics import mean_squared_error
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
        y = dataset.iloc[:, selected_feature]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1,1000))
        self.model.fit(X_train, y_train)
        return mean_squared_error(y_test, self.model.predict(X_test))
    @staticmethod
    def check_valid_chromosome(chromosome):
        gnome_lst = []
        for gnome in chromosome:
            if gnome == 1:
                gnome_lst.append(gnome)

        return len(gnome_lst) >= 10