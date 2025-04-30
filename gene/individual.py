import value
import random
import pandas as pd
from sklearn.metrics import mean_squared_error
## An individual will have a binary encoding chromosome for each features. 1 is selected, 0 is ignored
class Individual:
    ''' Class representing individual in the population '''
    def __init__(self, chromosome, dataset, model):
        self.chromosome = chromosome
        self.fitness = self.calculate_fitness(dataset)
        self.model = model

    @classmethod
    def create_gnome(self):
        ''' Create a random chromosome of selected features '''
        gnome = list(range(value.FEATURES))
        random.shuffle(gnome)
        return gnome
    
    def calculate_fitness(self, dataset):
        ''' Calculate fitness (MSE) of the selected model and features '''
        # Convert chromosome to a boolean mask
        column_mask = pd.Series(self.chromosome) == 1

        # Select columns where chromosome is 1
        X = dataset.loc[:, column_mask]
        y = dataset.loc[:, column_mask]

        return 

        
    
    def check_valid_chromosome(self):
        return 1 in self.chromosome