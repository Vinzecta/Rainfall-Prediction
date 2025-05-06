import random
from individual import Individual
import numpy as np

def one_point_crossover(partner1, partner2, dataset, model):
    point =  random.randint(0, len(partner1.chromosome) - 1)
    child1 = partner1.chromosome[:point] + partner2.chromosome[point:]
    child2 = partner2.chromosome[:point] + partner1.chromosome[point:]
    return Individual(child1, dataset, model), Individual(child2, dataset, model)

def uniform_crossover(partner1, partner2, dataset, model):
    child1 = list(partner1.chromosome)
    child2 = list(partner2.chromosome)
    for i in range(len(partner1.chromosome)):
        if bool(random.getrandbits(1)):
            child1[i], child2[i] = child2[i], child1[i]
    return Individual(child1, dataset, model), Individual(child2, dataset, model)