from value import POPULATION_SIZE
import random
import numpy as np
################################### Selection ################################################
def random_selection(population):
    return random.choice(population)

def roulette_wheel(population):
    ''' Choosing parent based on their fitness
        The lower the fitness, the higher chance it get picked
        The chance is proportional to the fitness
        n is population size
        Time complexity: O(3n)
        Space complexity: O(2n)
    '''
    fitness_scores = [1/individual.fitness for individual in population]
    total_fitness = sum(fitness_scores)
    probability_individual = [fitness/total_fitness for fitness in fitness_scores]
    return np.random.choice(population, p=probability_individual)
        
def rank_selection(population):
    ''' Choosing parent based on their fitness
        1. Assign each individual rank based on their fitness
        2. The higher the fitness, the higher the rank, the higher the chance to get selected
        3. The chance of getting selected is not propotional to the fitness
        n is population size
        Time complexity: O(n log n) + O(2n)
        Space complexity: O(n)
    '''
    population = sorted(population, key=lambda x: x.fitness, reverse=True)
    rank_sum = POPULATION_SIZE * (POPULATION_SIZE + 1) / 2
    probability = [i/rank_sum for i in range(1, POPULATION_SIZE + 1)]
    roll = random.random()
    cumulative_probability  = 0
    for i in range(0, POPULATION_SIZE):
        cumulative_probability += probability[i]
        if cumulative_probability >= roll:
            return population[i]
    
def tournament_selection(population, k=10):
    ''' Select k individuals from parents
        Select the best individual from k
        n is population size
        Time complexity: O(1)
        Space complexity: O(1)
    '''
    tournament = []
    for _ in range(k):
        tournament.append(population[random.randint(0, POPULATION_SIZE-1)])

    return min(tournament, key=lambda x: x.fitness)
################################### Selection ################################################