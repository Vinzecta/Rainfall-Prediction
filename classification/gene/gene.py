import random
import crossover
import mutate
import selection
import value
from individual import Individual
from itertools import permutations


####################################################################################################
def genetic_algorithm(dataset, model):
    population = [Individual(Individual.create_chromosome(), dataset, model) for _ in range(value.POPULATION_SIZE)]
    count = value.GENERATIONS
    best_fitness = []
    best_value = value.INT_MIN
    stagnant_generation = 0
    max_stagnant_generation = value.MAX_STAGNANT_GENERATION


    while count > 0:
        # Sort population by fitness (higher is better)
        population = sorted(population, key=lambda x: x.fitness, reverse=True)
        for i in range(len(population)):
            print(f"ID_tour: {i} | Tour: {population[i].chromosome} | Fitness: {population[i].fitness}")
        print(
            f"Generation: {value.GENERATIONS - count + 1} | Best Fitness: {population[0].fitness} | Tour: {population[0].chromosome}")

        # Create a new generation
        new_generation = []

        # Elitism: Carry over the top 10% of the population
        elite_size = int(0.1 * value.POPULATION_SIZE)
        new_generation.extend(population[:elite_size])

        while (len(new_generation) < int(value.POPULATION_SIZE * 0.8)):
            parent1 = selection.tournament_selection(population)
            parent2 = selection.tournament_selection(population)
            child1, child2 = crossover.one_point_crossover(parent1, parent2, dataset, model)
            if random.random() < 0.1:
                mutate.scramble_mutate(child1)
                child1.calculate_fitness(dataset)
            if random.random() < 0.1:
                mutate.scramble_mutate(child2)
                child2.calculate_fitness(dataset)
            if (child1.check_valid_chromosome(child1.chromosome)):
                new_generation.append(child1)
            if (child2.check_valid_chromosome(child2.chromosome)):
                new_generation.append(child2)

        while (len(new_generation) < value.POPULATION_SIZE):
            new_generation.append(Individual(Individual.create_chromosome(), dataset, model))
        # Replace the old population with the new one
        population = new_generation
        best_fitness.append(population[0].fitness)
        count -= 1
        current_best_fitness = population[0].fitness
        if current_best_fitness > best_value:
            best_value = current_best_fitness
            stagnant_generation = 0  # Reset stagnant counter
        else:
            stagnant_generation += 1
        # Early stopping condition
        if stagnant_generation >= max_stagnant_generation:
            print(f"Stopping early: No improvement for {value.MAX_STAGNANT_GENERATION} consecutive generations.")
            break

    return population[0].chromosome, best_fitness, (value.GENERATIONS - count)