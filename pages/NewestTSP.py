import matplotlib.pyplot as plt
from itertools import permutations
from random import shuffle
import random
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Default values
default_x = [0,3,6,7,15,10,16,5,8,1.5]
default_y = [1,2,1,4.5,-1,2.5,11,6,9,12]
default_city_names = ["Gliwice", "Cairo", "Rome", "Krakow", "Paris", "Alexandria", "Berlin", "Tokyo", "Rio", "Budapest"]
default_n_population = 250
default_crossover_per = 0.8
default_mutation_per = 0.2
default_n_generations = 200

# Input section in Streamlit
st.title("Traveling Salesperson Problem Solver with Genetic Algorithm")
n_population = st.number_input("Population Size", value=default_n_population)
crossover_per = st.slider("Crossover Percentage", min_value=0.1, max_value=1.0, value=default_crossover_per)
mutation_per = st.slider("Mutation Percentage", min_value=0.0, max_value=1.0, value=default_mutation_per)
n_generations = st.number_input("Number of Generations", value=default_n_generations)

# Input new city details
new_city = st.text_input("Enter city name:")
new_x = st.number_input("Enter X coordinate:")
new_y = st.number_input("Enter Y coordinate:")

# Add new city button
if st.button("Add City"):
    if new_city and (new_x is not None) and (new_y is not None):
        default_city_names.append(new_city)
        default_x.append(new_x)
        default_y.append(new_y)
        st.success(f"City {new_city} added!")

# Wait for user to start calculation
if st.button("Run Genetic Algorithm"):
    # Prepare city data
    cities_names = default_city_names
    x = default_x
    y = default_y
    city_coords = dict(zip(cities_names, zip(x, y)))

    # Initialize colors and icons for cities
    colors = sns.color_palette("pastel", len(cities_names))
    city_icons = {name: chr(9812 + i % 12) for i, name in enumerate(cities_names)}

    # Initial population generation
    def initial_population(cities_list, n_population):
        population_perms = []
        possible_perms = list(permutations(cities_list))
        random_ids = random.sample(range(0, len(possible_perms)), n_population)
        for i in random_ids:
            population_perms.append(list(possible_perms[i]))
        return population_perms

    # Calculate distance between cities
    def dist_two_cities(city_1, city_2):
        city_1_coords = city_coords[city_1]
        city_2_coords = city_coords[city_2]
        return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords))**2))

    # Calculate total distance of a route
    def total_dist_individual(individual):
        total_dist = 0
        for i in range(len(individual)):
            if i == len(individual) - 1:
                total_dist += dist_two_cities(individual[i], individual[0])
            else:
                total_dist += dist_two_cities(individual[i], individual[i + 1])
        return total_dist

    # Fitness probability function
    def fitness_prob(population):
        total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
        max_population_cost = max(total_dist_all_individuals)
        population_fitness = max_population_cost - np.array(total_dist_all_individuals)
        population_fitness_sum = np.sum(population_fitness)
        return population_fitness / population_fitness_sum

    # Selection via roulette wheel
    def roulette_wheel(population, fitness_probs):
        population_fitness_probs_cumsum = np.cumsum(fitness_probs)
        selected_index = np.searchsorted(population_fitness_probs_cumsum, np.random.rand())
        return population[selected_index]

    # Crossover function
    def crossover(parent_1, parent_2):
        cut = random.randint(1, len(cities_names) - 1)
        offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
        offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
        return offspring_1, offspring_2

    # Mutation function
    def mutation(offspring):
        idx1, idx2 = random.sample(range(len(offspring)), 2)
        offspring[idx1], offspring[idx2] = offspring[idx2], offspring[idx1]
        return offspring

    # Run the genetic algorithm
    def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
        population = initial_population(cities_names, n_population)
        for generation in range(n_generations):
            fitness_probs = fitness_prob(population)
            parents = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]
            offspring = []
            for i in range(0, len(parents), 2):
                child1, child2 = crossover(parents[i], parents[i + 1])
                if random.random() < mutation_per:
                    child1 = mutation(child1)
                if random.random() < mutation_per:
                    child2 = mutation(child2)
                offspring += [child1, child2]
            population = parents + offspring
            population = sorted(population, key=total_dist_individual)[:n_population]
        return population[0], total_dist_individual(population[0])

    # Run GA and get results
    best_route, min_distance = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)

    # Plotting the results
    fig, ax = plt.subplots()
    x_route, y_route = zip(*[city_coords[city] for city in best_route + [best_route[0]]])
    ax.plot(x_route, y_route, 'o-', label='Best Route', linewidth=2.5)
    for i, city in enumerate(best_route):
        ax.text(city_coords[city][0], city_coords[city][1], f"{i+1} - {city}", ha='center', fontsize=12)

    plt.legend()
    plt.title(f"TSP Best Route Using GA\nDistance: {round(min_distance, 3)}")
    fig.set_size_inches(10, 8)
    st.pyplot(fig)
