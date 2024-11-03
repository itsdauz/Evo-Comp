import matplotlib.pyplot as plt
from itertools import permutations
import random
import numpy as np
import seaborn as sns
import streamlit as st

# Initial parameters
n_population = 250
crossover_per = 0.8
mutation_per = 0.2
n_generations = 200

# Set up city names and allow user to enter x, y coordinates for each city without instant updates
num_cities = st.number_input("Number of Cities", min_value=2, max_value=20, value=10)
city_coords = {}

for i in range(num_cities):
    city_name = st.text_input(f"Enter name for City {i + 1}", f"City_{i + 1}")
    x_input = st.text_input(f"Enter X coordinate for {city_name}", key=f"x_{i}")
    y_input = st.text_input(f"Enter Y coordinate for {city_name}", key=f"y_{i}")

    if x_input and y_input:  # Only add the city if both x and y inputs are provided
        try:
            x_coord = float(x_input)
            y_coord = float(y_input)
            city_coords[city_name] = (x_coord, y_coord)
        except ValueError:
            st.error(f"Please enter valid numbers for coordinates of {city_name}.")

# Button to confirm and display graph
if st.button("Submit Coordinates and Run GA") and len(city_coords) == num_cities:

    # Pastel Palette for Cities
    colors = sns.color_palette("pastel", len(city_coords))

    # Helper Functions
    def initial_population(cities_list, n_population=250):
        population_perms = []
        possible_perms = list(permutations(cities_list))

    # Check if possible permutations are less than the population size
    if len(possible_perms) < n_population:
        # Sample with replacement if needed
        for _ in range(n_population):
            population_perms.append(list(random.choice(possible_perms)))
    else:
        # Sample without replacement if there are enough unique routes
        random_ids = random.sample(range(len(possible_perms)), n_population)
        for i in random_ids:
            population_perms.append(list(possible_perms[i]))
            return population_perms

    def dist_two_cities(city_1, city_2):
        city_1_coords = city_coords[city_1]
        city_2_coords = city_coords[city_2]
        return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords)) ** 2))

    def total_dist_individual(individual):
        total_dist = 0
        for i in range(len(individual)):
            if i == len(individual) - 1:
                total_dist += dist_two_cities(individual[i], individual[0])
            else:
                total_dist += dist_two_cities(individual[i], individual[i + 1])
        return total_dist

    def fitness_prob(population):
        total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
        max_population_cost = max(total_dist_all_individuals)
        population_fitness = max_population_cost - np.array(total_dist_all_individuals)
        population_fitness_sum = population_fitness.sum()
        return population_fitness / population_fitness_sum

    def roulette_wheel(population, fitness_probs):
        population_fitness_probs_cumsum = fitness_probs.cumsum()
        selected_individual_index = np.searchsorted(population_fitness_probs_cumsum, np.random.rand())
        return population[selected_individual_index]

    def crossover(parent_1, parent_2):
        cut = round(random.uniform(1, len(city_coords) - 1))
        offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
        offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
        return offspring_1, offspring_2

    def mutation(offspring):
        index_1, index_2 = random.sample(range(len(city_coords)), 2)
        offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
        return offspring

    def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
        population = initial_population(cities_names, n_population)
        for _ in range(n_generations):
            fitness_probs = fitness_prob(population)
            parents = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child_1, child_2 = crossover(parents[i], parents[i + 1])
                    if random.random() < mutation_per:
                        child_1 = mutation(child_1)
                    if random.random() < mutation_per:
                        child_2 = mutation(child_2)
                    offspring.extend([child_1, child_2])
            population = parents + offspring
            fitness_probs = fitness_prob(population)
            sorted_indices = np.argsort(fitness_probs)[::-1]
            population = [population[i] for i in sorted_indices[:n_population]]
        return population

    # Run Genetic Algorithm
    cities_names = list(city_coords.keys())
    best_population = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)
    total_distances = [total_dist_individual(ind) for ind in best_population]
    best_index = np.argmin(total_distances)
    shortest_path = best_population[best_index]
    min_distance = total_distances[best_index]

    # Plotting the Best Route
    fig, ax = plt.subplots()
    x_coords, y_coords = zip(*[city_coords[city] for city in shortest_path])
    x_coords += (x_coords[0],)  # To return to the start
    y_coords += (y_coords[0],)
    ax.plot(x_coords, y_coords, '--go', label='Best Route', linewidth=2.5)

    # Draw cities and annotate
    for i, (city, (x, y)) in enumerate(city_coords.items()):
        color = colors[i]
        ax.scatter(x, y, c=[color], s=1200, zorder=2)
        ax.annotate(f"{i + 1}- {city}", (x, y), fontsize=14, ha='center', va='center', zorder=3)

    ax.set_title(f"TSP Best Route Using GA\nTotal Distance: {min_distance:.2f}\nGenerations: {n_generations} | Population: {n_population}")
    fig.set_size_inches(12, 8)
    st.pyplot(fig)
