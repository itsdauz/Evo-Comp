import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import streamlit as st
from itertools import permutations

# Title and Instructions
st.title("Traveling Salesman Problem with Genetic Algorithm")
st.write("Input the number of cities and their coordinates to find the shortest route using Genetic Algorithm.")

# Collect city names, x, and y coordinates from user inputs
city_coords = {}
n_cities = st.number_input("Number of Cities", min_value=2, max_value=20, value=10)

# Create placeholders for city inputs to gather user input for each city before displaying graph
with st.form("city_input_form"):
    for i in range(int(n_cities)):
        city_name = st.text_input(f"Enter name of city {i+1}", f"City{i+1}")
        x_coord = st.number_input(f"Enter X coordinate for {city_name}", value=float(random.randint(1, 20)), key=f"x_{i}")
        y_coord = st.number_input(f"Enter Y coordinate for {city_name}", value=float(random.randint(1, 20)), key=f"y_{i}")
        city_coords[city_name] = (x_coord, y_coord)
    
    # Button to confirm input and display the graph
    submitted = st.form_submit_button("Enter")

# Only run the algorithm and show the graph if the button was pressed
if submitted:
    # Extract coordinates and city names
    cities_names = list(city_coords.keys())
    x = [coord[0] for coord in city_coords.values()]
    y = [coord[1] for coord in city_coords.values()]

    # GA Parameters
    n_population = st.number_input("Population Size", min_value=50, max_value=500, value=250)
    crossover_per = st.slider("Crossover Percentage", 0.0, 1.0, 0.8)
    mutation_per = st.slider("Mutation Percentage", 0.0, 1.0, 0.2)
    n_generations = st.number_input("Generations", min_value=50, max_value=500, value=200)

    # Pastel color palette for city markers
    colors = sns.color_palette("pastel", len(cities_names))

    # City Icons for display
    city_icons = {city: "♕" if i % 2 == 0 else "♔" for i, city in enumerate(cities_names)}

    # Visualize Cities and Connections
    fig, ax = plt.subplots()
    ax.grid(False)  # Remove grid

    for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
        color = colors[i]
        icon = city_icons[city]
        ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
        ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
        ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30),
                    textcoords='offset points')

        # Connect cities with opaque lines
        for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
            if i != j:
                ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

    fig.set_size_inches(16, 12)
    st.pyplot(fig)

    # Genetic Algorithm Functions

    # Initial Population
    def initial_population(cities_list, n_population=250):
        population_perms = []
        possible_perms = list(permutations(cities_list))
        random_ids = random.sample(range(0, len(possible_perms)), n_population)
        for i in random_ids:
            population_perms.append(list(possible_perms[i]))
        return population_perms

    # Distance between two cities
    def dist_two_cities(city_1, city_2):
        city_1_coords = city_coords[city_1]
        city_2_coords = city_coords[city_2]
        return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords)) ** 2))

    # Total distance for an individual route
    def total_dist_individual(individual):
        total_dist = 0
        for i in range(0, len(individual)):
            if i == len(individual) - 1:
                total_dist += dist_two_cities(individual[i], individual[0])
            else:
                total_dist += dist_two_cities(individual[i], individual[i + 1])
        return total_dist

    # Fitness function
    def fitness_prob(population):
        total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
        max_population_cost = max(total_dist_all_individuals)
        population_fitness = max_population_cost - np.array(total_dist_all_individuals)
        return population_fitness / sum(population_fitness)

    # Roulette Wheel Selection
    def roulette_wheel(population, fitness_probs):
        cumsum_probs = fitness_probs.cumsum()
        selected_idx = np.searchsorted(cumsum_probs, random.random())
        return population[selected_idx]

    # Crossover
    def crossover(parent_1, parent_2):
        cut = random.randint(1, len(cities_names) - 2)
        offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
        offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
        return offspring_1, offspring_2

    # Mutation
    def mutation(offspring):
        index_1, index_2 = random.sample(range(len(cities_names)), 2)
        offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
        return offspring

    # Run Genetic Algorithm
    def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
        population = initial_population(cities_names, n_population)
        for _ in range(n_generations):
            fitness_probs = fitness_prob(population)
            parents = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]

            offspring = []
            for i in range(0, len(parents), 2):
                offspring_1, offspring_2 = crossover(parents[i], parents[i + 1])
                if random.random() < mutation_per:
                    offspring_1 = mutation(offspring_1)
                if random.random() < mutation_per:
                    offspring_2 = mutation(offspring_2)
                offspring.extend([offspring_1, offspring_2])

            mixed_population = parents + offspring
            fitness_probs = fitness_prob(mixed_population)
            best_indices = np.argsort(fitness_probs)[-n_population:]
            population = [mixed_population[i] for i in best_indices]
        return population

    # Execute the Genetic Algorithm and get the shortest path
    best_population = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)
    best_individual = min(best_population, key=total_dist_individual)

    # Prepare data for the shortest path
    x_shortest = [city_coords[city][0] for city in best_individual] + [city_coords[best_individual[0]][0]]
    y_shortest = [city_coords[city][1] for city in best_individual] + [city_coords[best_individual[0]][1]]
    min_distance = total_dist_individual(best_individual)

    # Plot the shortest path
    fig, ax = plt.subplots()
    ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
    plt.legend()

    for i, (x_pos, y_pos, city) in enumerate(zip(x_shortest, y_shortest, best_individual + [best_individual[0]])):
        ax.annotate(f"{i+1} - {city}", (x_pos, y_pos), fontsize=10)

    plt.title(f"Best Route (Distance: {min_distance:.2f})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    fig.set_size_inches(12, 8)
    st.pyplot(fig)
