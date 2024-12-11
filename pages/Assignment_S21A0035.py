import csv
import streamlit as st
import random

################################# CSV READING FUNCTION ####################################
# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}

    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        # Skip the header
        header = next(reader)

        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
            program_ratings[program] = ratings

    return program_ratings

# Path to the CSV file
file_path = 'pages/program_ratings.csv'

# Get the data in the required format
program_ratings_dict = read_csv_to_dict(file_path)

######################################## INPUT PARAMETERS ########################################
st.title("Genetic Algorithm for Optimal Program Scheduling")
st.header("Input Parameters")

# Allow users to input parameters interactively
CO_R = st.slider(
    "Crossover Rate (CO_R)", min_value=0.0, max_value=0.95, value=0.8, step=0.01
)
MUT_R = st.slider(
    "Mutation Rate (MUT_R)", min_value=0.01, max_value=0.05, value=0.2, step=0.01
)

GEN = 100
POP = 50
EL_S = 2

# Display selected parameters
st.write("### Selected Parameters")
st.write(f"- **Crossover Rate:** {CO_R}")
st.write(f"- **Mutation Rate:** {MUT_R}")

######################################## DEFINING FUNCTIONS ########################################
ratings = program_ratings_dict

all_programs = list(ratings.keys())  # all programs
all_time_slots = list(range(6, 24))  # time slots

def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

def prioritize_high_rated_programs(time_slots):
    prioritized_schedule = []
    remaining_programs = all_programs.copy()

    for time_slot in time_slots:
        prioritized_programs = [program for program in remaining_programs if ratings[program][time_slot - 6] == 0.9]

        if prioritized_programs:
            selected_program = prioritized_programs[0]
            prioritized_schedule.append(selected_program)
            remaining_programs.remove(selected_program)
        else:
            random_program = random.choice(remaining_programs)
            prioritized_schedule.append(random_program)
            remaining_programs.remove(random_program)

    return prioritized_schedule

# Crossover
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

# Mutation
def mutate(schedule):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

# Genetic algorithm
def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S):
    population = [initial_schedule]

    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []

        # Elitism
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

    return population[0]

##################################### GENERATING RESULTS #####################################
# Prioritize programs with high ratings (0.9)
initial_prioritized_schedule = prioritize_high_rated_programs(all_time_slots)

# Genetic algorithm
rem_t_slots = len(all_time_slots) - len(initial_prioritized_schedule)
genetic_schedule = genetic_algorithm(initial_prioritized_schedule, generations=GEN, population_size=POP, elitism_size=EL_S)

final_schedule = initial_prioritized_schedule + genetic_schedule[:rem_t_slots]

st.write("### Final Optimal Schedule")
for time_slot, program in enumerate(final_schedule):
    st.write(f"Time Slot {all_time_slots[time_slot]:02d}:00 - Program {program}")

st.write("### Total Ratings:", fitness_function(final_schedule))
