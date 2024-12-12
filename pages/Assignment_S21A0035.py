import streamlit as st
import csv
import random
import pandas as pd

def read_csv_to_dict(file):
    program_ratings = {}
    reader = csv.reader(file)
    header = next(reader)  # Skip header

    for row in reader:
        program = row[0]
        ratings = [float(x) for x in row[1:]]
        program_ratings[program] = ratings

    return program_ratings

# Streamlit UI
st.title("Optimal Program Scheduler")
st.sidebar.header("Genetic Algorithm Parameters")

# Sidebar inputs for genetic algorithm parameters
CO_R = st.sidebar.slider("Crossover Rate", min_value=0.0, max_value=0.95, value=0.8, step=0.01)
MUT_R = st.sidebar.slider("Mutation Rate", min_value=0.01, max_value=0.05, value=0.02, step=0.01)
GEN = st.sidebar.number_input("Generations", min_value=10, max_value=500, value=100, step=10)
POP = st.sidebar.number_input("Population Size", min_value=10, max_value=200, value=50, step=10)
EL_S = st.sidebar.number_input("Elitism Size", min_value=1, max_value=10, value=2, step=1)

# File uploader
uploaded_file = st.file_uploader("Upload the CSV file with program ratings", type=["csv"])

if uploaded_file:
    program_ratings_dict = read_csv_to_dict(uploaded_file)
    all_programs = list(program_ratings_dict.keys())
    all_time_slots = list(range(6, 24))  # Time slots from 6:00 to 23:00

    def fitness_function(schedule):
        total_rating = 0
        for time_slot, program in enumerate(schedule):
            total_rating += program_ratings_dict[program][time_slot]
        return round(total_rating, 2)

    def initialize_pop(programs, time_slots):
        if not programs:
            return [[]]

        all_schedules = []
        for i in range(len(programs)):
            for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
                all_schedules.append([programs[i]] + schedule)

        return all_schedules

    def finding_best_schedule(all_schedules):
        best_schedule = []
        max_ratings = 0

        for schedule in all_schedules:
            total_ratings = fitness_function(schedule)
            if total_ratings > max_ratings:
                max_ratings = total_ratings
                best_schedule = schedule

        return best_schedule

    def crossover(schedule1, schedule2):
        crossover_point = random.randint(1, len(schedule1) - 2)
        child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
        child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
        return child1, child2

    def mutate(schedule):
        mutation_point = random.randint(0, len(schedule) - 1)
        new_program = random.choice(all_programs)
        schedule[mutation_point] = new_program
        return schedule

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

    all_possible_schedules = initialize_pop(all_programs, all_time_slots)
    initial_best_schedule = finding_best_schedule(all_possible_schedules)
    rem_t_slots = len(all_time_slots) - len(initial_best_schedule)

    genetic_schedule = genetic_algorithm(initial_best_schedule, generations=GEN, population_size=POP, elitism_size=EL_S)
    final_schedule = initial_best_schedule + genetic_schedule[:rem_t_slots]

    # Displaying the final schedule
    st.subheader("Final Optimal Schedule")

    data = {
        "Time Slot": [f"{hour}:00" for hour in all_time_slots],
        "Program": final_schedule
    }

    schedule_table = pd.DataFrame(data)
    st.table(schedule_table)

    st.write(f"**Total Ratings:** {fitness_function(final_schedule):.2f}")
else:
    st.warning("Please upload a CSV file to proceed.")
