import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


# Define the parameter ranges for randomized search
beta_vals_range = (0.01, 1.0)
starting_pops_range = (20, 2000)
doubling_times_range = (1, 10)

# Define momentum-related variables
momentum_weight = 0.1  # Adjust the momentum weight as needed
previous_best_config = None

# Perform the randomized grid search with momentum
num_runs = 1000  # Number of randomized search runs
treatment_days = [1, 4, 7, 10]
alpha = 0.2
total_days = 16
threshold_score = 0.5  # Adjust the threshold score as needed
best_config = None
best_score = float("-inf")

def linear_quadratic_survival(alpha, beta, dose):
    surviving_fraction = np.exp((-alpha * dose) - (beta * (dose ** 2)))
    return surviving_fraction

def normalized_ratios(final_populations):
    normalizer = final_populations[0]
    normal_arr = []
    for pop in final_populations:
        pop = pop / normalizer
        normal_arr.append(pop)
    return normal_arr

def cosine_similarity_score(target_array, normalized_ratios):
    target_array = np.array(target_array).reshape(1, -1)
    normalized_array = np.array(normalized_ratios).reshape(1, -1)
    similarity = cosine_similarity(target_array, normalized_array)[0, 0]
    return similarity

def simulate_run(treatment_days, alpha, beta_sub_pops, starting_sub_pops, doubling_sub_pops, days_of_treatment):
    pop_arr = []
    normalizer_input = []
    dose_plan_g = [[4, 4, 4, 4], [1, 3, 5, 7], [7, 5, 3, 1]]
    for dose_plan in dose_plan_g:
        for enum, beta in enumerate(beta_sub_pops):
            dose_counter = 0
            total_pop = 0
            for day in range(days_of_treatment):
                population = starting_sub_pops[enum]
                dose = 0
                if population < 1:
                    # print("EARLY STOPPING AT DAY {}".format(day))
                    population = 0
                    break
                if day in treatment_days:
                    dose = dose_plan[dose_counter]
                    dose_counter += 1
                if doubling_sub_pops[enum] != 0 and day % doubling_sub_pops[enum] == 0:
                    population = max(0, starting_sub_pops[enum] * 2)  # Ensure non-negativity
                surviving_fraction = linear_quadratic_survival(alpha, beta, dose)
                population = max(0, surviving_fraction * population)  # Ensure non-negativity
                total_pop += population
            pop_arr.append(total_pop)
        total_population_final = sum(pop_arr)
        normalizer_input.append(total_population_final)

    normalized = normalized_ratios(normalizer_input)
    css = cosine_similarity_score([1, 5, 57], normalized)
    return css

for _ in tqdm(range(num_runs)):
    # Initialize parameter values
    if previous_best_config is not None and random.random() < momentum_weight:
        # Use the previous best configuration with some probability and apply small random perturbations
        beta_vals = [max(0, val + random.uniform(-0.1, 0.1)) for val in previous_best_config[0]]  # Ensure non-negativity
        starting_pops = [max(0, val + random.randint(-10, 10)) for val in previous_best_config[1]]  # Ensure non-negativity
        doubling_times = [max(0, val + random.randint(-1, 1)) for val in previous_best_config[2]]  # Ensure non-negativity
    else:
        # Randomly sample new parameter values
        beta_vals = [max(0, random.uniform(*beta_vals_range)) for _ in range(5)]  # Ensure non-negativity
        starting_pops = [max(0, random.randint(*starting_pops_range)) for _ in range(5)]  # Ensure non-negativity
        doubling_times = [max(0, random.randint(*doubling_times_range)) for _ in range(5)]  # Ensure non-negativity

    key = (tuple(beta_vals), tuple(starting_pops), tuple(doubling_times))
    score = simulate_run(treatment_days, alpha, beta_vals, starting_pops, doubling_times, total_days)

    # Implement early stopping: Skip further evaluation if score is below a threshold
    if score < threshold_score:
        continue

    # Update the best configuration and score
    if score > best_score:
        best_score = score
        best_config = key

    # Update the previous best configuration for momentum
    previous_best_config = key

# Print the best configuration
print("Best Configuration:")
print(
    f"Cosine Similarity Score: {best_score}, Beta Vals: {best_config[0]}, Starting Pops: {best_config[1]}, Doubling Times: {best_config[2]}")
