from helper import *
import random

num_runs = 10000
beta_vals_range = (0.1, 1.0)
starting_pops_range = (5000, 10000)
doubling_times_range = (1, 10)
treatment_days = [1, 4, 7, 10]
alpha = 0.9
total_days = 16
target_ratio = [0.2328, 1.0, 0.0]
best_score = 0
best_beta = []
best_starting = []
best_doubling = []
best_final = []
best_normalized = []


for run in tqdm(range(num_runs)):


    beta_sub_pops = [round(random.uniform(beta_vals_range[0], beta_vals_range[1]), 2) for _ in range(5)]
    starting_sub_pops = [random.randint(starting_pops_range[0], starting_pops_range[1]) for _ in range(5)]
    doubling_sub_pops = [random.randint(doubling_times_range[0], doubling_times_range[1]) for _ in range(5)]

    final_pops = simulate_run(treatment_days, alpha, beta_sub_pops, starting_sub_pops, doubling_sub_pops, total_days, target_ratio)

    print(final_pops)
    normalized = normalizer(final_pops, 1)

    # print(normalized)

    # if normalized has any NaNs or Infs return array of 0
    if np.isnan(normalized).any() or np.isinf(normalized).any():
        # print('NaN or Inf found, skipping')
        normalized = [0, 0, 0]
        continue


