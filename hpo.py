import numpy as np
from itertools import combinations, product

pop_strat_arr = [1000, 1000, 1000, 1000, 1000]
alpha = 0.2
treatment_days = [1, 4, 7, 10]
days_of_treatment = 16
dose_global_arr = [[0.1, 0.1, 0.1, 0.1], [4, 4, 4, 4], [1, 3, 5, 7], [7, 5, 3, 1]]
beta_global_arr = [[0.001, 0.001, 0.001, 0.001, 0.001], [0.06, 0.06, 0.06, 0.06, 0.06], [0.11, 0.11, 0.11, 0.11, 0.11],
                   [0.16, 0.16, 0.16, 0.16, 0.16], [0.21, 0.21, 0.21, 0.21, 0.21]]
doubling_time_global_arr = [[2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4], [5, 5, 5, 5, 5], [6, 6, 6, 6, 6]]
import random


def gen_combos(start, end):

    # Define the dimensions of your array
    rows = 10000
    cols = 5

    # Initialize an empty array
    beta_global_arr = []

    # Loop to generate random values
    for _ in range(rows):
        row = [round(random.uniform(0.001, 0.9), 3) for _ in range(cols)]
        beta_global_arr.append(row)

    return beta_global_arr
#
beta_combo_array = gen_combos(0.001, 0.9)
doubling_combo_array = gen_combos(0.25, 15)

print(len(beta_combo_array))
print(len(doubling_combo_array))

def linear_quadratic_survival(alpha, beta, dose):
    surviving_fraction = np.exp((-alpha * dose) - (beta * (dose ** 2)))
    return surviving_fraction


def simulate(alpha, beta_global_arr, pop_strat_arr, doubling_times, treatment_days, days_of_treatment, dose_global_arr):

    global pop_arr
    total = []

    for dose_combo in dose_global_arr:
        print('dose combo: {}'.format(dose_combo))

        for beta_arr in beta_global_arr:
            print('beta arr: {}'.format(beta_arr))

            for doubling_time in doubling_times:
                print('doubling time: {}'.format(doubling_time))

                for enum, beta in enumerate(beta_arr):
                    print('beta: {}'.format(beta))

                    pop_arr = []
                    population = pop_strat_arr[enum]
                    dose_counter = 0

                    for day in range(0, days_of_treatment):
                        if population < 1:
                            print("EARLY STOPPING AT DAY {}".format(day))
                            break

                        if day in treatment_days:
                            dose = dose_combo[dose_counter]
                            dose_counter += 1

                        else:
                            dose = 0

                        if day % doubling_time[enum] == 0:
                            population *= 2

                        surviving_fraction = linear_quadratic_survival(alpha, beta, dose)
                        population = surviving_fraction * population

                        pop_arr.append(population)
                    total.append(pop_arr)
                    final_population = sum(pop_arr)
                    print('final population: {}'.format(final_population))
    return total



total = simulate(alpha, beta_global_arr, pop_strat_arr, doubling_time_global_arr, treatment_days, days_of_treatment,
                 dose_global_arr)
