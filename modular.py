import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random


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
                    print("EARLY STOPPING AT DAY {}".format(day))
                    population = 0
                    break
                if day in treatment_days:
                    dose = dose_plan[dose_counter]
                    dose_counter += 1
                if day % doubling_sub_pops[enum] == 0:
                    population = starting_sub_pops[enum] * 2
                surviving_fraction = linear_quadratic_survival(alpha, beta, dose)
                population = surviving_fraction * population
                total_pop += population
            pop_arr.append(total_pop)
        total_population_final = sum(pop_arr)
        normalizer_input.append(total_population_final)
        print('total population {}, for dose plan {}'.format(total_population_final, dose_plan))

    normalized = normalized_ratios(normalizer_input)
    print(normalized)

    css = cosine_similarity_score([1, 5, 57], normalized)
    print(css)

    return css

treatment_days = [1, 4, 7, 10]
alpha = 0.2
total_days = 16

score = simulate_run(treatment_days, alpha, beta_vals, starting_pops, doubling_times, total_days)
