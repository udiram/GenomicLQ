import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def normalizer(data, reference_index):
    reference_value = data[reference_index]
    normalized_data = [x / reference_value for x in data]
    return normalized_data

def linear_quadratic_survival(alpha, beta, dose):
    surviving_fraction = np.exp((-alpha * dose) - (beta * (dose ** 2)))
    return surviving_fraction

def cosine_similarity_score(target_array, normalized_ratios):
    target_array = np.array(target_array).reshape(1, -1)
    normalized_array = np.array(normalized_ratios).reshape(1, -1)
    similarity = cosine_similarity(target_array, normalized_array)[0, 0]
    return similarity

def simulate_run(treatment_days, alpha, beta_sub_pops, starting_sub_pops, doubling_sub_pops, days_of_treatment, target_ratio):
    pop_arr = []
    normalizer_input = []
    dose_plan_g = [[4, 4, 4, 4], [1, 3, 5, 7], [7, 5, 3, 1]]
    for dose_plan in dose_plan_g:
        fin_pop_arr = []
        for enum, beta in enumerate(beta_sub_pops):
            dose_counter = 0
            total_pop = 0
            for day in range(days_of_treatment):
                dose = 0
                if day == 0:
                    population = starting_sub_pops[enum]
                if population < 1:
                    population = 0
                    break
                if day in treatment_days:
                    dose = dose_plan[dose_counter]
                    dose_counter += 1
                if day % doubling_sub_pops[enum] == 0 and day != 0:
                    population = population * 2
                surviving_fraction = round(linear_quadratic_survival(alpha, beta, dose), 2)
                population = round(population * surviving_fraction, 1)
                pop_arr.append(population)
            # print('total_population for beta = {} is {}'.format(beta, pop_arr[-1]))
            fin_pop_arr.append(pop_arr[-1])
        # print('total population for dose plan {} is {}'.format(dose_plan, sum(fin_pop_arr)))
        normalizer_input.append(sum(fin_pop_arr))
    print('normalizer input', normalizer_input)

    return normalizer_input
