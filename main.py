import numpy as np
import matplotlib.pyplot as plt
def linear_quadratic_survival(alpha, beta, dose, number_of_cells):
    surviving_fraction = number_of_cells * np.exp((-alpha * dose) - (beta * (dose ** 2)))
    return surviving_fraction
days_of_treatment = 30
start_dose = 0
end_dose = 4
ramp_up = np.linspace(start_dose, end_dose, days_of_treatment)
ramp_down = np.linspace(end_dose, start_dose, days_of_treatment)
a = 0.2
pop_arr = []
beta_combos = [0.01, 0.06, 0.11, 0.16, 0.21]
pop_strat_arr = [1000, 5000, 10000, 5000, 1000]
doubling_time = [2, 3, 4, 5, 6]
final_sum = []
for enum, beta in enumerate(beta_combos):
    population = pop_strat_arr[enum]
    for i in range(0, days_of_treatment):
        dose_in_gy = ramp_down[i]
        # print('day {}'.format(i))
        print(population)
        if population < 1:
            print('population is less than 1 for beta = {} at day {}'.format(beta, i))
            break
        if i % doubling_time[enum] == 0:
            population *= 2
        # print('population: {}'.format(start_pop))
        surviving_fraction = linear_quadratic_survival(a, beta, dose_in_gy, population)
        # print('surviving fraction: {}'.format(surviving_fraction))
        population = surviving_fraction
        pop_arr.append(population)
    plt.plot(pop_arr, label='beta = {}'.format(beta))
    surviving_population_final = pop_arr[-1]
    print('surviving population: {}'.format(surviving_population_final))
    final_sum.append(surviving_population_final)
    plt.title('Population vs. Time')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    pop_arr = []
print('final sum: {}'.format(sum(final_sum)))
plt.legend()
plt.show()