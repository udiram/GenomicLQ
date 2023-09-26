import matplotlib.pyplot as plt
import numpy as np

def linear_quadratic_survival(alpha, beta, dose, number_of_cells):
    surviving_fraction = number_of_cells * np.exp((-alpha * dose) - (beta * (dose ** 2)))
    return surviving_fraction

days_of_treatment = 16
# start_dose = 1
# end_dose = 10
# number_of_doses = 4
# treatment_interval = round(days_of_treatment / number_of_doses)
# print('treatment every: {} days'.format(treatment_interval))
# ramp_up = np.arange(start_dose, end_dose, (end_dose - start_dose) / number_of_doses)
# ramp_up = [1, 3, 5, 7]
# print('ramp up: {}'.format(ramp_up))
# # ramp_down = np.arange(end_dose, start_dose, (start_dose - end_dose) / number_of_doses)
# ramp_down = [7,5,3,1]
# print('ramp down: {}'.format(ramp_down))
alpha = 0.2
beta_sub_pop = [0.01, 0.06, 0.11, 0.16, 0.21]
# beta_sub_pop = [0.0740744719754168, 0.35681948822515697, 0.11336202777681112, 0.047351293614121145, 0.5816749695526988]
pop_strat_arr = [1000, 5000, 10000, 5000, 1000]
# pop_strat_arr = [49899, 5592, 9428, 15492, 16450]
doubling_time = [2, 3, 4, 5, 6]
# doubling_time = [10, 7, 10, 5, 5]
treatment_days = [1, 4, 7, 10]
total = []
for dose_plan in [[4, 4, 4, 4], [1, 3, 5, 7], [7, 5, 3, 1]]:
    for enum, beta in enumerate(beta_sub_pop):
        pop_arr = []
        # print('doubling period for beta = {} is every {} days'.format(beta, doubling_time[enum]))
        starting_population = pop_strat_arr[enum]
        # print('starting population: {}'.format(starting_population))
        dose_counter = 0
        for day in range(0, days_of_treatment):
            # print('day: {}, population: {}'.format(day, starting_population))
            dose = 0
            if starting_population < 1:
                # print('population is less than 1 for beta = {} at day {}'.format(beta, day))
                starting_population = 0
                break
            if day in treatment_days:
                dose = dose_plan[dose_counter]
                dose_counter += 1
                # print("total doses administered: {}".format(dose_counter))
                # print('dose: {} administered on day {}'.format(dose, day))
            if day % doubling_time[enum] == 0:
                # print('doubling on day {}'.format(day))
                starting_population *= 2

            surviving_fraction = round(linear_quadratic_survival(alpha, beta, dose, starting_population))

            starting_population = surviving_fraction
            pop_arr.append(starting_population)
        plt.plot(pop_arr, label='beta = {}'.format(beta))
        plt.title('Population vs. Time')
        plt.xlabel('Time (days)')
        plt.ylabel('Population')
        plt.legend()
        # print('final population for beta = {}: {}, survived {} days'.format(beta, pop_arr[-1], len(pop_arr)))
        total.append(pop_arr[-1])
    print('total population: {} with dose plan {}'.format(sum(total), dose_plan))
    plt.show()

