from sklearn.dummy import DummyRegressor

from helper import *
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.metrics import make_scorer
import numpy as np

# Define your target function to optimize
def target_function(params):
    beta_sub_pops = [params[f'beta_{i}'] for i in range(5)]
    starting_sub_pops = [int(params[f'start_{i}']) for i in range(5)]
    doubling_sub_pops = [int(params[f'double_{i}']) for i in range(5)]

    final_pops = simulate_run(treatment_days, alpha, beta_sub_pops, starting_sub_pops, doubling_sub_pops, total_days, target_ratio)

    normalized = normalized_ratios(final_pops)

    if np.isnan(normalized).any() or np.isinf(normalized).any():
        return -float('inf')  # Return a very low score for invalid parameters

    score = cosine_similarity_score(target_ratio, normalized)
    return -score  # Minimize the negative score

# Define a custom scoring function
def custom_scoring_function(estimator, X, y):
    return -target_function(estimator)

num_runs = 10000
param_space = {
    'constant': Real(0.01, 1.0, prior='uniform'),
    'constant': Real(0.01, 1.0, prior='uniform'),
    'constant': Real(0.01, 1.0, prior='uniform'),
    'constant': Real(0.01, 1.0, prior='uniform'),
    'constant': Real(0.01, 1.0, prior='uniform'),
    'constant': Integer(50000, 100000, prior='uniform', dtype=int),  # Specify dtype=int
    'constant': Integer(50000, 100000, prior='uniform', dtype=int),  # Specify dtype=int
    'constant': Integer(50000, 100000, prior='uniform', dtype=int),  # Specify dtype=int
    'constant': Integer(50000, 100000, prior='uniform', dtype=int),  # Specify dtype=int
    'constant': Integer(50000, 100000, prior='uniform', dtype=int),  # Specify dtype=int
    'constant': Integer(1, 10, prior='uniform'),
    'constant': Integer(1, 10, prior='uniform'),
    'constant': Integer(1, 10, prior='uniform'),
    'constant': Integer(1, 10, prior='uniform'),
    'constant': Integer(1, 10, prior='uniform'),
}

treatment_days = [1, 4, 7, 10]
alpha = 0.2
total_days = 16
target_ratio = [56, 240, 1]
# Create dummy input data with the same number of samples as target_ratio
dummy_input_data = np.zeros((len(target_ratio), 1))
# Create a custom scorer using the custom scoring function
custom_scorer = make_scorer(custom_scoring_function, greater_is_better=False)

# Use a DummyRegressor as the estimator
estimator = DummyRegressor(strategy='mean')  # You can choose a strategy based on your use case

optimizer = BayesSearchCV(
    estimator=estimator,  # Use the DummyRegressor as the estimator
    search_spaces=param_space,
    scoring=custom_scorer,
    n_iter=num_runs,
    cv=3,
    random_state=0,
    verbose=1,
    n_jobs=-1
)

best_params = optimizer.fit(dummy_input_data, target_ratio).best_params_

# Print the best parameters
print('Best Parameters:')
print(best_params)