import numpy as np
import pandas as pd
from keras.losses import MSE
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer, RobustScaler
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.metrics import mean_squared_error as MSE


# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('../data/cclebeta.csv', index_col=0)
features = tpot_data.drop('Beta', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Beta'], random_state=123)

# Average CV score on the training set was: -1.285009615623126
exported_pipeline = make_pipeline(
    RobustScaler(),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=9, min_samples_leaf=14, min_samples_split=3)),
    Normalizer(norm="l1"),
    ElasticNetCV(l1_ratio=0.2, tol=0.01)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 123)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

# RMSE Computation
rmse = np.sqrt(MSE(testing_target, results))
print("RMSE : % f" %(rmse))

