# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline, make_union
# from sklearn.svm import LinearSVR
# from tpot.builtins import StackingEstimator
# from xgboost import XGBRegressor
# from tpot.export_utils import set_param_recursive
#
# # NOTE: Make sure that the outcome column is labeled 'target' in the data file
# tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
# features = tpot_data.drop('target', axis=1)
# training_features, testing_features, training_target, testing_target = \
#             train_test_split(features, tpot_data['target'], random_state=123)
#
# # Average CV score on the training set was: -0.4775086660707946
# exported_pipeline = make_pipeline(
#     StackingEstimator(estimator=LinearSVR(C=0.01, dual=False, epsilon=0.0001, loss="squared_epsilon_insensitive", tol=0.001)),
#     XGBRegressor(learning_rate=0.1, max_depth=3, min_child_weight=2, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.7500000000000001, verbosity=0)
# )
# # Fix random state for all the steps in exported pipeline
# set_param_recursive(exported_pipeline.steps, 'random_state', 123)
#
# exported_pipeline.fit(training_features, training_target)
# results = exported_pipeline.predict(testing_features)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ og above
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive
import matplotlib.pyplot as plt

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('../data/cclebeta.csv', index_col=0)
features = tpot_data[['3859', '283352', '10653', '286827', '100506542', '51626', '4065', '23242', '118430', '5172', '250', '9615', '4145', '119395', '169834', '100507178', '503693', '7369', '56180', '26103', '2312', '10170', '200159', '170622', '80317', '83887', '5480', '590', '8204', '115361', '84141', '1825', '2067', '100127950', '50852', '80759', '55752', '148109', '389840', '83450', '23052', '10962', '9369', '5325', '948', '90410', '6338', '6457', '5617', '389932', '646962', '643616', '56931', '57480', '7699', '323', '55966', '3883', '100653022', '4660', '84766', '4680', '100128285', '8477', '7739', '286530', '84167', '26080', '399670', '120114', '6048', '164284', '22852', '30009', '378828', '195814', '27342', '222962', '29761', '10845', '100505679', '114900', '4118', '91947', '163183', '283576', '8418', '9750', '4033', '147746', '140807', '140886', '7032', '56034', '135932', '642426', '283601', '125972', '54457', '400566', '113540', '118461', '284083', '89792', '25894', '2068', '128486', '81792', '1026', '80185', '400940', '54737', '282980', '339874', '8518', '54545', '100128098', '171177', '255743', '85301', '100131138', '113246', '90141', '92292', '2181', '84752', '79822', '388698', '83881', '643677', '80210', '9696', '11153', '123041', '118426', '285957', '256227', '54360', '11227', '9290', '28999', '26164', '149461', '374739', '5816', '165904', '399949', '10548', '8386', '3741', '4781', '136', '56925', '164684', '3730', '55250', '79166', '135152', '261734', '9488', '55425', '56914', '27098', '100130950', '1075', '389289', '53944', '55150', '144776', '90427', '65055', '132660', '23040', '85509', '729', '56659', '80274', '150763', '79998', '7803', '27032', '5562', '7515', '4126', '50839', '285613', '390', '118442', '182', '9411', '126661', '285386', '4100', '23178', '253573', '3488', '6872', '116123', '282775', '208', '6422', '734', '29114', '27039', '23767', '8970', '2784', '85415', '64600', '149620', '374286', '6160', '140686', '84525', '79074', '79083', '100129361', '81930', '151278', '112942', '146713', '8110', '54544', '1016', '6549', '8581', '284353', '27020', '90139', '1345', '4311', '168507', '100506606', '170712', '563', '1401', '1591', '148823', '6193', '200132', '1060', '81696', '7766', '5342', '642636', '9099', '139322', '60686', '148898', '23116', '51660', '8365', '84692', '285498', '128817', '57459', '140609', '650794', '63933', '100270679', '91646', '440730', '27122', '79443', '1379', '65059', '1294', '83468', '2737', '51181', '8292', '149351', '2559', '4354', '100129175', '9310', '27443', '1795', '84288', '9607', '10875', '51060', '25941', '119710', '10866', '9757', '56259', '11136', '6721', '56261', '84342', '640', '117531', '57710', '151176', '57732', '25806', '9961', '245911', '6619', '202333', '3856', '158358', '339976', '91120', '55297', '150962', '3437', '4744', '79583', '79365', '158234', '23677', '100129387', '1109', '1938', '9093', '9754', '56882', '1010', '203074', '460', '916', '54465', '5266', '29118', '4092', '54807', '134111', '4626', '9768', '2290', '11254', '26577', '286272', '85365', '5349', '29999', '1991', '57001', '2170', '51705', '100131510', '5592', '130574', '1734', '54913', '23232', '57531', '84099', '29760', '25963', '6005', '690', '28986', '84726', '162494', '84300', '6670', '494470', '51806', '115106', '728485', '59286', '23464', '10884', '9524', '10695', '23237', '26136', '355', '7093', '389421', '93058', '9024', '8360', '64433', '63926', '85236', '8344', '114800', '7024', '375612', '84223', '80727', '5636', '122786', '280664', '79784', '280665', '130557', '1545', '89846', '90024', '127544', '100506639', '7341', '9475', '50840', '84302', '81', '57169', '23308', '23351', '8834', '57379', '645158', '4628', '54221', '55286', '128025', '6865', '100129196', '59272', '55387', '1032', '6368', '4731', '5924', '80144', '29988', '283078', '340252', '266675', '1499', '100506530', '132299', '6563', '130013', '100499467', '219347', '51010', '162966', '3239', '158471', '55532', '100129455', '170062', '100507362', '201595', '732253', '7549', '2590', '4199', '27294', '133957', '317703', '7673', '115201', '1038', '100506049', '5646', '28983', '10814', '348327', '3350', '3400', '51337', '339210', '6677', '3875', '3120', '2015', '10287', '100133319', '971', '643733', '27199', '100133205', '26297', '54766', '147912', '282997', '1605', '203414', '140606', '2018', '283871', '8828', '27147', '51530', '84777', '4241', '573', '9851', '642587', '7031', '64782', '9223', '97', '219527', '29110', '56903', '1455', '51147', '200407', '160857', '5675', '51297', '57697', '23378', '81539', '118788', '51450', '2523', '2299', '221091', '9013', '140801', '121793', '55068', '100292680', '7350', '283685', '400709', '7709', '283848', '4225', '150005', '307', '84629', '9380', '25996', '6041', '51351', '84626', '404201', '23300', '140461', '84809', '3157', '11202', '339535', '140597', '5352', '646383', '5777', '10265', '339290', '84779', '2017', '6764', '55268', '8844', '54780', '644128', '926', '84186', '114788', '1489', '9684', '55350', '653308', '79940', '10499', '4885', '7409', '7479', '7626', '146712', '114795', '133688', '10473', '2321', '6227', '6123', '7844', '10261', '5338', '80318', '22824', '80124', '442721', '57369', '8535', '38', '7768', '57787', '83549', '84900', '29934', '55412', '369', '100507398', '9294', '10198', '285533', '8521', '9940', '3363', '22828', '55503', '10551', '283486', '50632', '84634', '753', '857', '27077', '93190', '285555', '5669', '55213', '126231', '339366', '121391', '2566', '283869', '339400', '55601', '51233', '9097', '1284', '55662', '23534', '146225', '10730', '196993', '3590', '2571', '8347', '10656', '1282', '2358', '286676', '22903', '2962', '4610', '146956', '6993', '92691', '8566', '170082', '4322', '3216', '151242', '100506328', '56154', '260434', '388394', '1232', '79930', '54926', '80078', '84278', '83642', '2706', '51765', '8562', '64423', '706', '284323', '375484', '27236', '23765', '5138', '64208', '65997', '2117', '5336', '22949', '57501', '375791', '79969', '84077', '8125', '57150', '91966', '5744', '339988', '23259', '8809', '283129', '23601', '54967', '63923', '3329', '10010', '23277', '84959', '56099', '2245', '3422', '541565', '400748', '1613', '9693', '79400', '4110', '84152', '30847', '9984', '84455', '55734', '5720', '55195', '4773', '149773', '51230', '144406', '100303728', '768096', '285848', '55283', '84695', '6654', '163479', '201477', '2850', '10647', '221687', '374378', '27076', '55125', '79183', '9751', '10780', '219790', '23528', '100129195', '285855', '64759', '10316', '95681', '54981', '79695', '6283', '100505696', '8655', '5293', '9532', '9656', '4633', '23555', '2593', '8019', '644246', '4632', '10150', '131870', '219287', '339479', '374860', '8372', '257000', '84280', '148113', '7329', '65078', '3781', '10905', '6642', '60560', '115572', '29104', '84844', '2259', '55225', '83544', '64843', '1365', '6507', '131616', '158376', '11315', '81627', '3577', '2879', '9997', '255043', '6662', '5875', '100507316', '23370', '79750', '10911', '2550', '55471', '5791', '1454', '5319', '375444', '9808', '158586', '643', '57061', '59307', '154215', '2086', '27244', '412', '100128927', '114042', '8287', '8688', '4919', '9722', '57560', '55776', '386758', '122830', '1520', '23613', '57126', '64207', '91137', '55076', '60492', '10155', '91574', '2799', '4624', '27101', '665', '908', '5261', '145173', '57758', '57530', '799', '9692']]
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Beta'], random_state=123)

# Average CV score on the training set was: -0.4775086660707946
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LinearSVR(C=0.01, dual=False, epsilon=0.0001, loss="squared_epsilon_insensitive", tol=0.001)),
    XGBRegressor(learning_rate=0.1, max_depth=3, min_child_weight=2, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.7500000000000001, verbosity=0)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 123)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
# RMSE Computation
rmse = np.sqrt(MSE(testing_target, results/100))
print("RMSE : % f" %(rmse))
# R2 Computation
import scipy.stats as stats
# print(y_test.tolist())
# print(y_pred_list)
plt.plot(testing_target, results, 'o')
plotting_df = pd.DataFrame()
plotting_df['Actual'] = testing_target
plotting_df['Predicted'] = results/100
plotting_df.to_csv('../data/plotting.csv')
plotting_df.plot(x='Actual', y='Predicted', kind='scatter')

a, b = np.polyfit(testing_target, results/100, 1)
plt.plot(testing_target, a*testing_target+b, color='steelblue', linestyle='--', linewidth=2)
slope, intercept, r_value, p_value, std_err = stats.linregress(testing_target, results/100)
print("Slope:", slope)
print("Intercept:", intercept)
print("R-Squared:", r_value**2)
print("P-Value:", p_value)
print("Standard Error:", std_err)
plt.xlabel('Test Labels')
plt.ylabel('Predictions')
plt.title('Test Labels vs. Predictions')
plt.show()
