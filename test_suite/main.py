import pandas as pd

df_in = pd.read_csv('input/y_train_ccle_alphaRNA.csv')
df_gt = pd.read_csv('input/CTD2_raw_output.csv')
df_master = df_in.set_index('cell_line').join(df_gt.set_index('cell_line'), on='cell_line',
                                              how='left').loc[:, ['Alpha', '2', '3', '4', '5', '6', '8', '10']]

df_master.to_csv('updated_y_train_ccle_alpha.csv', index=True)
print(len(df_master))