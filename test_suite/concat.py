import pandas as pd

df_in = pd.read_csv('good_data/entrez_structured_258.csv')

df_in['cell_line'] = df_in['cell_line'].str[3:]


df_in.to_csv("entrez_struc_258_3rem.csv", index=False)

