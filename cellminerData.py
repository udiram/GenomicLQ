import pandas as pd
from scipy.stats import stats, gmean
from statistics import geometric_mean


df_in = pd.read_excel('RNA__Affy_HG_U133_Plus_2.0_RMA.xls')

df_in = df_in.groupby('Entrez gene id e').agg(gmean).reset_index()

df_in = df_in.transpose()

new_header = df_in.iloc[0] #grab the first row for the header
df = df_in[1:] #take the data less the header row
df.columns = new_header #

df.to_csv('RNA__Affy_HG_U133_Plus_2.0_RMA_entrez_gmean.csv', sep=',')


print(df_in.head())





