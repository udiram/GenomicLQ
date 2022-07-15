import pandas as pd

df_in = pd.read_csv('good_data/RNA_duplicated_rows.csv')
f = open('good_data/rna_seq_structured_258.csv', 'w')

f.write('cell_line,dose\n')
for index,row in df_in.iterrows():
    name = row[0]
    dose = name[-3]
    f.write('{},{}\n'.format(name, dose))

f.close()
