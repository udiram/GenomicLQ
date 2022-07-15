import pandas as pd

df_in = pd.read_csv('x_train_234568.csv')
f = open('input/x_train_structured_234568.csv', 'w')

f.write('cell_line,dose\n')
for index,row in df_in.iterrows():
    name = row[0]
    dose = name[-1]
    f.write('{},{}\n'.format(name, dose))

f.close()
