import pandas as pd

df_in = pd.read_csv('input/X_test_ccle_alpha (1).csv')
f = open('x_test_234568.csv', 'w')

# build header for new file
col = df_in.columns.tolist()
col[0] = 'cell_line'
headers = ','.join(col)
f.write(headers + '\n')

#append file names
for index,row in df_in.iterrows():
    row_r = [r for r in row [1:]]
    name = row[0]
    arr_int = [2,3,4,5,6,8]
    for i in range(len(arr_int)):
        name_i = name + '-' + str(arr_int[i])
        row_ri = [name_i] + row_r
        row_ri_str = [str(i) for i in row_ri]
        f.write(','.join(row_ri_str) + '\n')

f.close()

