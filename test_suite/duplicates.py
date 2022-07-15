import pandas as pd

df_in = pd.read_csv('good_data/rna_seq.csv')

# df_in = df_in.transpose()
#
# new_header = df_in.iloc[0] #grab the first row for the header
# df = df_in[1:] #take the data less the header row
# df.columns = new_header #
#
# df.to_csv('rna_seq.csv', sep=',')

f = open('good_data/RNA_duplicated_rows.csv', 'w')

# build header for new file
col = df_in.columns.tolist()
col[0] = 'cell_line'
headers = ','.join(col)
f.write(headers + '\n')

#append file names
for index,row in df_in.iterrows():
    row_r = [r for r in row [1:]]
    name = row[0]
    arr_int = ["2-R", "5-R", "8-R"]
    for i in range(len(arr_int)):
        name_i = name + '-' + str(arr_int[i])
        row_ri = [name_i] + row_r
        row_ri_str = [str(i) for i in row_ri]
        f.write(','.join(row_ri_str) + '\n')

f.close()

