# import pandas as pd
#
# df_in = pd.read_csv('good_data/updated_y_train_ccle_alpha.csv', usecols=[0, 1, 2, 3, 4, 5, 6, 7])
# f = open('good_data/y_train_sf_234568.csv', 'w')
#
# # build header for new file
# f.write('cell_line,SF\n')
#
# #append file names
# for index,row in df_in.iterrows():
#     name = row[0]
#     arr_int = [2,3,4,5,6,8]
#
#     for i in range(len(arr_int)):
#         SF = str(arr_int[i])
#         name_i = name + '-' + SF
#         sf = df_in.loc[index, SF]
#         f.write('{},{}\n'.format(name_i, sf))
#
# f.close()


import pandas as pd

df_in = pd.read_csv('good_data/entrez_survival_fractions.csv', usecols=[0, 1, 2, 3])
f = open('good_data/rna_seq_sf.csv', 'w')

# build header for new file
f.write('cell_line,SF\n')

#append file names
for index,row in df_in.iterrows():
    name = row[0]
    print(name)
    arr_int = [2,5,8]

    for i in range(len(arr_int)):
        SF = str(arr_int[i])
        name_i = name + '-' + SF + "-R"
        print(name_i)
        print(SF)
        print(index)
        sf = df_in.loc[index, SF]
        f.write('{},{}\n'.format(name_i, sf))

f.close()