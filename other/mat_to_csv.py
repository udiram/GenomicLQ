import pandas

dataframe = pandas.read_csv("NCI60AlphaBeta (1).txt",delimiter="\t")
dataframe.to_csv("NCI_60ab.csv", encoding='utf-8', index=False)


