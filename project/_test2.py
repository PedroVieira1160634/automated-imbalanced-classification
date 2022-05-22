import sys
import pandas as pd
#from ml import read_file, features_labels

df_kb_c = pd.read_csv(sys.path[0] + "/output/" + "kb_characteristics.csv", sep=",")

df_kb_c2 = df_kb_c.loc[df_kb_c['dataset'] == "page-blocks0.dat"]

#print(df_kb_c2)

#print(df_kb_c2.loc[1, 'rows number'])

if( 
    df_kb_c2.loc[1, 'rows number'] == 0 and 
    df_kb_c2.loc[1, 'columns number'] == 0 and
    df_kb_c2.loc[1, 'numeric columns'] == 0 and 
    df_kb_c2.loc[1, 'non-numeric columns'] == 0 and
    df_kb_c2.loc[1, 'maximum correlation'] == 0 and 
    df_kb_c2.loc[1, 'average correlation'] == 0 and
    df_kb_c2.loc[1, 'minimum correlation'] == 0 and
    df_kb_c2.loc[1, 'average of distinct values in columns'] == 0 and
    df_kb_c2.loc[1, 'imbalance ratio'] == 0
):
    print("true")
else:
    print("false")
