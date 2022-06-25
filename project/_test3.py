from cmath import nan
import sys
import pandas as pd
import numpy as np

def get_best_results_by_characteristics(dataset_name):
    #do with validations
    #do with validations
    #do with validations

    df_c = pd.read_csv(sys.path[0] + "/output/" + "kb_characteristics.csv", sep=",")

    # a = (1, 2, 3)
    # b = (4, 5, 6)
    # dst = np.linalg.norm(np.array(a) - np.array(b))
    # print(dst)

    #drop all columns where nan
    df_c = df_c.dropna(axis='columns')

    #get dataset_name row
    df_c_a = df_c.loc[df_c['dataset'] == dataset_name]
    #drop non meta features columns
    df_c_a = df_c_a.drop(['dataset', 'pre processing','algorithm'], axis=1)
    #get list "a"
    list_a = df_c_a.values.tolist()[0]
    #normalize values
    list_a = [(float(i)-min(list_a))/(max(list_a)-min(list_a)) for i in list_a]

    #drop dataset_name row
    df_c = df_c.loc[df_c['dataset'] != dataset_name]

    list_dist = []
    for index, row in df_c.iterrows():
        list_b = row.drop(['dataset', 'pre processing','algorithm']).values.tolist()
        list_b = [(float(i)-min(list_b))/(max(list_b)-min(list_b)) for i in list_b]
        list_dist.append((row['dataset'], row['pre processing'], row['algorithm'], np.linalg.norm(np.array(list_a) - np.array(list_b))))
        
    #list of tuples to dataframe
    df_dist = pd.DataFrame(list_dist, columns=["dataset", "pre processing", "algorithm","result"])
    #sort by result
    df_dist = df_dist.sort_values(by=['result'])
    
    # print(df_dist)
    
    #get first 3 elements
    df_dist = df_dist.head(3)
    
    return df_dist


dataset_name = "glass1.dat"
df_dist = get_best_results_by_characteristics(dataset_name)

print(df_dist)

