import sys
from decimal import Decimal
import pandas as pd
import openml.datasets
import traceback
from ml import read_file_openml

def step_1():

    openml_list = openml.datasets.list_datasets()  # returns a dict
    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    # print(datalist)
    #4124 datasets

    #NumberOfInstancesWithMissingValues == 0
    datalist = datalist.query("status == 'active' & NumberOfClasses == 2 & NumberOfInstances > 200 & NumberOfInstances < 10000 & NumberOfFeatures < 500 & NumberOfInstancesWithMissingValues > 0")

    # print(datalist)

    #1st run
    #problem at line 210 - id:40588 ...
    # print(datalist.iloc[210,:])
    # 337 datasets -> 267 datasets
    
    #2nd run
    #...


    # i=0
    # for id in datalist["did"]:
    #     print(i)
    #     i +=1
        
    #     try:
    #         dataset = openml.datasets.get_dataset(id)

    #         X, y, categorical_indicator, attribute_names = dataset.get_data(
    #             target=dataset.default_target_attribute, dataset_format="dataframe")

    #         df = pd.DataFrame(X, columns=attribute_names)
    #         df["class"] = y

    #         #X = df.iloc[: , :-1]
    #         y = df.iloc[: , -1:]

    #         df = df.dropna()

    #         #print(y["class"].dtype.name)

    #         encoded_columns = []
    #         for column_name in y.columns:
    #             if y[column_name].dtype == object or y[column_name].dtype.name == 'category':
    #                 encoded_columns.extend([column_name])
    #             else:
    #                 pass

    #         if encoded_columns:
    #             y = pd.get_dummies(y, y[encoded_columns].columns, drop_first=True)


    #         imbalance_ratio = 0
    #         if y.values.tolist().count([0]) > 0 and y.values.tolist().count([1]) > 0:
    #             if y.values.tolist().count([0]) >= y.values.tolist().count([1]):
    #                 imbalance_ratio = round(y.values.tolist().count([0])/y.values.tolist().count([1]),3)
    #             else:
    #                 imbalance_ratio = round(y.values.tolist().count([1])/y.values.tolist().count([0]),3)

    #         # print("imbalance ratio      :", imbalance_ratio)
    #         # print("")
            
    #         #datasets_openml.csv
    #         df_openml = pd.read_csv(sys.path[0] + "/input/" + "datasets_openml_missingvalues.csv", sep=",")
            
    #         df_openml.loc[len(df_openml.index)] = [Decimal(id), imbalance_ratio]
            
    #         #datasets_openml.csv
    #         df_openml.to_csv(sys.path[0] + "/input/" + "datasets_openml_missingvalues.csv", sep=",", index=False)
        
    #     except:
    #         print("An exception occurred with id", id)
    #         traceback.print_exc()
    #         print("\n")
    
    print("end")

def step_2():
    #datasets_openml.csv
    df_openml = pd.read_csv(sys.path[0] + "/input/" + "datasets_openml_missingvalues.csv", sep=",")

    #exel results - openml csv
    
    # rows_to_keep = df_openml["imbalance ratio"] > 5
    # df_openml = df_openml[rows_to_keep]
    
    rows_to_keep = df_openml["imbalance ratio"] <= 2.5
    df_openml = df_openml[rows_to_keep]
    rows_to_keep = df_openml["imbalance ratio"] > 2
    df_openml = df_openml[rows_to_keep]
    

    df_openml = df_openml.sort_values(by=['imbalance ratio'])

    pd.set_option('display.max_rows', df_openml.shape[0]+1)
    print(df_openml)
    # print("\ncount:")
    print(df_openml.count())
    
    # print(df_openml.sample(n=10))
    
    print("end")
    
def step_3():
    df, dataset_name = read_file_openml(951)

    print("\ndf:")
    print(df)

    print("\ndataset_name:")
    print(dataset_name)


# step_1()
step_2()
# step_3()
