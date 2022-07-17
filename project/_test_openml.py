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

    # imbalance ratio > 10              -> 22 datasets
    # imbalance ratio > 5               -> 63 datasets
    # imbalance ratio > 2.5 and < 5     -> 21 datasets
    # imbalance ratio > 2   and < 2.5   -> 14 datasets

    rows_to_keep = df_openml["imbalance ratio"] > 5
    df_openml = df_openml[rows_to_keep]
    
    # rows_to_keep = df_openml["imbalance ratio"] <= 2.5
    # df_openml = df_openml[rows_to_keep]
    # rows_to_keep = df_openml["imbalance ratio"] > 2
    # df_openml = df_openml[rows_to_keep]
    

    df_openml = df_openml.sort_values(by=['imbalance ratio'])

    pd.set_option('display.max_rows', df_openml.shape[0]+1)
    print(df_openml)
    print(df_openml.count())
    
    # print("\ncount:")
    # print(df_openml.count())
    
    #get X random datasets whithout ...
    # df_openml = df_openml.loc[(df_openml["id"] != 1069) &   #bad metrics
    #                           (df_openml["id"] != 1506) & 
    #                           (df_openml["id"] != 1452) & 
    #                           (df_openml["id"] != 43905) & 
                              
    #                           (df_openml["id"] != 1056) &   #too long
                              
    #                           (df_openml["id"] != 43897) &  #perfect metrics
    #                           (df_openml["id"] != 43895) & 
    #                           (df_openml["id"] != 40666) &
                              
    #                           (df_openml["id"] != 951) &    #testing UI
    #                           (df_openml["id"] != 43051) &
    #                           (df_openml["id"] != 1451) &
    #                           (df_openml["id"] != 40713) &
    #                           (df_openml["id"] != 1487) &
    #                           (df_openml["id"] != 1116) &
    #                           (df_openml["id"] != 41538) &
    #                           (df_openml["id"] != 4329) &
    #                           (df_openml["id"] != 1447) &
    #                           (df_openml["id"] != 949) &
                              
    #                           (df_openml["id"] != 1069) &   #already trainned
    #                           (df_openml["id"] != 40994) &
    #                           (df_openml["id"] != 450) &
    #                           (df_openml["id"] != 41946) &
    #                           (df_openml["id"] != 1020) &
    #                           (df_openml["id"] != 1506) &
    #                           (df_openml["id"] != 765) &
    #                           (df_openml["id"] != 43894) &
    #                           (df_openml["id"] != 40900) &
    #                           (df_openml["id"] != 1452) &
    #                           (df_openml["id"] != 1558) &
    #                           (df_openml["id"] != 1467) &
    #                           (df_openml["id"] != 995) &
    #                           (df_openml["id"] != 43905) &
    #                           (df_openml["id"] != 958) &
    #                           (df_openml["id"] != 1065) &
    #                           (df_openml["id"] != 1067) &
    #                           (df_openml["id"] != 764) &
    #                           (df_openml["id"] != 311) &
    #                           (df_openml["id"] != 1068) &
    #                           (df_openml["id"] != 43893) &
    #                           (df_openml["id"] != 1022) &
    #                           (df_openml["id"] != 976) &
    #                           (df_openml["id"] != 1444) &
    #                           (df_openml["id"] != 954) &
    #                           (df_openml["id"] != 1050) &
    #                           (df_openml["id"] != 1446) &
    #                           (df_openml["id"] != 767) &
    #                           (df_openml["id"] != 1049) &
    #                           (df_openml["id"] != 980)
    #                           ]
    
    # print(df_openml)
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
