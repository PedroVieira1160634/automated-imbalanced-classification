import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from ml import read_file, read_file_openml




#glass1.dat page-blocks0.dat kddcup-rootkit-imap_vs_back.dat car-good.dat
dataset_name = "car-good.dat"
df, dataset_name = read_file(sys.path[0] + "/input/" + dataset_name)

#1069 1056
# df, dataset_name = read_file_openml(1056)

# print(df)

# print(df["Atr-1"].value_counts())
# print(df["Atr-2"].value_counts())
# print(df["Atr-3"].value_counts())
# print(df["Class"].value_counts())

# for each in df.columns:
#     print(each, df[each].dtype)

# print(df[["Class"]])

oe_style = OneHotEncoder(drop="first", handle_unknown="ignore")
oe_results = oe_style.fit_transform(df)
#print(pd.DataFrame(oe_results.toarray(), columns=oe_style.categories_).head())
print(pd.DataFrame(oe_results.toarray()))


X = df.iloc[: , :-1]
y = df.iloc[: , -1:]

# print(X)
# print(y)

# for each in y.columns:
#     print(each, y[each].dtype == object) #bool


# encoded_columns = []
# for column_name in X.columns:
#     if X[column_name].dtype == object or X[column_name].dtype.name == 'category' or X[column_name].dtype == bool or X[column_name].dtype == str:
#         encoded_columns.extend([column_name])
#         print("coluna alterada:", column_name)

# if encoded_columns:
#     X = pd.get_dummies(X, columns=X[encoded_columns].columns, drop_first=True) #prefix=encoded_columns
    

# encoded_columns = []
# preserve_name = ""
# for column_name in y.columns:
#     if y[column_name].dtype == object or y[column_name].dtype.name == 'category' or y[column_name].dtype == bool or y[column_name].dtype == str:
#         encoded_columns.extend([column_name])
#         preserve_name = column_name
#         print("coluna alterada:", column_name)

# # print(y[encoded_columns].columns)

# if encoded_columns:
#     y = pd.get_dummies(y, columns=y[encoded_columns].columns, drop_first=True)

# if preserve_name:
#     y.rename(columns={y.columns[0]: preserve_name}, inplace = True)

# print("")




# print(X)
# print(y)

# for each in X.columns:
#     print(each, X[each].dtype)

# print("")

# for each in y.columns:
#     print(each, y[each].dtype)


