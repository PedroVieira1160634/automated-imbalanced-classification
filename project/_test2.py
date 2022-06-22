import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from ml import read_file, read_file_openml


#glass1.dat page-blocks0.dat kddcup-rootkit-imap_vs_back.dat car-good.dat
dataset_name = "glass1.dat"
df, dataset_name = read_file(sys.path[0] + "/input/" + dataset_name)

#1069 1056
# df, dataset_name = read_file_openml(1056)

print(df)

X = df.iloc[: , :-1]
y = df.iloc[: , -1:]


# for feature in df.columns:
#     if df[feature].dtype == object or X[feature].dtype.name == 'category' or X[feature].dtype == bool or X[feature].dtype == str:
#         labelencoder = LabelEncoder()
#         df[feature] = labelencoder.fit_transform(df[feature])
#         enc = OneHotEncoder(drop="first", handle_unknown='ignore', sparse=False)
#         df[feature] = enc.fit_transform(df[feature].values.reshape(-1, 1))
#         # enc_df = pd.DataFrame(enc.fit_transform(df[feature].values.reshape(-1, 1))) #.toarray() .values.reshape(-1, 1)
#         # df = df.join(enc_df, how='left', lsuffix='_left', rsuffix='_right')
        

# print(df)


