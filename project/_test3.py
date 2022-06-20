import sys
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import ExtraTreesClassifier
from ml import read_file

dataset_name = "car-good.dat"
df, dataset_name = read_file(sys.path[0] + "/input/" + dataset_name)

print(df)

X = df.iloc[: , :-1]
y = df.iloc[: , -1:]

categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)
cat_cols = df.select_dtypes(include=["category","object","bool"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, cat_cols)
    ],
    remainder='passthrough'
)

df_print = pd.DataFrame(preprocessor.fit_transform(df)) #.toarray()
print(df_print)

# print(preprocessor.fit(df).feature_names_in_)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("sampling", RandomOverSampler(random_state=42)),
        ("classifier", ExtraTreesClassifier(random_state=42, class_weight='balanced', n_jobs=-1))
    ]
)
