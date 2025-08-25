# https://www.kaggle.com/competitions/titanic
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# read data
df = pl.read_csv("data/train.csv")

id_col = "PassengerId"
y_col = "Survived"

x_cols_numeric = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
x_cols_categorial = ["Sex", "Cabin", "Embarked"]
x_cols = x_cols_numeric + x_cols_categorial


# encode
encs = {}
df_encoded = pl.DataFrame()
df_encoded = df_encoded.with_columns(df[id_col].alias(id_col))

for i in x_cols_numeric:
    df_encoded = df_encoded.with_columns(df[i].alias(i))

for i in x_cols_categorial:
    # encode
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    values = df[i].to_numpy().reshape(-1, 1)
    enc.fit(values)
    encs[i] = enc

    # transform
    one_hot_encoded_data = enc.fit_transform(values)
    one_hot_df = pd.DataFrame(
        one_hot_encoded_data, columns=enc.get_feature_names_out(["category"])
    )
    encoded_series = one_hot_df.idxmax(axis=1)

    # map these column names to a simple numeric value
    mapping = {name: i for i, name in enumerate(one_hot_df.columns)}
    numeric_series = encoded_series.map(mapping)
    df_encoded = df_encoded.with_columns(pl.Series(numeric_series.to_list()).alias(i))

df_encoded = df_encoded.with_columns(df[y_col].alias(y_col))


# train model
rdf = RandomForestClassifier(
    n_estimators=200, max_depth=20, max_leaf_nodes=5, random_state=2
)
rdf.fit(df_encoded[x_cols], df_encoded[y_col])

# prep test data
df_test = pl.read_csv("data/test.csv")
df_test_encoded = pl.DataFrame()

df_test_encoded = df_test_encoded.with_columns(df_test[id_col].alias(id_col))

for i in x_cols_numeric:
    df_test_encoded = df_test_encoded.with_columns(df_test[i].alias(i))

for i in x_cols_categorial:
    values = df_test[i].to_numpy().reshape(-1, 1)

    # transform
    one_hot_encoded_data = encs[i].fit_transform(values)
    one_hot_df = pd.DataFrame(
        one_hot_encoded_data, columns=encs[i].get_feature_names_out(["category"])
    )
    encoded_series = one_hot_df.idxmax(axis=1)

    # map these column names to a simple numeric value
    mapping = {name: i for i, name in enumerate(one_hot_df.columns)}
    numeric_series = encoded_series.map(mapping)
    df_test_encoded = df_test_encoded.with_columns(
        pl.Series(numeric_series.to_list()).alias(i)
    )

pred = rdf.predict(df_test_encoded[x_cols])

df_prediction = pl.DataFrame()
df_prediction = df_prediction.with_columns(df_test[id_col].alias(id_col))
df_prediction = df_prediction.with_columns(pl.Series(pred).alias("Survived"))
df_prediction.write_csv("data/prediction.csv")

# evaluate
df_ground_truth = pl.read_csv("data/gender_submission.csv").rename(
    {"Survived": "Survived_ground_truth"}
)

df_prediction = df_prediction.join(
    df_ground_truth, on=id_col, how="inner"
).with_columns((pl.col("Survived") == pl.col("Survived_ground_truth")).alias("correct"))

correct = len(df_prediction.filter(pl.col("correct")))
accuracy = correct / len(df_prediction)
print(f"accuracy: {accuracy}")
