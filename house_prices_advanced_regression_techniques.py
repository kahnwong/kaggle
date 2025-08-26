# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# read data
df = pl.read_csv("data/train.csv", null_values=["NA"])

# for k in df.schema.values().mapping:
#     dtype = df.schema.values().mapping[k].__str__()
#
#     # if dtype == "Int64":
#     if dtype == "String":
#         print(k)
#
# print('--')

id_col = "Id"
y_col = "SalePrice"

# x_cols_numeric = [
#     "MSSubClass",
#     "LotFrontage",
#     "LotArea",
#     "OverallQual",
#     "OverallCond",
#     "YearBuilt",
#     "YearRemodAdd",
#     "MasVnrArea",
#     "BsmtFinSF1",
#     "BsmtFinSF2",
#     "BsmtUnfSF",
#     "TotalBsmtSF",
#     "1stFlrSF",
#     "2ndFlrSF",
#     "LowQualFinSF",
#     "GrLivArea",
#     "BsmtFullBath",
#     "BsmtHalfBath",
#     "FullBath",
#     "HalfBath",
#     "BedroomAbvGr",
#     "KitchenAbvGr",
#     "TotRmsAbvGrd",
#     "Fireplaces",
#     "GarageYrBlt",
#     "GarageCars",
#     "GarageArea",
#     "WoodDeckSF",
#     "OpenPorchSF",
#     "EnclosedPorch",
#     "3SsnPorch",
#     "ScreenPorch",
#     "PoolArea",
#     "MiscVal",
#     "MoSold",
#     "YrSold",
# ]
# x_cols_categorial = [
#     "MSZoning",
#     "Street",
#     "Alley",
#     "LotShape",
#     "LandContour",
#     "Utilities",
#     "LotConfig",
#     "LandSlope",
#     "Neighborhood",
#     "Condition1",
#     "Condition2",
#     "BldgType",
#     "HouseStyle",
#     "RoofStyle",
#     "RoofMatl",
#     "Exterior1st",
#     "Exterior2nd",
#     "MasVnrType",
#     "ExterQual",
#     "ExterCond",
#     "Foundation",
#     "BsmtQual",
#     "BsmtCond",
#     "BsmtExposure",
#     "BsmtFinType1",
#     "BsmtFinType2",
#     "Heating",
#     "HeatingQC",
#     "CentralAir",
#     "Electrical",
#     "KitchenQual",
#     "Functional",
#     "FireplaceQu",
#     "GarageType",
#     "GarageFinish",
#     "GarageQual",
#     "GarageCond",
#     "PavedDrive",
#     "PoolQC",
#     "Fence",
#     "MiscFeature",
#     "SaleType",
#     "SaleCondition",
# ]
x_cols_categorial = ["Neighborhood"]
x_cols_numeric = ["BedroomAbvGr", "FullBath", "HalfBath", "GrLivArea", "LotArea"]

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
# le = LabelEncoder()
# le.fit(df[y_col])
# y_encoded = le.transform(df[y_col])
# df_encoded = df_encoded.with_columns(pl.Series(y_encoded).alias(y_col))

# feature engineering
df_encoded = df_encoded.with_columns(
    (
            (pl.col("BedroomAbvGr") * 2)
            + (pl.col("FullBath") * 1)
            + (pl.col("HalfBath") * 0.5)
    ).alias("RoomWeight")
)

# train model
rdf = RandomForestClassifier(
    n_estimators=200, max_depth=50, max_leaf_nodes=5, random_state=2
)
rdf.fit(df_encoded[x_cols + ["RoomWeight"]], df_encoded[y_col])

# prep test data
df_test = pl.read_csv("data/test.csv", null_values=["NA"])
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

df_test_encoded = df_test_encoded.with_columns(
    (
            (pl.col("BedroomAbvGr") * 2)
            + (pl.col("FullBath") * 1)
            + (pl.col("HalfBath") * 0.5)
    ).alias("RoomWeight")
)
pred = rdf.predict(df_test_encoded[x_cols + ["RoomWeight"]])
# pred = le.inverse_transform(pred)

df_prediction = pl.DataFrame()
df_prediction = df_prediction.with_columns(df_test[id_col].alias(id_col))
df_prediction = df_prediction.with_columns(pl.Series(pred).alias(y_col))
df_prediction.to_pandas().to_csv(
    "data/prediction.csv", index=False
)  # polars write true/false, but kaggle expects capitalized form

# evaluate
df_ground_truth = pl.read_csv("data/submission.csv").rename(
    {y_col: f"{y_col}_ground_truth"}
)

# ## classification
# df_prediction = df_prediction.join(
#     df_ground_truth, on=id_col, how="inner"
# ).with_columns((pl.col(y_col) == pl.col(f"{y_col}_ground_truth")).alias("correct"))
#
# correct = len(df_prediction.filter(pl.col("correct")))
# accuracy = correct / len(df_prediction)
# print(f"accuracy: {accuracy}")

## regression
df_prediction = df_prediction.join(df_ground_truth, on=id_col, how="inner")
df_prediction = df_prediction.with_columns(
    ((pl.col(y_col) - pl.col(f"{y_col}_ground_truth")).abs()).alias("AbsError")
)

mae = df_prediction["AbsError"].median()
print(f"accuracy: {mae}")
