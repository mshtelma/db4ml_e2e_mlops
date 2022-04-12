from typing import Dict

import pandas as pd
import pyspark
import pyspark.pandas as ps
from db4ml import model, transform, pipeline, model_scoring
from mlflow.pyfunc import PyFuncModel
from pyspark.sql import DataFrame, SparkSession
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SKLPipeline
from sklearn.preprocessing import OneHotEncoder

###
# Data Transformations
###


def process_label(
    psdf: pyspark.pandas.DataFrame,
    rename_to: str = "churn",
    label_col: str = "churnString",
) -> pyspark.pandas.DataFrame:
    psdf[label_col] = psdf[label_col].map({"Yes": 1, "No": 0})
    psdf = psdf.astype({label_col: "int32"})
    psdf.rename(columns={label_col: rename_to}, inplace=True)
    return psdf


def pyspark_pandas_ohe(psdf: ps.DataFrame, cat_cols: list) -> pyspark.pandas.DataFrame:
    return ps.get_dummies(psdf, columns=cat_cols, dtype="int64")


def process_col_names(psdf: pyspark.pandas.DataFrame) -> pyspark.pandas.DataFrame:
    cols = psdf.columns.to_list()
    new_col_names = [
        col.replace(" ", "").replace("(", "_").replace(")", "") for col in cols
    ]

    psdf.columns = new_col_names

    return psdf


def drop_missing_values(psdf: pyspark.pandas.DataFrame) -> pyspark.pandas.DataFrame:
    return psdf.dropna()  # noqa


def drop_primary_keys(df: DataFrame) -> DataFrame:
    return df.drop("customerID")


@transform(dataset="telco_churn_dataset")
def load_train(
    source_df: DataFrame,
    drop_missing: bool = False,
) -> DataFrame:
    psdf = drop_primary_keys(source_df).to_pandas_on_spark()
    psdf = process_label(psdf, rename_to="churn")

    # if ohe:
    #     if cat_cols is None:
    #         raise RuntimeError('cat_cols must be provided if ohe=True')
    #     psdf = pyspark_pandas_ohe(psdf, cat_cols)
    #     psdf = process_col_names(psdf)

    if drop_missing:
        psdf = drop_missing_values(psdf)

    preproc_df = psdf.to_spark()
    print("Train DF:")
    preproc_df.show(10, truncate=False)
    print("END of Train DF")
    return preproc_df


@transform(dataset="scoring_df")
def load_scoring(source_df: DataFrame) -> DataFrame:
    scoring_df = load_train(source_df).drop("churn")
    print("Scoring DF:")
    scoring_df.show(10, truncate=False)
    print("END of Scoring DF")
    return scoring_df


###
# Model
###
@model(name="telco_churn")
def create_model(params: Dict[str, str]):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric_transformer",
                SimpleImputer(strategy="median"),
                make_column_selector(dtype_exclude="object"),
            ),
            (
                "categorical_transformer",
                OneHotEncoder(handle_unknown="ignore"),
                make_column_selector(dtype_include="object"),
            ),
        ],
        remainder="passthrough",
        sparse_threshold=0,
    )

    rf_classifier = RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
    )

    _pipeline = SKLPipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", rf_classifier),
        ]
    )

    return _pipeline


###
# Batch Scoring
###
@pipeline(name="telco_churn_scoring_spark")
@model_scoring(model_name="telco_churn", spark_udf=True)
def telco_churn_scoring_spark(
    scoring_df: DataFrame, spark: SparkSession, params: Dict[str, str]
):
    scoring_df.createOrReplaceTempView("telco_churn")
    cols = ", ".join(scoring_df.columns)

    df = spark.sql(f"select *, telco_churn({cols}) as prediction from telco_churn")
    df.show(10)
    df.write.format(params["format"]).mode("overwrite").saveAsTable(
        params["table_name"]
    )


@pipeline(name="telco_churn_scoring_pandas")
@model_scoring(model_name="telco_churn", inject=True)
def telco_churn_scoring_pandas(
    scoring_df: pd.DataFrame,
    telco_churn: PyFuncModel,
    spark: SparkSession,
    params: Dict[str, str],
):
    preds = telco_churn.predict(scoring_df)
    scoring_df["predictions"] = preds
    sdf = spark.createDataFrame(scoring_df)
    sdf.show(10)
    sdf.write.format(params["format"]).mode("overwrite").saveAsTable(
        params["table_name"]
    )
