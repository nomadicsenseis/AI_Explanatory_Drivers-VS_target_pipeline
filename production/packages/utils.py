"""
Utils for NPS model
"""
from argparse import Namespace
from logging import Logger
from os.path import dirname, join, realpath
from re import split as regex_split
from typing import Dict, Optional

from boto3 import client as b3_client
from yaml import safe_load
import pandas as pd
import numpy as np
import shap

# Sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    log_loss,
    classification_report,
    confusion_matrix,
    make_scorer,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer



import darts
from darts import TimeSeries
from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)

from darts.metrics import mape, smape, mae
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries

from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor

import lightgbm

from darts.models import LightGBMModel

from darts.models import LightGBMModel, RandomForest, LinearRegressionModel
from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis

from darts.explainability.shap_explainer import ShapExplainer
import pickle
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from darts.models import LinearRegressionModel, LightGBMModel, RandomForest
from calendar import month_name as mn
import os

# Random
import random

#Warnings
import warnings
warnings.filterwarnings("ignore")



def read_config_data(path: Optional[str] = None) -> Dict:
    """Read the config.yml file asociated.

    The config.yml file asociated is the one in the same path.

    Parameters
    ----------
        path: Path where is saved the config.yml file.
    Returns
    -------
        Dictionary with the configuration of the process.
    """
    if path is None:
        base_path = dirname(realpath(__file__))
        config_file_path = f"{base_path}/config.yml"
    else:
        config_file_path = path
    with open(config_file_path) as conf_file:
        configuration = conf_file.read()
    return safe_load(configuration)


def get_last_s3_partition(
    s3_dir: str,
    execution_date: int,
    preffix: Optional[str] = None,
    n_partition: int = 1,
) -> str:
    """This function get the las partitition of a given path from an specified execution_date.

    :param s3_dir: S3 path data ending with '/'
    :param execution_date: Execution date to limit the search perimeter.
    :param preffix: Preffix of the s3 key for the date partition. (Could be 'insert_date_ci=').
    :param n_partition: 1 means select the last available partition, 2 the following one, and so on.
    :return: Complete path of the last partition to read.
    """
    preffix = " " if preffix is None else preffix
    print(preffix)
    
    s3_client = b3_client("s3")
    s3_bucket = s3_dir.split("/", 1)[0]
    s3_prefix = s3_dir.split("/", 1)[-1]
    print(f's3_bucket: {s3_bucket}')
    print(f's3_prefix: {s3_prefix}')
    s3_contents = s3_client.list_objects_v2(
        Bucket=s3_bucket, Prefix=s3_prefix, Delimiter="/"
    ).get("CommonPrefixes")
    print(f's3_contents: {s3_contents}')
    partition_date_aux = [
        int(
            content["Prefix"]
            .strip("/")
            .split("/")[-1]
            .replace("-", "")
            .split(preffix)[-1]
        )
        for content in s3_contents
    ]
    partition_date = [
        content["Prefix"].strip("/").split("/")[-1].split(preffix)[-1]
        for content in s3_contents
    ]
    filtered_dates = list(
        filter(
            lambda e: e[0] <= execution_date, zip(partition_date_aux, partition_date)
        )
    )
    sorted_dates = sorted(filtered_dates, key=lambda e: e[0])
    try:
        return_path = join(s3_dir, f"{preffix}{str(sorted_dates[-n_partition][-1])}".strip())
    except IndexError:
        return_path = join(s3_dir, f"{preffix}_notfoundpreviousdate".strip())
    return return_path


def get_path_to_read_and_date(
    read_last_date: bool,
    bucket: str,
    key: str,
    partition_date: str,
    n_partition: int = 1,
):
    """Get path to read (given or last) and the chosen date.

    :param read_last_date: Boolean to read last valid date (True) or given date (False).
    :param bucket: S3 bucket.
    :param key: S3 key.
    :param partition_date: String with the execution date (could be separated by '=' sign).
    :param n_partition: 1 means select the last available partition, 2 the following one, and so on.
    :return: Tuple[
            Path with data,
            year of the read data,
            month of the read data,
            day of the read data
        ]
    """
    if read_last_date:
        exec_date = int(partition_date.split("=")[-1].replace("-", ""))
        date_preffix = (
            regex_split(r"[0-9]{4}-?[0-9]{2}-?[0-9]{2}", partition_date)[0]
            if "=" in partition_date
            else None
        )

        path = get_last_s3_partition(
            s3_dir=f"{bucket}/{key}/",
            execution_date=exec_date,
            preffix=date_preffix,
            n_partition=n_partition,
        )
        date = path.split("/")[-1].split("=")[-1].replace("-", "")
        # date = partition_date.split("/")[-1].split("=")[-1].replace("-", "")
        year, month, day = date[:4], date[4:6], date[6:]
        path = f"s3://{path}"
    else:
        path = f"s3://{bucket}/{key}/{partition_date}"
        date = partition_date.split("=")[-1]
        if "-" in partition_date:
            date = date.split("-")
            year, month, day = date[0], date[1], date[2]
        else:
            year, month, day = date[:4], date[4:6], date[6:]
    return path, year, month, day


def process_dataframe(df):
    # Agrupar y procesar los datos
    grouped_dfs = {}
    for group_name, group_data in df.groupby(['cabin_in_surveyed_flight', 'haul']):
        cabin_value, haul_value = group_name
        group_df = group_data.copy()
        group_df_name = f'{cabin_value}_{haul_value}_df'
        grouped_dfs[group_df_name] = group_df

    # Reconstruir el DataFrame original
    df = pd.concat(grouped_dfs.values())
    df.reset_index(drop=True, inplace=True)

    return df, grouped_dfs

def compute_shap_and_prediction(row, key, features_cols, future_scaler, model):
    """
    Computes SHAP values and the predicted NPS for a given row.
    
    Parameters:
    - row_df: The DataFrame row for which to compute SHAP values and prediction.
    - key: The key identifying the specific model and scaler to use.
    - features_cols: List of column names representing features used by the model.
    
    Returns:
    - A tuple containing SHAP values as a dictionary and the predicted NPS.
    """
    # Logic to prepare the row for SHAP value computation and prediction
    aux_nps_ts = TimeSeries.from_series(pd.Series([0]))
    aux_row = pd.DataFrame(0, index=[0], columns=row.columns)
    row_df = pd.concat([aux_row, row]).reset_index(drop=True)
    
    future_covariates_ts = TimeSeries.from_dataframe(row_df[features_cols])[-1:]
    future_covariates_ts_scaled = future_scaler.transform(future_covariates_ts)
    
    # Compute SHAP values and prediction
    shap_explain = ShapExplainer(model=model)
    shap_explained = shap_explain.explain(aux_nps_ts, foreground_future_covariates=future_covariates_ts_scaled)
    shap_explanation = shap_explained.get_shap_explanation_object(horizon=1)

    shap_values = shap_explanation[0].values
    base_value = shap_explanation[0].base_values
    pred_value = base_value + shap_values.sum()
    feature_names=[]
    for feat in shap_explanation.feature_names:
        name = [f for f in features_cols if f in feat]
        feature_names.append(name[0])
    
    
    # Convert SHAP values to a dictionary and adjust the logic based on your ShapExplainer
    shap_values_dict = {f"{feature}_nps": value for feature, value in zip(feature_names, shap_values)}
    shap_values_dict["out_prob_base"] = base_value,
    shap_values_dict["out_prob_nps"] = pred_value,
    
    return shap_values_dict

class AbstractArguments:
    """Abstract class with functionalities for input arguments."""

    def __init__(self) -> None:
        """Abstract constructor."""
        self.args = NotImplemented

    def get_arguments(self) -> Namespace:
        """Get dictionary with arguments."""
        return self.args

    def info(self, logger: Logger) -> None:
        """Add to the logger the chosen arguments."""
        for arg_name, arg_value in vars(self.args).items():
            logger.info(f"--> Argument --{arg_name}={arg_value}")



