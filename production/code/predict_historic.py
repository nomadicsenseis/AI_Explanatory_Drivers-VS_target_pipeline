from subprocess import check_call
from sys import executable

STEP = "PREDICT_HISTORIC"

check_call([executable, "-m", "pip", "install", "-r", f"./{STEP.lower()}.txt"])

# General
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from datetime import datetime, timedelta
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import os
import numpy as np
import datetime
import boto3
import s3fs
from itertools import combinations
import pickle
import json
import re
import gc
import argparse
import logging
from os import environ
import utils
from boto3 import resource
from pandas import read_csv
import yaml 

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

SAGEMAKER_LOGGER = logging.getLogger("sagemaker")
SAGEMAKER_LOGGER.setLevel(logging.INFO)
SAGEMAKER_LOGGER.addHandler(logging.StreamHandler())


# Inherits from the AbstractArguments class
class Arguments(utils.AbstractArguments):
    """Class to define the arguments used in the main functionality."""

    def __init__(self):
        """Class constructor."""
        # Call to the constructor of the parent class
        super().__init__()

        # Create an ArgumentParser object
        parser = argparse.ArgumentParser(description=f"Inputs for {STEP} step.")

        # Add the command line arguments that will be used
        parser.add_argument("--s3_bucket", type=str)  # S3 bucket name
        parser.add_argument("--s3_path_write", type=str)  # S3 path to write data
        parser.add_argument("--str_execution_date", type=str)  # Execution date
        parser.add_argument("--is_last_date", type=str, default="1")  # Indicator for the last date
        parser.add_argument("--quarter", type=str)

        # Parse the arguments and store them in the 'args' attribute
        self.args = parser.parse_args()


if __name__ == "__main__":
    """Main functionality of the script."""

    # Log the start of the step
    SAGEMAKER_LOGGER.info("userlog: Starting %s step...", STEP)

    # Initialize the Arguments class and get the arguments
    arguments = Arguments()
    arguments.info(logger=SAGEMAKER_LOGGER)
    args = arguments.get_arguments()

    # Extract the argument values
    S3_BUCKET = args.s3_bucket
    S3_PATH_WRITE = args.s3_path_write
    STR_EXECUTION_DATE = args.str_execution_date
    IS_LAST_DATE = args.is_last_date
    quarter = args.quarter

    # Parse date from STR_EXECUTION_DATE
    date = STR_EXECUTION_DATE.split("/")[-1].split("=")[-1].replace("-", "")
    year, month, day = date[:4], date[4:6], date[6:]

    # Load the configuration data
    config = utils.read_config_data()
    
    keys = list(config['PREDICT']['CABIN_HAULS'])
    model_names = list(config['PREDICT']['MODEL_NAME'])
    scaler_names = list(config['PREDICT']['SCALER_NAME'])
    features = list(config['PREDICT']['FEATURES'])
    cols_to_save = list(config['PREDICT']['COLUMNS_SAVE'])

    # Initialize boto3 S3 client
    s3 = boto3.client('s3')

    # Define the paths for reading data and the trained model
    read_path = f"{S3_PATH_WRITE}/01_preprocess_step/predict_historic/{year}{month}{day}"

    # Load the data to predict
    df_predict = pd.read_csv(f"s3://{S3_BUCKET}/{read_path}/data_for_historic_prediction.csv")
    def split_df_by_quarters(df):
        num_rows = len(df)
        quarter_size = num_rows // 4
        quarters = {}
        for i in range(4):
            start_index = i * quarter_size
            if i == 3:  # Asegurar que el último "quarter" incluya el resto de las filas
                end_index = num_rows
            else:
                end_index = start_index + quarter_size
            quarters[f"q{i+1}"] = df.iloc[start_index:end_index]
        return quarters

    def get_data_by_quarter(df, quarter):
        quarters = split_df_by_quarters(df)
        return quarters[quarter]

    # Uso del código
    df_predict = get_data_by_quarter(df_predict, quarter)  # Cambia 'q1' por 'q2', 'q3', o 'q4' según sea necesario

    day_predict_df, day_predict_df_grouped_dfs = utils.process_dataframe(df_predict)

    # Initialize a dictionary to store the augmented DataFrames, models, and scalers
    augmented_dfs = {}
    lgbm_model = {}
    future_scalers = {}
    future_scaler_key = {}
    model_key = {}
    scaler_response = {}
    model_response = {}

    for key in day_predict_df_grouped_dfs.keys():
        # Initialize a list to collect augmented rows
        augmented_rows = []

        # Load the pre-trained model and scaler from S3
        path = f"{S3_PATH_WRITE}/targets_pretrained_model"
        future_scaler_key[key] = f"{path}/future_scaler_{key}.pkl"
        model_key[key] = f"{path}/best_tuned_mae_model_{key}_LightGBMModel.pkl"

        # Load scaler
        scaler_response[key] = s3.get_object(Bucket=S3_BUCKET, Key=future_scaler_key[key])
        future_scalers[key] = pickle.loads(scaler_response[key]['Body'].read())

        # Load model
        model_response[key] = s3.get_object(Bucket=S3_BUCKET, Key=model_key[key])
        lgbm_model[key] = pickle.loads(model_response[key]['Body'].read())

        for index in range(len(day_predict_df_grouped_dfs[key])):
            # Access the row by its index using .iloc
            row_df = day_predict_df_grouped_dfs[key].iloc[[index]]

            # Compute SHAP values and predicted NPS here...
            # Assuming `compute_shap_and_prediction` is a function you'd implement
            # This function should return SHAP values as a dict and the predicted NPS
            shap_values = utils.compute_shap_and_prediction(row_df, key, features, future_scalers[key], lgbm_model[key])

            # For each feature, add its SHAP value to the row
            for feature_name, shap_value in shap_values.items():
                row_df[f'{feature_name}'] = shap_value

            # Add base value and predicted NPS columns
            # row_df['Base Value'] = shap_values['base_value']  # Adjust based on how you obtain the base value
            # row_df['Predicted NPS'] = predicted_nps
            # Append the augmented row to the list
            augmented_rows.append(row_df)


        # Concatenate all augmented rows to form the complete augmented DataFrame
        augmented_dfs[key] = pd.concat(augmented_rows).reset_index(drop=True)
        
        # Reconstruir el DataFrame original
    df = pd.concat(augmented_dfs.values())
    df.reset_index(drop=True, inplace=True)

    # Rename columns, add insert date and select columns to save
    df = df[cols_to_save]
    SAGEMAKER_LOGGER.info(f"userlog: {df.info()}")
    

    # Save the prediction results to S3
    save_path = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/03_predict_historic_step/{year}{month}{day}/historic_predictions_{quarter}.csv"
    SAGEMAKER_LOGGER.info("userlog: Saving information for predict step in %s.", save_path)
    df.to_csv(save_path, index=False)

    # Obtener el año para cada fecha (necesario para construir rangos de fechas)
    # df_predict = df_predict[df_predict['date_flight_local'].dt.year >= 2023]
    
    # df_predict = df_predict[df_predict['date_flight_local'].dt.day == 1]

    
    

