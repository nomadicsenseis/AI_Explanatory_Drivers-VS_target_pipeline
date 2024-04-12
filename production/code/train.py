from subprocess import check_call
from sys import executable

STEP = "TRAIN"
FIRST_TIME = True

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


# Models
from catboost import CatBoostClassifier, cv, Pool
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor


# SHAP
import shap

# Random
import random

#Warnings
import warnings
warnings.filterwarnings("ignore")


SAGEMAKER_LOGGER = logging.getLogger("sagemaker")
SAGEMAKER_LOGGER.setLevel(logging.INFO)
SAGEMAKER_LOGGER.addHandler(logging.StreamHandler())


class Arguments(utils.AbstractArguments):
    """Class to define the arguments used in the main functionality."""

    def __init__(self):
        """Class constructor."""
        super().__init__()
        parser = argparse.ArgumentParser(description=f"Inputs for {STEP} step.")
        parser.add_argument("--s3_bucket", type=str)
        parser.add_argument("--s3_path_write", type=str)
        parser.add_argument("--str_execution_date", type=str)
        parser.add_argument("--is_last_date", type=str, default="1")
        parser.add_argument(
            "--model_dir", type=str, default=environ["SM_MODEL_DIR"]
        )

        self.args = parser.parse_args()


def should_update_model(old_metrics, new_metrics, compare_metrics, min_increase, min_thresholds):
    """
    Compara las métricas de dos modelos para decidir si se debe actualizar el modelo antiguo.

    :param old_metrics: Diccionario con las métricas del modelo antiguo.
    :param new_metrics: Diccionario con las métricas del modelo nuevo.
    :param compare_metrics: Lista de métricas que se utilizarán para comparar los modelos.
    :param min_increase: Incremento mínimo requerido en las métricas de comparación.
    :param min_thresholds: Diccionario con umbrales mínimos para las métricas que no están en compare_metrics.
    :return: True si el modelo nuevo es mejor, False en caso contrario.
    """

    train_new_metrics = new_metrics["train"]
    test_old_metrics = old_metrics["test"]
    test_new_metrics = new_metrics["test"]

    # Verificar si el nuevo modelo supera el incremento mínimo en las métricas de comparación
    for metric in compare_metrics:
        improvement = test_new_metrics[metric] - test_old_metrics[metric]
        if improvement < min_increase:
            return False

    # Verificar si el nuevo modelo cumple con los umbrales mínimos para las demás métricas
    for metric, threshold in min_thresholds.items():
        if metric not in compare_metrics and test_new_metrics[metric] < threshold:
            return False

    # Verificar si la diferencia entre las métricas de train y test no es mayor al 5%
    for metric in train_new_metrics:
        if abs(train_new_metrics[metric] - test_new_metrics[metric]) > 0.05:
            return False

    # Verificar si las métricas de train no superan el 97%
    for metric in train_new_metrics:
        if train_new_metrics[metric] > 0.97:
            return False

    return True

def get_metrics(model,X_test,y_test,dataset):
    try:
        proba_predictions = model.predict_proba(X_test)[:, 1]
        predictions = model.predict(X_test)

        metrics = {
            dataset : {
                "AUC": roc_auc_score(y_test, proba_predictions),
                "Recall": recall_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions),
                "F1-score": f1_score(y_test, predictions),
                "Accuracy": accuracy_score(y_test, predictions),
            }
        }
    except:
        metrics = {
            dataset: {
                "AUC": 'Not measure',
                "Recall": 'Not measure',
                "Precision": 'Not measure',
                "F1-score": 'Not measure',
                "Accuracy": 'Not measure',
            }
        }

    SAGEMAKER_LOGGER.info(f"userlog: {dataset}-METRICS: {str(metrics)}")
    return metrics


# Function to cast variables to their appropriate data types
def cast_variables_types(df):
    categorical_features = df.select_dtypes(include=['object'])  # Identify categorical columns
    int_features = df.select_dtypes(include=['int64'])  # Identify integer columns
    float_features = df.select_dtypes(include=['float64'])  # Identify float columns
    boolean_features = df.select_dtypes(include=['bool'])  # Identify boolean columns
    for c in categorical_features:
        df[c] = pd.Categorical(df[c])  # Convert to categorical type
    for c in int_features:
        df[c] = df[c].astype('int16', errors='raise')  # Convert to integer type
    for c in float_features:
        df[c] = df[c].astype('float16', errors='raise')  # Convert to float type
    for c in boolean_features:
        df[c] = df[c].astype('bool', errors='raise')  # Convert to boolean type
    return df  # Return the dataframe with casted variables

# Function to save the model and its metrics to S3
def dumpModel(model,metrics_json, name):
    SAGEMAKER_LOGGER.info(f"Dumping model...")  # Log the start of the model dumping process
    fitted_clf_model = pickle.dumps(model)  # Serialize the model
    s3_resource.Object(
        S3_BUCKET,
        f"{save_path[name]}/model/CatBoostClassifier_cv.pkl",
    ).put(Body=fitted_clf_model)  # Save the serialized model to S3
    SAGEMAKER_LOGGER.info(f"Dumping metrics...")  # Log the start of the metrics dumping process
    s3_resource.Object(
        S3_BUCKET,
        f"{save_path[name]}/metrics/clf_metrics.json",
    ).put(Body=(bytes(json.dumps(metrics_json).encode("UTF-8"))))  # Save the metrics to S3



def eval_set(X_set, y_set, model, features, set_name):  # Define function with input parameters
    X_set = cast_variables_types(X_set)  # Casting variable types of the features in the dataset
    SAGEMAKER_LOGGER.info(f"X {set_name} SHAPE {X_test.shape} ; {y_set.shape}")  # Log the shape of the input data
    SAGEMAKER_LOGGER.info(f"WARNING X {set_name}: rows with na {X_set[features].isnull().any(axis=1).sum()}")  # Log the number of rows with missing values
    missing_rows = X_set[features].isnull().any(axis=1)  # Identify rows with missing values
    X_set = X_set[~missing_rows]  # Remove rows with missing values from the features set
    y_set = y_set[~missing_rows]  # Remove corresponding rows from the target set
    SAGEMAKER_LOGGER.info(f"X {set_name} SHAPE {X_set.shape} ; {y_set.shape}")  # Log the shape of the cleaned data
    metrics_set = get_metrics(model, X_set[features], y_set, set_name)  # Calculate metrics for the model
    del X_set  # Delete the features set to free up memory
    del y_set  # Delete the target set to free up memory
    gc.collect()  # Force garbage collection to free up more memory
    return metrics_set  # Return the calculated metrics



def train_cv_ctb(X_train, y_train, labels, n_splits=5):
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    
    cat_features=['otp15_takeoff']
    # cat_features=['otp15_takeoff'] + [col for col in df_nps_tkt.columns if 'country_agg' in col]
    # cat_features=[]
    
    clf = {}
    cv_scores = {}  # Diccionario para almacenar los scores por cada target

    for target in labels:
        clf[target] = CatBoostClassifier(cat_features=cat_features, random_seed=0, verbose=0)
        cv_scores[target] = []  # Inicializa la lista de scores para el target actual

        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train[target].iloc[train_index], y_train[target].iloc[val_index]

            # Creando el Pool con las características categóricas especificadas
            train_pool = Pool(X_train_fold, y_train_fold, cat_features=cat_features)
            val_pool = Pool(X_val_fold, y_val_fold, cat_features=cat_features)

            # Aquí se corrige a clf[target] para referirse al modelo correcto
            clf[target].fit(train_pool, eval_set=val_pool, early_stopping_rounds=10, verbose=False)

            # Guarda el mejor score para el target actual
            cv_scores[target].append(clf[target].get_best_score())
    
    return clf


if __name__ == "__main__":

    """Main functionality of the script."""
    # DEFINE ARGUMENTS
    SAGEMAKER_LOGGER.info("userlog: Starting %s step...", STEP)
    arguments = Arguments()
    arguments.info(logger=SAGEMAKER_LOGGER)
    args = arguments.get_arguments()
    config = utils.read_config_data()
    S3_BUCKET = args.s3_bucket
    S3_PATH_WRITE = args.s3_path_write
    STR_EXECUTION_DATE = args.str_execution_date
    date = STR_EXECUTION_DATE.split("/")[-1].split("=")[-1].replace("-", "")
    year, month, day = date[:4], date[4:6], date[6:]

    # Variables
    features = list(config['TRAIN']['FEATURES'])
    labels = ['promoter_binary', 'detractor_binary']
    labels = config.get("VARIABLES_ETL").get('LABELS')
    model_names = list(config['TRAIN']['MODEL_NAME'])

    # Paths
    read_path = f"{S3_PATH_WRITE}/01_preprocess_step/train/{year}{month}{day}"
    save_path = {}
    for name in model_names:
        save_path[name] = f"{S3_PATH_WRITE}/02_train_step/{name}/{year}-{month}-{day}"
    SAGEMAKER_LOGGER.info("userlog: Read date path %s.", read_path)


    # Read data
    X_train = read_csv(f"s3://{S3_BUCKET}/{read_path}/data_train/X_train.csv")
    y_train={}
    for target in labels:
        y_train[target] = read_csv(f"s3://{S3_BUCKET}/{read_path}/data_train/y_train_{target}.csv")
    X_train = cast_variables_types(X_train)
    SAGEMAKER_LOGGER.info(f"X_TRAIN SHAPE {X_train.shape} ; {y_train[labels[0]].shape} ;  {y_train[labels[1]].shape}")
    SAGEMAKER_LOGGER.info(f"WARNING X_TRAIN: rows with na {X_train[features].isnull().any(axis=1).sum()}")

    # Estimator
    SAGEMAKER_LOGGER.info(f"userlog: INPUT COLS: {str(features)}")
    models = train_cv_ctb(X_train[features], y_train, labels, n_splits=5)
    metrics_train = {}
    for target in labels:
        metrics_train[target] = get_metrics(models[target], X_train[features], y_train[target], 'train')
        SAGEMAKER_LOGGER.info(f"Trained model: {str(models[target])}")

    del X_train
    del y_train
    gc.collect()

    X_test = read_csv(f"s3://{S3_BUCKET}/{read_path}/data_test/X_test.csv")
    y_test = {}
    metrics_test = {}
    for target in labels:
        y_test[target] = read_csv(f"s3://{S3_BUCKET}/{read_path}/data_test/y_test_{target}.csv")
        metrics_test[target] = eval_set(X_test, y_test[target], models[target], features, 'test')

    X_val = read_csv(f"s3://{S3_BUCKET}/{read_path}/data_val/X_val.csv")
    y_val = {}
    metrics_val = {}
    for target in labels:
        y_val[target] = read_csv(f"s3://{S3_BUCKET}/{read_path}/data_val/y_val_{target}.csv")
        metrics_val[target] = eval_set(X_val, y_val[target], models[target], features, 'test')

    # Initialize Amazon S3 as a resource
    s3_resource = resource("s3")

    # Log information about the start of the process of dumping metrics
    SAGEMAKER_LOGGER.info(f"Dumping metrics...")

    # Combine all metrics from different datasets (train, test, validation, and out of sample)
    clf_metrics = {}
    for name, target in zip(model_names, labels):
        clf_metrics[name] = {**metrics_train[target], **metrics_test[target], **metrics_val[target]}

    # Log the combined metrics
    SAGEMAKER_LOGGER.info(f"METRICS: {clf_metrics}")

    # Create an Amazon S3 resource object using boto3 (Python SDK for AWS)
    s3_resource = boto3.resource('s3')

    # If it's not the first run
    if not FIRST_TIME:
        for target, name in zip(labels, model_names):
            model = models[target]
            # Retrieve the current model from the S3 bucket
            my_bucket = s3_resource.Bucket(S3_BUCKET)
            model_paths = []

            # Filter objects in the bucket under the specified prefix
            for obj in my_bucket.objects.filter(Prefix=f'{S3_PATH_WRITE}/02_train_step/{name}'):
                # Extract the version number (or any numerical value) from the object key
                match = re.search(r'(\d+)/', obj.key)
                if match:
                    # Store the keys with numerical values
                    model_paths.append(obj.key)

            # Assume there is only one numeric folder and get the latest model path
            latest_model_path = sorted(model_paths)[-1]
            SAGEMAKER_LOGGER.info(f"latest_model_path: {latest_model_path}")

            # Get the object (model) from the bucket and deserialize it
            prod_model = (
                my_bucket.Object(f"{latest_model_path}").get()
            )
            prod_model = pickle.loads(prod_model["Body"].read())

            # Define a function to get the number of features used in the model
            def get_number_of_training_variables(model):
                if hasattr(model, 'n_features_in_'):
                    return model.n_features_in_
                if hasattr(model, 'n_features_'):
                    return model.n_features_
                else:
                    return None

            # Get the number of features used in the model
            number_of_training_variables = get_number_of_training_variables(model)

            # If the number of features used is equal to the number of features in the dataset
            if number_of_training_variables == len(features):
                # Get the metrics of the model
                prod_model_metrics_test = get_metrics(prod_model, X_test[features], y_test[target], 'test')

                # Define metrics for comparison and minimum thresholds for model update
                compare_metrics = ["Recall"]
                min_increase = 0.05
                min_thresholds = {"AUC": 0.9, "Recall": 0.8, "Precision": 0.2, "Accuracy": 0.8, "F1-Score": 0.25}

                # Decide whether to update the model based on the defined metrics
                result = should_update_model(prod_model_metrics_test, clf_metrics[name], compare_metrics, min_increase,
                                            min_thresholds)

                # Log the result of whether to update the model
                print("Actualizar modelo:", result)
                clf_metrics_json = json.dumps(clf_metrics[name])

                # If the model should be updated
                if result:
                    # Update the model
                    dumpModel(model, clf_metrics[name], name)
                else:
                    path = f"{S3_PATH_WRITE}/02_train_step/not_deployed"
                    SAGEMAKER_LOGGER.info(f"Dumping FAILED MODEL metrics...")
                    s3_resource.Object(
                        S3_BUCKET,
                        # Define the path for storing the metrics of the model that was not updated
                        f"{path}/{year}{month}{day}_metrics.json",
                        # Write the metrics in JSON format to the specified location in the S3 bucket
                    ).put(Body=(bytes(json.dumps(clf_metrics_json).encode("UTF-8"))))
        else:
            # If the number of features used is not equal to the number of features in the dataset, update the model
            dumpModel(model, clf_metrics[name], name)
    else:
        # If it's the first run, dump (save) the model and its metrics
        for target, name in zip(labels, model_names):
            SAGEMAKER_LOGGER.info(f"Type of model being dumped: {type(models[target])}")
            dumpModel(models[target], clf_metrics[name], name)
