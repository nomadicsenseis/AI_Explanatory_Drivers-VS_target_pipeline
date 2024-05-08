from subprocess import check_call
from sys import executable

STEP = "JOIN_HISTORIC_PREDICTIONS"

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
from boto3 import resource
from pandas import read_csv
import yaml
import utils
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

    # Parse date from STR_EXECUTION_DATE
    date = STR_EXECUTION_DATE.split("/")[-1].split("=")[-1].replace("-", "")
    year, month, day = date[:4], date[4:6], date[6:]

    # Define the paths for reading data and the trained model
    read_path = f"{S3_PATH_WRITE}/03_predict_historic_step/{year}{month}{day}"

    # Inicializar una lista vacía para almacenar los DataFrames temporales
    dfs = []

    # Lista de trimestres
    quarters = ["q1", "q2", "q3", "q4"]

    # Cargar y almacenar cada archivo CSV en la lista
    for q in quarters:
        df = pd.read_csv(f"s3://{S3_BUCKET}/{read_path}/historic_predictions_{q}.csv")
        SAGEMAKER_LOGGER.info(f"userlog: Uploaded historic_predictions_{q}")
        dfs.append(df)

    # Concatenar todos los DataFrames en la lista en uno solo
    df_historic = pd.concat(dfs, ignore_index=True)
    
    ## Voy a meter esto aquí a cholon de momento, he calcuado los shaps del target a parte y para subirlos  redshift ha resultado ser un dolor.
    targets_df = pd.read_csv('s3://iberia-data-lake/customer/nps_aggregated_explainability/prod/targets_pretrained_model/corrected_nps_data.csv')
    
    # Added shap targets
    concatenated_df = pd.concat([df_historic,targets_df], ignore_index=True)
        
    SAGEMAKER_LOGGER.info(f"userlog: historic_predict post with targets: {str(concatenated_df.shape)}")
    
    # Save the prediction results to S3
    save_path = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/03_predict_historic_step/{year}{month}{day}/historic_predictions.csv"
    SAGEMAKER_LOGGER.info("userlog: Saving information for joined predictions step in %s.", save_path)
    concatenated_df.to_csv(save_path, index=False)
    
    

