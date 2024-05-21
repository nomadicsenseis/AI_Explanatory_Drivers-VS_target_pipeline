### This step is going to apply a preprocesing to my 2 dataframes (surveys_data_df and lod_factor_df)
### and then is going to merge them into a single df.
### After this is done it 


from subprocess import check_call
from sys import executable

STEP = "PREPROCESS"

check_call([executable, "-m", "pip", "install", "-r", f"./{STEP.lower()}.txt"])

import sklearn.preprocessing as prep
import warnings
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import set_config
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
import yaml 

warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
import argparse
import logging
import pandas as pd
import pickle
import time
from pandas import DataFrame

import boto3
import utils
from io import StringIO

SAGEMAKER_LOGGER = logging.getLogger("sagemaker")
SAGEMAKER_LOGGER.setLevel(logging.INFO)
SAGEMAKER_LOGGER.addHandler(logging.StreamHandler())


# We define the Arguments class that inherits from the AbstractArguments abstract class.
class Arguments(utils.AbstractArguments):
    """Class to define the arguments used in the main functionality."""

    def __init__(self):
        """Class constructor."""
        # We call the constructor of the parent class.
        super().__init__()

        # We create an ArgumentParser object that will contain all the necessary arguments for the script.
        parser = argparse.ArgumentParser(description=f"Inputs for the {STEP} step.")

        # We define the arguments that will be passed to the script.
        # "--s3_bucket": is the name of the S3 bucket where the data will be stored or from where it will be read.
        parser.add_argument("--s3_bucket", type=str)

        # "--s3_path_read": is the path in the S3 bucket from where the data will be read.
        parser.add_argument("--s3_path_read", type=str)

        # "--s3_path_write": is the path in the S3 bucket where the data will be written.
        parser.add_argument("--s3_path_write", type=str)

        # "--str_execution_date": is the execution date of the script.
        parser.add_argument("--str_execution_date", type=str)
        
        # "--str_intervals_starting_date": is the execution date of the script.
        parser.add_argument("--str_intervals_starting_date", type=str)

        # "--use_type": specifies the type of use, it can be "predict" to predict or "train" to train the model.
        parser.add_argument("--use_type", type=str, choices=["predict", "predict_historic"])

        # Finally, we parse the arguments and store them in the self.args property of the class.
        self.args = parser.parse_args()


def calculate_nps(promoters, detractors, total_responses):
    """Calcula el Net Promoter Score (NPS)."""
    return ((promoters - detractors) / total_responses) * 100 if total_responses != 0 else 0

def calculate_weighted_nps(group_df):
    """Calcula el NPS ponderado para un grupo de datos."""
    promoters_weight = group_df.loc[group_df['nps_100'] > 8, 'monthly_weight'].sum()
    detractors_weight = group_df.loc[group_df['nps_100'] <= 6, 'monthly_weight'].sum()
    total_weight = group_df['monthly_weight'].sum()
    
    if total_weight > 0:
        return (promoters_weight - detractors_weight) / total_weight * 100
    else:
        return 0

def calculate_satisfaction(df, variable):
    """Calcula la tasa de satisfacción para una variable dada, utilizando pesos mensuales donde están disponibles,
    y asignando un peso de 1 donde no están disponibles."""
    
    # Asegurarse de que todos los valores NaN en 'monthly_weight' se reemplacen con 1
    df['monthly_weight'].fillna(1, inplace=True)
    
    # Filtrar filas donde la variable no es NaN
    valid_data = df[df[variable].notnull()]

    # Suma de los pesos donde la variable es >= 8 y satisface la condición de estar satisfecho
    satisfied_weight = valid_data[valid_data[variable] >= 8]['monthly_weight'].sum()
    
    # Suma de todos los pesos para las respuestas válidas
    total_weight = valid_data['monthly_weight'].sum()

    # Calcula el porcentaje de satisfacción usando los pesos
    return (satisfied_weight / total_weight) * 100 if total_weight != 0 else 0


def calculate_otp(df, variable='otp15_takeoff'):
    """Calcula el On-Time Performance (OTP) como el porcentaje de valores igual a 1."""
    on_time_count = (df[variable] == 0).sum()
    total_count = df[variable].notnull().sum()
    return (on_time_count / total_count) * 100 if total_count > 0 else 0


def calculate_load_factor(df, pax_column, capacity_column):
    """Calcula el factor de carga para una cabina específica."""
    total_pax = df[pax_column].sum()
    total_capacity = df[capacity_column].sum()
    # Evitar la división por cero
    if total_capacity > 0:
        return (total_pax / total_capacity) * 100
    else:
        return 0

    
def calculate_metrics_summary(df, start_date, end_date, touchpoints):
    # Filtrar por rango de fechas
    df_filtered = df[(df['date_flight_local'] >= pd.to_datetime(start_date)) & (df['date_flight_local'] <= pd.to_datetime(end_date))]
    
    # Mapeo de cabinas a columnas de pax y capacidad
    cabin_mapping = {
        'Economy': ('pax_economy', 'capacity_economy'),
        'Business': ('pax_business', 'capacity_business'),
        'Premium Economy': ('pax_premium_ec', 'capacity_premium_ec')
    }
    
    results_list = []
    
    for (cabin, haul), group_df in df_filtered.groupby(['cabin_in_surveyed_flight', 'haul']):
        result = {
            'start_date': start_date,
            'end_date': end_date,
            'cabin_in_surveyed_flight': cabin,
            'haul': haul,
            'otp15_takeoff': calculate_otp(group_df)
        }
        
        # Calcula el NPS para el grupo
        promoters = (group_df['nps_100'] >= 9).sum()
        detractors = (group_df['nps_100'] <= 6).sum()
        total_responses = group_df['nps_100'].notnull().sum()
        result['NPS'] = calculate_nps(promoters, detractors, total_responses) if total_responses else None
        
        # Calcula el NPS ponderado para el grupo
        result['NPS_weighted'] = calculate_weighted_nps(group_df)
        
        # Satisfacción para cada touchpoint
        for tp in touchpoints:
            result[f'{tp}_satisfaction'] = calculate_satisfaction(group_df, tp)
        
        # Calcula el factor de carga para la cabina
        pax_column, capacity_column = cabin_mapping.get(cabin, (None, None))
        if pax_column and capacity_column:
            result['load_factor'] = calculate_load_factor(group_df, pax_column, capacity_column)
        
        results_list.append(result)
    
    return pd.DataFrame(results_list)

def generate_date_intervals(start_date, end_date):
    """Genera una lista de tuplas con intervalos de fechas desde start_date hasta end_date."""
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    intervals = [(start_date + pd.Timedelta(days=d), end_date) for d in range((end_date - start_date).days + 1)]
    return intervals

def calculate_metrics_for_intervals(df, touchpoints, start_date, end_date):
    """Calcula las métricas para todos los intervalos posibles hasta end_date."""
    intervals = generate_date_intervals(start_date, end_date)
    all_metrics = []

    for interval_start, interval_end in intervals:
        interval_metrics = calculate_metrics_summary(df, interval_start, interval_end, touchpoints)
        all_metrics.append(interval_metrics)

    
    # Concatenar todos los DataFrames de resultados en uno solo
    results_df = pd.concat(all_metrics, ignore_index=True)
    return results_df



def get_names_from_pipeline(preprocessor):
    """
    This function returns the names of the columns that are outputted by the preprocessor.

    Parameters:
    preprocessor (ColumnTransformer): The preprocessor to get output column names from.

    Returns:
    output_columns (list): List of the output column names.
    """
    output_columns = []

    # For each transformer in the preprocessor
    for name, transformer, cols in preprocessor.transformers_:
        # If the transformer is 'drop' or columns are 'drop', continue to the next transformer
        if transformer == 'drop' or cols == 'drop':
            continue

        # If the transformer is a Pipeline, get the last step of the pipeline
        if isinstance(transformer, Pipeline):
            transformer = transformer.steps[-1][1]  # get the last step of the pipeline

        # Depending on the type of the transformer, get the transformed column names
        if isinstance(transformer, ce.TargetEncoder):
            names = [f'{col}_target_enc' for col in cols]
            output_columns += names
        elif isinstance(transformer, ce.WOEEncoder):
            names = [f'{col}_woe_enc' for col in cols]
            output_columns += names
        elif isinstance(transformer, prep.OneHotEncoder):
            names = [f'{col}_enc' for col in transformer.get_feature_names_out(cols)]
            output_columns += names
        else:
            output_columns += cols

    # Return the list of output column names
    return output_columns




def read_data(prefix) -> DataFrame:
    """This function automatically reads a dataframe processed
    with all features in S3 and return this dataframe with
    cid as index

    Parameters
    ----------

    Returns
    -------
    Pandas dataframe containing all features
    """

    s3_keys = [item.key for item in s3_resource.Bucket(S3_BUCKET).objects.filter(Prefix=prefix) if item.key.endswith(".csv")]
    preprocess_paths = [f"s3://{S3_BUCKET}/{key}" for key in s3_keys]
    SAGEMAKER_LOGGER.info(f"preprocess_paths: {preprocess_paths}")
    df_features = pd.DataFrame()
    for file in preprocess_paths:
        df = pd.read_csv(file, error_bad_lines=False)
        df_features = pd.concat([df_features, df], axis=0)
    SAGEMAKER_LOGGER.info(f"Data size: {str(len(df_features))}")
    SAGEMAKER_LOGGER.info(f"Columns: {df_features.columns}")
    df_features.index = df_features[config['VARIABLES_ETL']['ID']]
    df_features.index.name = config['VARIABLES_ETL']['ID']

    return df_features

def read_csv_from_s3(bucket_name, object_key):
    # Create a boto3 S3 client
    s3_client = boto3.client('s3')
    
    # Get the object from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    
    # Read the CSV content
    csv_string = response['Body'].read().decode('utf-8')
    
    # Convert to a Pandas DataFrame
    df = pd.read_csv(StringIO(csv_string))
    
    return df


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
    USE_TYPE = args.use_type
    STR_EXECUTION_DATE = args.str_execution_date
    STR_INTERVALS_STARTING_DATE = args.str_intervals_starting_date
    IS_LAST_DATE = 1
    date = STR_EXECUTION_DATE.split("/")[-1].split("=")[-1].replace("-", "")
    year, month, day = date[:4], date[4:6], date[6:]
    
    # Convert to datetime object
    execution_date = datetime.strptime(STR_EXECUTION_DATE, "%Y-%m-%d")

    # Format dates as strings for S3 prefixes
    today_date_str = execution_date.strftime("%Y-%m-%d")

    # s3 object
    s3_resource = boto3.resource("s3")

    # path
    src_path = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/00_etl_step/{USE_TYPE}/{year}{month}{day}/historic.csv"

    out_path = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/01_preprocess_step/{USE_TYPE}/{year}{month}{day}"

    # Read data
    df_historic = pd.read_csv(src_path)
    df_historic['date_flight_local'] = pd.to_datetime(df_historic['date_flight_local'])
    SAGEMAKER_LOGGER.info(f"userlog: historic_predict pre: {str(df_historic.shape)}")


    # Execute preprocess
    touchpoints = config.get("VARIABLES_PREPROCESS").get('TOUCHPOINTS')

    if USE_TYPE == 'predict_historic':
        # Convert start_date and end_date strings to datetime objects for manipulation
        start_date = datetime.strptime(STR_INTERVALS_STARTING_DATE, '%Y-%m-%d')
        original_end_date = today_date_str

        # Initialize an empty DataFrame to store the results from each interval
        all_intervals_results = pd.DataFrame()
        
        # Generar todas las fechas entre start_date y original_end_date (inclusive)
        date_range = pd.date_range(start=start_date, end=original_end_date)

        # Iterar sobre cada fecha en el rango de fechas
        for current_date in date_range:
            current_date_str = current_date.strftime('%Y-%m-%d')

            # Call your function with the current interval's end_date
            interval_results = calculate_metrics_for_intervals(df_historic, touchpoints, start_date.strftime('%Y-%m-%d'), current_date_str)

            # Append the results for this interval to the all_intervals_results DataFrame
            all_intervals_results = pd.concat([all_intervals_results, interval_results])

        # Reset the index of the final DataFrame if necessary
        all_intervals_results.reset_index(drop=True, inplace=True)
        all_intervals_results['insert_date_ci']= original_end_date
        
        SAGEMAKER_LOGGER.info(f"userlog: historic_predict post: {str(all_intervals_results.shape)}")
        SAGEMAKER_LOGGER.info(f"userlog: historic_predict columns: {str(all_intervals_results.columns)}")
        
        all_intervals_results.to_csv(f"{out_path}/data_for_historic_prediction.csv", index=False)

    else:
        # Convert start_date and end_date strings to datetime objects for manipulation
        start_date = datetime.strptime(STR_INTERVALS_STARTING_DATE, '%Y-%m-%d')
        original_end_date = execution_date

        # Initialize an empty DataFrame to store the results from each interval
        all_intervals_results = pd.DataFrame()

        # Loop over the range from (original_end_date - 45 days) to original_end_date
        for offset in range(0, 46):  # Including the 15th day
            # Calculate the new end_date for this iteration
            end_date = original_end_date - timedelta(days=offset)

            # Convert end_date back to string format if your function expects a string
            end_date_str = end_date.strftime('%Y-%m-%d')

            # Call your function with the current interval's end_date
            interval_results = calculate_metrics_for_intervals(df_historic, touchpoints, start_date.strftime('%Y-%m-%d'), end_date_str)
            
            SAGEMAKER_LOGGER.info(f"userlog: interval_results: {str(interval_results.shape)}")

            # Append the results for this interval to the all_intervals_results DataFrame
            all_intervals_results = pd.concat([all_intervals_results, interval_results])

        # Reset the index of the final DataFrame if necessary
        all_intervals_results.reset_index(drop=True, inplace=True)
        all_intervals_results['insert_date_ci']= original_end_date
        
        SAGEMAKER_LOGGER.info(f"userlog: historic_predict post: {str(all_intervals_results.shape)}")
        all_intervals_results.to_csv(f"{out_path}/data_for_prediction.csv", index=False)
        
        
