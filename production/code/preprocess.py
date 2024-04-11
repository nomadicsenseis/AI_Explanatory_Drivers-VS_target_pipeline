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

        # "--use_type": specifies the type of use, it can be "predict" to predict or "train" to train the model.
        parser.add_argument("--use_type", type=str, choices=["predict", "train"])

        # Finally, we parse the arguments and store them in the self.args property of the class.
        self.args = parser.parse_args()


def feature_processer(df: DataFrame, use_type='predict', y_train=None) -> DataFrame:
    """
    This function processes the features of the input DataFrame depending on the mode (prediction or training).
    If in prediction mode, it applies a pre-saved pipeline transformation.
    If in training mode, it fits and applies the transformation pipeline, and saves the pipeline for future use.

    Parameters:
    df (DataFrame): The input DataFrame to be processed.
    use_type (str): The mode, either 'predict' or 'train'. Default is 'predict'.
    y_train (Series): The target variable Series if in 'train' mode.

    Returns:
    DataFrame: The processed DataFrame.
    """
    save_path = f'{S3_PATH_WRITE}/01_preprocess_step/{USE_TYPE}/{year}{month}{day}'
    save_path_read = f'{S3_PATH_WRITE}/01_preprocess_step/{USE_TYPE}'

    # Convert 'date_flight_local' to datetime and extract 'month' and 'year' if not already done
    df['date_flight_local'] = pd.to_datetime(df['date_flight_local'])
    df['month'] = df['date_flight_local'].dt.month
    df['year'] = df['date_flight_local'].dt.year

    # In prediction mode
    if use_type == 'predict':
        # Get the path to the saved transformation pipeline
        model_path, _, _, _ = utils.get_path_to_read_and_date(
            read_last_date=bool(int(IS_LAST_DATE)),
            bucket=S3_BUCKET,
            key=f'{S3_PATH_WRITE}/01_preprocess_step/train',
            partition_date=STR_EXECUTION_DATE,
        )
        # # Remove s3 info path
        if 's3://' in model_path:
            model_path = model_path.split('//')[1].replace(f"{S3_BUCKET}/", '')
        SAGEMAKER_LOGGER.info(f"userlog: path for models: {model_path + '/models/'}")

        # Loop over each month to reead the transformation pipeline
        for m in sorted(df['month'].unique()):
            # Filter the data for the specific month
            df_month = df[df['month'] == m]
            # Split data into features and target
            X = df_month.drop(columns=['ticket_price', 'date_flight_local', 'month', 'year'])
            y = df_month['ticket_price']
        
            # Read the transformation pipeline
            pipeline = (
                s3_resource.Object(S3_BUCKET, f"{model_path}/models/{config['PREPROCESS']['PIPELINE_NAME']}_{m}.pkl")
                .get()
            )

            # Load the pipeline and apply transformation
            pipe = pickle.loads(pipeline["Body"].read())
            time.sleep(10)
            
            # Identify indices where ticket_price is NaN for imputation
            idx_missing = y[y.isna()].index
                
            # Predict missing ticket_price values
            X_missing = X.loc[idx_missing]
            if len(X_missing) > 0:
                imputed_values = pipe.predict(X_missing)
                # Fill in the missing values in the original DataFrame
                df.loc[idx_missing, 'ticket_price'] = imputed_values

    # In training mode
    else:
        # Complex 2019 ticket_price inputer
        # Identify categorical and numerical columns; make sure 'month' and 'year' are not in these lists
        categorical_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.drop(['ticket_price', 'month', 'year']).tolist()

        # Define the preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ]
        )

        # Define the model pipeline
        pipeline = make_pipeline(
            preprocessor,
            GradientBoostingRegressor(random_state=42)
        )

        # Loop over each month to train a model and impute missing values
        for m in sorted(df['month'].unique()):
            # Filter the data for the specific month
            df_month = df[df['month'] == m]
            
            # Split data into features and target
            X = df_month.drop(columns=['ticket_price', 'date_flight_local', 'month', 'year'])
            y = df_month['ticket_price']
            
            # Further split your data into training and missing data (for imputation)
            X_train = X.loc[y.notna()]
            y_train = y.loc[y.notna()]
            
            if len(X_train) > 0:
                # Train the model
                pipeline.fit(X_train, y_train)

                # Save model/pipeline for predict step.
                s3_resource.Object(S3_BUCKET, f"{save_path}/models/{config['PREPROCESS']['PIPELINE_NAME']}_{m}.pkl").put(
                    Body=pickle.dumps(pipeline)
                )

                
                # Identify indices where ticket_price is NaN for imputation
                idx_missing = y[y.isna()].index
                
                # Predict missing ticket_price values
                X_missing = X.loc[idx_missing]
                if len(X_missing) > 0:
                    imputed_values = pipeline.predict(X_missing)
                    
                    # Fill in the missing values in the original DataFrame
                    df.loc[idx_missing, 'ticket_price'] = imputed_values
                    

        SAGEMAKER_LOGGER.info(f'Processing X_train {X_train.shape} ')
        SAGEMAKER_LOGGER.info(f'Processing y_train {y_train.shape} ')
        SAGEMAKER_LOGGER.info(f'Procesed X_train {X.shape} ')

    df.reset_index(drop=True, inplace=True)

    return df


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


def split_train_val_test(X: DataFrame, target) -> object:
    # Extraer el año antes de las divisiones
    years = X["date_flight_local"].dt.year
    
    # División inicial entre conjuntos de entrenamiento+prueba y validación
    X_traintest, X_val, y_traintest, y_val, years_traintest, years_val = train_test_split(
        X, target, years, stratify=years, test_size=0.2)
    
    # Segunda división para separar entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_traintest, y_traintest, stratify=years_traintest, test_size=0.2)
    
    return X_train, X_test, X_val, y_train, y_test, y_val

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
    src_path_historic = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/00_etl_step/{USE_TYPE}/{year}{month}{day}/historic.csv"
    src_path_incremental = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/00_etl_step/{USE_TYPE}/{year}{month}{day}/incremental.csv"

    out_path = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/01_preprocess_step/{USE_TYPE}/{year}{month}{day}"



    # Execute preprocess
    in_features_train = config.get("VARIABLES_ETL").get('COLUMNS_TO_SAVE')
    # in_features_train = config.get("TRAIN").get('FEATURES') 
    # + [config.get("VARIABLES_ETL").get('ID')]

    if USE_TYPE == 'train':
        # Read data
        # df_features = read_csv_from_s3(S3_BUCKET, src_path_historic)
        df_features = pd.read_csv(src_path_historic)
        df_features['date_flight_local'] = pd.to_datetime(df_features['date_flight_local'])
        labels = config.get("VARIABLES_ETL").get('LABELS')

        # Divide train and test
        X_train, X_test, X_val, y_train, y_test, y_val = split_train_val_test(df_features[in_features_train],
                                                                              df_features[labels])
        # Features encoder
        SAGEMAKER_LOGGER.info(f"X_train, X_test, X_val, y_train, y_test, y_val: {X_train.shape}, {X_test.shape}, "
                              f"{X_val.shape}, {y_train.shape}, {y_test.shape}, {y_val.shape}")

        X_train = feature_processer(X_train, use_type='train', y_train=y_train)

        X_test = feature_processer(X_test)
        SAGEMAKER_LOGGER.info(f'Processing X_test, shape {X_test.shape}')
        X_val = feature_processer(X_val)
        SAGEMAKER_LOGGER.info(f'Processing X_val, shape {X_val.shape}')

        X_train.to_csv(f"{out_path}/data_train/X_train.csv", index=False)
        X_test.to_csv(f"{out_path}/data_test/X_test.csv", index=False)
        X_val.to_csv(f"{out_path}/data_val/X_val.csv", index=False)

        for target in labels:
            y_train[target].to_csv(f"{out_path}/data_train/y_train_{target}.csv", index=False)
            y_test[target].to_csv(f"{out_path}/data_test/y_test_{target}.csv", index=False)
            y_val[target].to_csv(f"{out_path}/data_val/y_val_{target}.csv", index=False)

        # Pass on full historic data for prediction.
        X_pred = df_features[in_features_train]
        SAGEMAKER_LOGGER.info(f"userlog: feature_processer_historic_predict pre: {str(X_pred.shape)}")
        X_pred = feature_processer(X_pred)
        SAGEMAKER_LOGGER.info(f"userlog: feature_processer_historic_predict post: {str(X_pred.shape)}")
        X_pred.to_csv(f"{out_path}/data_for_historic_prediction.csv", index=False)

    else:
        # Read data
        df_features = pd.read_csv(src_path_incremental)
        # Pass on incremental data for prediction.
        X_pred = df_features[in_features_train]
        SAGEMAKER_LOGGER.info(f"userlog: feature_processer_incremental_predict pre: {str(X_pred.shape)}")
        X_pred = feature_processer(X_pred)
        SAGEMAKER_LOGGER.info(f"userlog: feature_processer_incremental_predict post: {str(X_pred.shape)}")
        X_pred.to_csv(f"{out_path}/data_for_prediction.csv", index=False)
