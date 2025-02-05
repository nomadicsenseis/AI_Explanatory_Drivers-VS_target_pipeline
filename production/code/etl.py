from subprocess import check_call
from sys import executable


STEP = "ETL"
check_call([executable, "-m", "pip", "install", "-r", f"./{STEP.lower()}.txt"])

import argparse
import utils
import logging
import json
import pandas as pd
import boto3
import s3fs
import yaml 
from datetime import datetime, timedelta


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
        parser.add_argument("--s3_bucket_nps", type=str)

        # "--s3_bucket": is the name of the S3 bucket where the data will be stored or from where it will be read.
        parser.add_argument("--s3_bucket_lf", type=str)

        # "--s3_path_read": is the path in the S3 bucket from where the data will be read.
        parser.add_argument("--s3_path_read_nps", type=str)

        # "--s3_path_read": is the path in the S3 bucket from where the data will be read.
        parser.add_argument("--s3_path_read_lf", type=str)

        # "--s3_path_write": is the path in the S3 bucket where the data will be written.
        parser.add_argument("--s3_path_write", type=str)

        # "--str_execution_date": is the execution date of the script.
        parser.add_argument("--str_execution_date", type=str)

        # "--use_type": specifies the type of use, it can be "predict" to predict or "predict_historic" to make a onse shot prediction over the whole dataset.
        parser.add_argument("--use_type", type=str, choices=["predict", "predict_historic"])

        # Finally, we parse the arguments and store them in the self.args property of the class.
        self.args = parser.parse_args()

if __name__ == "__main__":

    """Main functionality of the script."""
    # Arguments
    SAGEMAKER_LOGGER.info("userlog: Starting %s step...", STEP)
    arguments = Arguments()
    arguments.info(logger=SAGEMAKER_LOGGER)
    args = arguments.get_arguments()
    config = utils.read_config_data()
    USE_TYPE = args.use_type
    S3_BUCKET_NPS = args.s3_bucket_nps
    S3_BUCKET_LF = args.s3_bucket_lf
    S3_PATH_READ_NPS = args.s3_path_read_nps
    S3_PATH_READ_LF = args.s3_path_read_lf
    S3_PATH_WRITE = args.s3_path_write
    STR_EXECUTION_DATE = args.str_execution_date
    date = STR_EXECUTION_DATE.split("/")[-1].split("=")[-1].replace("-", "")
    year, month, day = date[:4], date[4:6], date[6:]
    
    # Convert to datetime object
    execution_date = datetime.strptime(STR_EXECUTION_DATE, "%Y-%m-%d")
    today_date_str = execution_date.strftime("%Y-%m-%d")
    

    # Columns to save
    COLUMNS_TO_SAVE = config.get("VARIABLES_ETL").get('COLUMNS_TO_SAVE')

    # READ NPS DATA SOURCE
    # Read df_nps_surveys
    s3_resource = boto3.resource("s3")

    # READ TODAY DATA (HISTORIC NPS)
    today_nps_surveys_prefix = f'{S3_PATH_READ_NPS}/insert_date_ci={today_date_str}/'
    s3_keys = [item.key for item in s3_resource.Bucket(S3_BUCKET_NPS).objects.filter(Prefix=today_nps_surveys_prefix)]
    preprocess_paths = [f"s3://{S3_BUCKET_NPS}/{key}" for key in s3_keys]

    SAGEMAKER_LOGGER.info("userlog: Read historic nps_surveys data path %s.", today_nps_surveys_prefix)
    df_nps_historic = pd.DataFrame()
    for file in preprocess_paths:
        df = pd.read_csv(file)
        df_nps_historic = pd.concat([df_nps_historic, df], axis=0)
    df_nps_historic = df_nps_historic.reset_index(drop=True)

    
    # READ LF DATA SOURCE
    load_factor_prefix = f's3://{S3_BUCKET_LF}/{S3_PATH_READ_LF}/'

    # Assume rol for prod
    sts_client = boto3.client('sts')
    assumed_role = sts_client.assume_role(
        RoleArn="arn:aws:iam::320714865578:role/ibdata-prod-role-assume-customer-services-from-ibdata-aip-prod",
        RoleSessionName="test"
    )
    credentials = assumed_role['Credentials']
    fs = s3fs.S3FileSystem(key=credentials['AccessKeyId'], secret=credentials['SecretAccessKey'], token=credentials['SessionToken'])

    # Listall the files
    load_factor_list = fs.ls(load_factor_prefix)
    
    SAGEMAKER_LOGGER.info("userlog: Read historic load_factor data path %s.", load_factor_prefix)
    dataframes = []
    for file_path in load_factor_list:
        try:
            file_info = fs.info(file_path)
            if file_info['Size'] == 0:
                SAGEMAKER_LOGGER.info(f"Skipping empty file: {file_path}")
                continue

            with fs.open(f's3://{file_path}') as f:
                df = pd.read_csv(f)
                dataframes.append(df)
        except pd.errors.EmptyDataError:
            SAGEMAKER_LOGGER.info(f"Caught EmptyDataError for file: {file_path}, skipping...")
        except Exception as e:
            SAGEMAKER_LOGGER.error(f"Error reading file {file_path}: {e}")

    if dataframes:
        df_lf_historic = pd.concat(dataframes, ignore_index=True)
    else:
        df_lf_historic = pd.DataFrame()

    # 1. Filter dataframes by carrier code.
    SAGEMAKER_LOGGER.info("userlog: ETL 1.0 Filter dataframes by carrier code.")
    df_nps_historic['haul'] = df_nps_historic['haul'].replace('MH', 'SH')
    # NPS HISTORIC
    condition_1 = (df_nps_historic['operating_airline_code'].isin(['IB', 'YW']))
    condition_2 = ((df_nps_historic['invitegroup_ib'] != 3) | (df_nps_historic['invitegroup_ib'].isnull()))
    condition_3 = (df_nps_historic['invitegroup'] == 2)
    
    df_nps_historic = df_nps_historic.loc[condition_1 & (condition_2 & condition_3)]

    # LOAD FACTOR HISTORIC
    # df_lf_historic = df_lf_historic.loc[(df_lf_historic['operating_carrier'].isin(['IB', 'YW']))]
    # "Operating carrier no longer exists, it has been replaced by op_carrier or op_carrier_ib_group. Still i dont think it is needed in the merge"

    # 2. Transform date column to datetime format
    SAGEMAKER_LOGGER.info("userlog: ETL 2.0 Transform date column to datetime format.")
    delay_features = ['real_departure_time_local', 'scheduled_departure_time_local']
    for feat in delay_features:
        df_nps_historic[feat] = pd.to_datetime(df_nps_historic[feat], format="%Y%m%d %H:%M:%S", errors = 'coerce')
            
    df_nps_historic['delay_departure'] = (df_nps_historic['real_departure_time_local'] - df_nps_historic['scheduled_departure_time_local']).dt.total_seconds()/60
    
    # NPS
    df_nps_historic['date_flight_local'] = pd.to_datetime(df_nps_historic['date_flight_local'])

    # Load Factor
    df_lf_historic['flight_date_local'] = pd.to_datetime(df_lf_historic['flight_date_local'])

    # 3. Filter out covid years
    SAGEMAKER_LOGGER.info("userlog: ETL 3.0 Filter out covid years.")
    # NPS (historic)
    df_nps_historic = df_nps_historic[df_nps_historic['date_flight_local'].dt.year >= 2019]
    df_nps_historic = df_nps_historic[~df_nps_historic['date_flight_local'].dt.year.isin([2020, 2021])]
    # Load factor (historic)
    df_lf_historic = df_lf_historic[df_lf_historic['flight_date_local'].dt.year >= 2019]
    df_lf_historic = df_lf_historic[~df_lf_historic['flight_date_local'].dt.year.isin([2020, 2021])]

    # 4. Create otp, promoter, detractor and load factor columns.
    SAGEMAKER_LOGGER.info("userlog: ETL 4.0 Create otp, promoter, detractor and load factor columns.")
    # OTP
    df_nps_historic['otp15_takeoff'] = (df_nps_historic['delay'] > 15).astype(int)

    # Promoter and Detractor columns
    df_nps_historic["promoter_binary"] = df_nps_historic["nps_category"].apply(lambda x: 1 if x == "Promoter" else 0)
    df_nps_historic["detractor_binary"] = df_nps_historic["nps_category"].apply(lambda x: 1 if x == "Detractor" else 0)

    # Load Factor
    df_lf_historic['load_factor_business'] = df_lf_historic['pax_business'] / df_lf_historic['capacity_business']
    df_lf_historic['load_factor_premium_ec'] = df_lf_historic['pax_premium_ec'] / df_lf_historic['capacity_premium_ec']
    df_lf_historic['load_factor_economy'] = df_lf_historic['pax_economy'] / df_lf_historic['capacity_economy']
    
    # 5. Merge dataframes.
    SAGEMAKER_LOGGER.info("userlog: ETL 5.0 Merge dataframes.")
    cabin_to_load_factor_column = {
        'Economy': 'load_factor_economy',
        'Business': 'load_factor_business',
        'Premium Economy': 'load_factor_premium_ec'
    }

    # HISTORIC
    df_lf_historic.columns = ['date_flight_local' if x=='flight_date_local' else 
                                    'operating_airline_code' if x=='operating_carrier' else
                                    'surveyed_flight_number' if x=='op_flight_num' else
                                    x for x in df_lf_historic.columns]
    
    df_lf_historic['date_flight_local']=pd.to_datetime(df_lf_historic['date_flight_local'])
    df_lf_historic['surveyed_flight_number'] = df_lf_historic['surveyed_flight_number'].astype('float64')
    
    # List of columns to transform
    load_factor_columns = ['load_factor_business', 'load_factor_premium_ec', 'load_factor_economy']

    # Automatically determine id_vars by excluding load_factor_columns from all columns
    id_vars = [col for col in df_lf_historic.columns if col not in load_factor_columns]

    # Reshaping the DataFrame while dynamically keeping all other columns
    df_lf_historic = pd.melt(df_lf_historic, id_vars=id_vars, 
                      value_vars=load_factor_columns,
                      var_name='cabin_in_surveyed_flight', value_name='load_factor')

    # Replacing the column names in 'cabin_in_surveyed_flight' with the desired cabin types
    df_lf_historic['cabin_in_surveyed_flight'] = df_lf_historic['cabin_in_surveyed_flight'].map({
        'load_factor_business': 'Business',
        'load_factor_premium_ec': 'Premium Economy',
        'load_factor_economy': 'Economy'
    })

    df_historic = pd.merge(df_nps_historic, df_lf_historic, 
                        how='left', 
                        on=['date_flight_local', 'surveyed_flight_number', 'cabin_in_surveyed_flight', 'haul'])


    # 6. Filter out final columns for the model
    SAGEMAKER_LOGGER.info("userlog: ETL 6.0 Filter out final columns for the model")

    df_historic = df_historic[COLUMNS_TO_SAVE]
    
    # Condition for dropping rows
    condition = (df_historic['cabin_in_surveyed_flight'] == 'Premium Economy') & (df_historic['haul'] == 'SH')

    # Keeping rows that do not meet the condition
    df_historic = df_historic[~condition]

    df_historic = df_historic.drop_duplicates(subset='respondent_id', keep='first')
    
    SAGEMAKER_LOGGER.info("userlog: Size of resulting df_historic:", df_historic.shape)
    
    # Save data for historic prediction
    # Historic
    save_path = f"s3://{S3_BUCKET_NPS}/{S3_PATH_WRITE}/00_etl_step/{USE_TYPE}/{year}{month}{day}"
    SAGEMAKER_LOGGER.info("userlog: Saving information for etl step in %s.", save_path)
    df_historic.to_csv(f'{save_path}/historic.csv', index=False)







