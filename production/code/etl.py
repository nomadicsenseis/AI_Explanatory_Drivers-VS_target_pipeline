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

        # "--use_type": specifies the type of use, it can be "predict" to predict or "train" to train the model.
        parser.add_argument("--use_type", type=str, choices=["predict", "train"])

        # Finally, we parse the arguments and store them in the self.args property of the class.
        self.args = parser.parse_args()

if __name__ == "__main__":

    """Main functionality of the script."""
    # Arguments
    SAGEMAKER_LOGGER.info("userlog: Starting %s step...", STEP)
    arguments = Arguments()
    arguments.info(logger=SAGEMAKER_LOGGER)
    args = arguments.get_arguments()
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

    # Calculate yesterday's date
    yesterday_date = execution_date - timedelta(days=1)
    # Format dates as strings for S3 prefixes
    today_date_str = execution_date.strftime("%Y-%m-%d")
    yesterday_date_str = yesterday_date.strftime("%Y-%m-%d")
    

    # Config file read
    # config = utils.read_config_data(path=SparkFiles.get(filename="config.yml"))
    # config_variables = config.get("VARIABLES")
    # config_etl = config.get(STEP)

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

    # READ PREVIOUS NPS DATA (FOR INCREMENTAL)
    yesterday_nps_surveys_prefix = f'{S3_PATH_READ_NPS}/insert_date_ci={yesterday_date_str}/'
    yesterday_s3_keys = [item.key for item in s3_resource.Bucket(S3_BUCKET_NPS).objects.filter(Prefix=yesterday_nps_surveys_prefix)]
    yesterday_preprocess_paths = [f"s3://{S3_BUCKET_NPS}/{key}" for key in yesterday_s3_keys]

    SAGEMAKER_LOGGER.info("userlog: Read historic nps_surveys data path %s.", yesterday_nps_surveys_prefix)
    df_nps_yesterday = pd.DataFrame()
    for file in yesterday_preprocess_paths:
        df = pd.read_csv(file)
        df_nps_yesterday = pd.concat([df_nps_yesterday, df], axis=0)
    df_nps_yesterday = df_nps_yesterday.reset_index(drop=True)

    # INCREMENTAL NPS  
    SAGEMAKER_LOGGER.info("Pre-merge shapes - Historic: %s, Yesterday: %s", df_nps_historic.shape, df_nps_yesterday.shape)

    df_nps_incremental = pd.merge(df_nps_historic, df_nps_yesterday, how='left', indicator=True, on=df_nps_historic.columns.tolist())
    df_nps_incremental = df_nps_incremental[df_nps_incremental['_merge'] == 'left_only']
    df_nps_incremental = df_nps_incremental.drop(columns=['_merge'])
    df_nps_incremental = df_nps_incremental.reset_index(drop=True)

    
    # READ LF DATA SOURCE
    # lf_dir = 's3://ibdata-prod-ew1-s3-customer/customer/load_factor_to_s3_nps_model/'    
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
                if today_date_str in file_path:
                    df_lf_incremental = pd.read_csv(f)
                    SAGEMAKER_LOGGER.info(f"Loading incremental: {df_lf_incremental.shape}")
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
        
    # # Assume rol for aip again
    # sts_client = boto3.client('sts')
    # assumed_role = sts_client.assume_role(
    #     RoleArn="arn:aws:iam::077156906314:role/ibdata-aip-role-sagemaker-customer-user",
    #     RoleSessionName="test"
    # )
    # credentials = assumed_role['Credentials']
    # fs = s3fs.S3FileSystem(key=credentials['AccessKeyId'], secret=credentials['SecretAccessKey'], token=credentials['SessionToken'])
    # ETL Code

    # 1. Filter dataframes by carrier code.
    SAGEMAKER_LOGGER.info("userlog: ETL 1.0 Filter dataframes by carrier code.")
    # NPS HISTORIC
    condition_1 = (df_nps_historic['operating_airline_code'].isin(['IB', 'YW']))
    condition_2 = ((df_nps_historic['invitegroup_ib'] != 3) | (df_nps_historic['invitegroup_ib'].isnull()))
    condition_3 = (df_nps_historic['invitegroup'] == 2)
    
    df_nps_historic = df_nps_historic.loc[condition_1 & (condition_2 & condition_3)]

    # NPS INCREMENTAL
    condition_1 = (df_nps_incremental['operating_airline_code'].isin(['IB', 'YW']))
    condition_2 = ((df_nps_incremental['invitegroup_ib'] != 3) | (df_nps_incremental['invitegroup_ib'].isnull()))
    condition_3 = (df_nps_incremental['invitegroup'] == 2)

    df_nps_incremental = df_nps_incremental.loc[condition_1 & (condition_2 & condition_3)]

    # LOAD FACTOR HISTORIC
    df_lf_historic = df_lf_historic.loc[(df_lf_historic['operating_carrier'].isin(['IB', 'YW']))]

    # LOAD FACTOR INCREMENTAL
    df_lf_incremental = df_lf_incremental.loc[(df_lf_incremental['operating_carrier'].isin(['IB', 'YW']))]


    # 2. Transform date column to datetime format
    SAGEMAKER_LOGGER.info("userlog: ETL 2.0 Transform date column to datetime format.")
    delay_features = ['real_departure_time_local', 'scheduled_departure_time_local']
    for feat in delay_features:
        df_nps_historic[feat] = pd.to_datetime(df_nps_historic[feat], format="%Y%m%d %H:%M:%S", errors = 'coerce')
        df_nps_incremental[feat] = pd.to_datetime(df_nps_incremental[feat], format="%Y%m%d %H:%M:%S", errors = 'coerce')
            
    df_nps_historic['delay_departure'] = (df_nps_historic['real_departure_time_local'] - df_nps_historic['scheduled_departure_time_local']).dt.total_seconds()/60
    df_nps_incremental['delay_departure'] = (df_nps_incremental['real_departure_time_local'] - df_nps_incremental['scheduled_departure_time_local']).dt.total_seconds()/60
    
    # NPS
    df_nps_historic['date_flight_local'] = pd.to_datetime(df_nps_historic['date_flight_local'])
    df_nps_incremental['date_flight_local'] = pd.to_datetime(df_nps_incremental['date_flight_local'])

    # Load Factor
    df_lf_historic['flight_date_local'] = pd.to_datetime(df_lf_historic['flight_date_local'])
    df_lf_incremental['flight_date_local'] = pd.to_datetime(df_lf_incremental['flight_date_local'])

    # 3. Filter out covid years
    SAGEMAKER_LOGGER.info("userlog: ETL 3.0 Filter out covid years.")
    # NPS (historic)
    df_nps_historic = df_nps_historic[~df_nps_historic['date_flight_local'].dt.year.isin([2020, 2021])]
    # Load factor (historic)
    df_lf_historic = df_lf_historic[~df_lf_historic['flight_date_local'].dt.year.isin([2020, 2021])]

    # 4. Create otp, promoter, detractor and load factor columns.
    SAGEMAKER_LOGGER.info("userlog: ETL 4.0 Create otp, promoter, detractor and load factor columns.")
    # OTP
    df_nps_historic['otp15_takeoff'] = (df_nps_historic['delay_departure'] > 15).astype(int)
    df_nps_incremental['otp15_takeoff'] = (df_nps_incremental['delay_departure'] > 15).astype(int)

    # Promoter and Detractor columns
    df_nps_historic["promoter_binary"] = df_nps_historic["nps_category"].apply(lambda x: 1 if x == "Promoter" else 0)
    df_nps_historic["detractor_binary"] = df_nps_historic["nps_category"].apply(lambda x: 1 if x == "Detractor" else 0)
    df_nps_incremental["promoter_binary"] = df_nps_incremental["nps_category"].apply(lambda x: 1 if x == "Promoter" else 0)
    df_nps_incremental["detractor_binary"] = df_nps_incremental["nps_category"].apply(lambda x: 1 if x == "Detractor" else 0)

    # Load Factor
    df_lf_historic['load_factor_business'] = df_lf_historic['pax_business'] / df_lf_historic['capacity_business']
    df_lf_historic['load_factor_premium_ec'] = df_lf_historic['pax_premium_ec'] / df_lf_historic['capacity_premium_ec']
    df_lf_historic['load_factor_economy'] = df_lf_historic['pax_economy'] / df_lf_historic['capacity_economy']

    df_lf_incremental['load_factor_business'] = df_lf_incremental['pax_business'] / df_lf_incremental['capacity_business']
    df_lf_incremental['load_factor_premium_ec'] = df_lf_incremental['pax_premium_ec'] / df_lf_incremental['capacity_premium_ec']
    df_lf_incremental['load_factor_economy'] = df_lf_incremental['pax_economy'] / df_lf_incremental['capacity_economy']

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

    df_historic = pd.merge(df_nps_historic, df_lf_historic, 
                        how='left', 
                        on=['date_flight_local', 'operating_airline_code', 'surveyed_flight_number', 'haul'])

    df_historic['load_factor'] = df_historic.apply(lambda row: row[cabin_to_load_factor_column[row['cabin_in_surveyed_flight']]], axis=1)

    # INCREMENTAL
    df_lf_incremental.columns = ['date_flight_local' if x=='flight_date_local' else 
                                    'operating_airline_code' if x=='operating_carrier' else
                                    'surveyed_flight_number' if x=='op_flight_num' else
                                    x for x in df_lf_incremental.columns]

    df_incremental = pd.merge(df_nps_incremental, df_lf_incremental, 
                        how='left', 
                        on=['date_flight_local', 'operating_airline_code', 'surveyed_flight_number', 'haul'])

    df_incremental['load_factor'] = df_incremental.apply(lambda row: row[cabin_to_load_factor_column[row['cabin_in_surveyed_flight']]], axis=1)

    # 6. Filter out final columns for the model
    SAGEMAKER_LOGGER.info("userlog: ETL 6.0 Filter out final columns for the model")
    features_dummy = ['ticket_price', 'load_factor', 'otp15_takeoff'] + ['bkg_200_journey_preparation', 'pfl_100_checkin', 
                  'pfl_200_security', 'pfl_300_lounge', 'pfl_500_boarding', 'ifl_300_cabin', 
                  'ifl_200_flight_crew_annoucements', 'ifl_600_wifi', 'ifl_500_ife', 'ifl_400_food_drink', 
                  'ifl_100_cabin_crew', 'arr_100_arrivals', 'con_100_connections', 
                  'loy_200_loyalty_programme', 'img_310_ease_contact_phone']

    labels = ['promoter_binary', 'detractor_binary']

    df_historic = df_historic[['respondent_id' , 'date_flight_local'] + features_dummy + labels]
    df_incremental = df_incremental[['respondent_id' , 'date_flight_local'] + features_dummy + labels]

    df_historic = df_historic.drop_duplicates()
    df_incremental = df_incremental.drop_duplicates()
    
    SAGEMAKER_LOGGER.info("userlog: Size of resulting df_historic:", df_historic.shape)
    SAGEMAKER_LOGGER.info("userlog: Size of resulting df_incremental:", df_incremental.shape)
    # Save data for training
    # Historic
    save_path = f"s3://{S3_BUCKET_NPS}/{S3_PATH_WRITE}/00_etl_step/{USE_TYPE}/{year}{month}{day}"
    SAGEMAKER_LOGGER.info("userlog: Saving information for etl step in %s.", save_path)
    df_historic.to_csv(f'{save_path}/historic.csv', index=False)
    # Incremental
    save_path = f"s3://{S3_BUCKET_NPS}/{S3_PATH_WRITE}/00_etl_step/{USE_TYPE}/{year}{month}{day}"
    SAGEMAKER_LOGGER.info("userlog: Saving information for etl step in %s.", save_path)
    df_incremental.to_csv(f'{save_path}/incremental.csv', index=False)






