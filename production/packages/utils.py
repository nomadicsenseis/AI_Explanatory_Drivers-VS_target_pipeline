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

def wifi_var(df):

    df["wifi_not_working"] = df["cla_600_wifi_t_f"].apply(lambda x: 1 if x in ["Could not get it to work", "No - I could not get it to work"] else 0)
    df["wifi_used_success"] = df["cla_600_wifi_t_f"].apply(lambda x: 1 if x in ["Yes", "Yes, but not enough"] else 0)
    
    return df

def group_journey_reason(df):
    
    df["tvl_journey_reason"] = df["tvl_journey_reason"].apply(lambda x: 1 if x in ["Business", "Business/work"] else 0)
    
    return df

def feature_covid_indicator(df):
    
    df.loc[df["date_flight_local"] <= "2020-03-13", "indicator_covid"] = 0
    df.loc[df["date_flight_local"] > "2020-03-13", "indicator_covid"] = 1
    
    return df

def group_residence_country(df):
    
    ESP = ["ES"]
    
    EUR = ["AT", "BE", "CH", "DE", "FR", "GB", "UX", "GR", "IL", "IT", "MA", "NL", "PT", "RU", "SE", "CZ", "DZ", 
                 "IE", "SI", "AD", "DK", "FI", "HR", "HU", "LU", "PL", "RO", "SK", "MC", "EE", "NO", "SN"]    
    
    AM_NOR = ["CA", "MX", "PR", "US"]

    LATAM = ["CR", "CU", "DO", "GT", "NI", "PA", "SV", "VE", "HN", "AR", "BR", "CL", "CO", "EC", "PE", "UY", "BO"]
    
    df.loc[df["res100_country_code_survey"].isin(ESP), "country_aggr"] = "ESP"
    df.loc[df["res100_country_code_survey"].isin(EUR), "country_aggr"] = "EUR"
    df.loc[df["res100_country_code_survey"].isin(AM_NOR), "country_aggr"] = "AM_NOR"
    df.loc[df["res100_country_code_survey"].isin(LATAM), "country_aggr"] = "LATAM"
    
    df["country_aggr"] = df["country_aggr"].fillna("Others")
    return df


def group_residence_country_custom(df):
    
    regions = {'EUROPA': ['Albania', 'AL','Andorra','AD','Armenia','AM','Austria','AT','Azerbaijan','Belarus','BY','Belgium','BE','Bosnia and Herzegovina','BA','Bulgaria','BG','Croatia','HR',
      'Cyprus','CY','Czech Republic','CZ','Denmark','DK','Estonia','EE','Faeroe Islands','FO','Finland','FI','France','FR','Georgia','GE','Germany','DE','Gibraltar','GI','Greece','GR','Greenland',
      'GL','Hungary','HU','Iceland','IS','Ireland','IE','Italy','IT','Latvia','LV','Lebanon','LB','Liechtenstein','LI','Lithuania','LT','Luxembourg','LU','Malta','MT','Moldova','MD','Monaco',
      'ME','Netherlands','NL','MK','Northern Ireland','Norway','NO','Poland','PL','Portugal','PT','MC','Romania','RO','Russia','RU','San Marino','SM','RS','Slovakia','SK','Slovenia','SI','Sweden',
      'SE','Switzerland','CH','Turkey','TR','UK (excl NI)','Ukraine','UA','GB','Vatican City','Yugoslavia'],
     'AFRICA': ['Algeria','DZ','Angola','AO','Botswana','Burkina Faso','BF','Burundi','Cameroon','CM','Cape Verde','CV','Central African Republic','Comoros','Congo',"Côte d'Ivoire",'Djibouti','DJ',
      'Egypt','EG','Equatorial Guinea','GQ','Eritrea','ER','SZ','Ethiopia','ET','French Southern Territories','Gabon','GA','GM','Ghana','GH','Guinea','Guinea-Bissau','GW','CI','Jordan','JO','Kenya',
      'KE','Liberia','LR','Libya','LY','Macedonia','Madagascar','MG','Malawi','MW','Mali','ML','Mauritania','MR','Mauritius','MU','Mayotte','YT','Morocco','MA','Mozambique','MZ','Namibia','Niger',
      'Nigeria','NG','CG','RE','Réunion','Rwanda','São Tomé and Príncipe','Senegal','SN','Seychelles','SC','Sierra Leone','SO','South Africa','ZA','SD','Sudan','Swaziland','Tanzania','TZ','The Gambia',
      'Togo','Trinidad and Tobago','TT','Tunisia','TN','Uganda','UG','Western Sahara''Zambia','Zimbabwe','ZW','Somalia','Benin'],
     'ASIA': ['Afghanistan','AF','Australia','AU','Bangladesh','Bhutan','BN','Cambodia','KH','China','CN','Cocos (Keeling) Islands','Cook Islands','East Timor','Fiji','FJ','French Polynesia','PF',
      'GU','Hong Kong','HK','India','IN','Indonesia','ID','Japan','JP','MO','Macau','Malaysia','MY','Maldives','MV','Micronesia','MN','Myanmar','MM','Nepal','NP','New Caledonia','NC','New Zealand',
      'NZ','Northern Marianas','Papua New Guinea','PG','Philippines','PH','Singapore','SG','Solomon Islands','South Korea','KR','Sri Lanka','LK','Taiwan','TW','Tajikistan','Thailand','TH','Turkmenistan',
      'TM','Uzbekistan','UZ','VU','Vanuatu','Vietnam','VN','Wallis and Futuna','Brunei','Guam'],
     'AMERICA CENTRO': ['American Samoa','Anguilla','Antigua and Barbuda','Aruba','AW','BS','Barbados','Belize','BZ','British Virgin Islands','VG','Cayman Islands','KY','Colombia','CO','Costa Rica','CR',
      'Cuba','CU','Dominica','DM','Dominican Republic','DO','El Salvador','SV','Falkland Islands','FK','Grenada','GD','Guadeloupe','GP','Guatemala','GT','Haiti','HT','Honduras','HN','Jamaica','JM',
      'Martinique','MQ','Mexico','MX','Montserrat','MS','Netherlands Antilles','Panama','PA','Puerto Rico','PR','Saint Helena','KN','Saint Lucia','Saint Vincent and the Grenadines','The Bahamas',
      'Turks and Caicos Islands','TC','US Virgin Islands','VI','Saint Kitts and Nevis'],
     'AMERICA SUR': ['Argentina','AR','Bolivia','BO','Brazil','BR','Chile','CL','Ecuador','EC','French Guiana','GF','Guyana','GY','Nicaragua','NI','Paraguay','PY','Peru','PE','Suriname','Uruguay','UY',
      'Venezuela','VE'],
     'ORIENTE MEDIO': ['Bahrain','BH','Bermuda','BM','Iran','IR','Iraq','IQ','Israel','IL','Kazakhstan','KZ','Kuwait','KW','Kyrgyzstan','KG','Oman','OM','Pakistan','PK','Qatar','QA','Saudi Arabia',
      'SA','Syria','United Arab Emirates','AE','Yemen'],
     'AMERICA NORTE': ['Canada', 'CA', 'United States', 'US'],
     'ESPAÑA': ['Spain', 'ES']}
    
    # Defining regions with corresponding country codes directly within the function
    EUROPA = regions["EUROPA"]
    AFRICA = regions["AFRICA"]
    ASIA = regions["ASIA"]
    AMERICA_CENTRO = regions["AMERICA CENTRO"]
    AMERICA_SUR = regions["AMERICA SUR"]
    ORIENTE_MEDIO = regions["ORIENTE MEDIO"]
    AMERICA_NORTE = regions["AMERICA NORTE"]
    ESPAÑA = regions["ESPAÑA"]
    
    # Assigning regions to 'country_aggr' based on the country codes
    df.loc[df["res100_country_code_survey"].isin(ESPAÑA), "country_aggr"] = "ESPAÑA"
    df.loc[df["res100_country_code_survey"].isin(EUROPA), "country_aggr"] = "EUROPA"
    df.loc[df["res100_country_code_survey"].isin(AFRICA), "country_aggr"] = "AFRICA"
    df.loc[df["res100_country_code_survey"].isin(ASIA), "country_aggr"] = "ASIA"
    df.loc[df["res100_country_code_survey"].isin(AMERICA_CENTRO), "country_aggr"] = "AMERICA CENTRO"
    df.loc[df["res100_country_code_survey"].isin(AMERICA_SUR), "country_aggr"] = "AMERICA SUR"
    df.loc[df["res100_country_code_survey"].isin(ORIENTE_MEDIO), "country_aggr"] = "ORIENTE MEDIO"
    df.loc[df["res100_country_code_survey"].isin(AMERICA_NORTE), "country_aggr"] = "AMERICA NORTE"
    
    # Defaulting to 'Others' for countries not listed
    df["country_aggr"] = df["country_aggr"].fillna("Others")
    
    return df

def issues_binary(df, dict_touchpoints):
    all_unique_issues = [i for i in dict_touchpoints["issue_type_2"].unique()]
    for issue in all_unique_issues:
        df[issue] = df[issue].apply(lambda x: 0 if pd.isna(x) else 1)

    for touchpoint in dict_touchpoints["associated_touchpoint"].unique():
        related_issues = [i for i in dict_touchpoints.loc[dict_touchpoints["associated_touchpoint"] == touchpoint]["issue_type_2"]]
        new_feature = "issues_{}".format(touchpoint)
        df[new_feature] = df[related_issues].sum(axis = 1)
        df[new_feature] = df[new_feature].apply(lambda x: 1 if x >= 1 else 0)
    return df


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


def inv_logit(x):
    return 1 / (1 + np.exp(-x))

def calculate_SHAP_and_probability_binary(model_promoter, model_detractor, df):
    # Extraer ID y fechas, manteniendo el índice
    id_df = df[['respondent_id', 'date_flight_local']]
    
    # Preparar el conjunto de datos para predicciones, excluyendo ID y fechas
    test_set = df.drop(['respondent_id', 'date_flight_local'], axis=1, errors='ignore')
    
    # Predicciones y probabilidades para promotores
    promoter_test_set = test_set.drop(['promoter_binary'], axis=1, errors='ignore')
    predictions_promoter = pd.DataFrame(model_promoter.predict(promoter_test_set), index=promoter_test_set.index, columns=["prediction_prom"])
    proba_promoter = pd.DataFrame(model_promoter.predict_proba(promoter_test_set)[:, 1], index=promoter_test_set.index, columns=["out_prob_prom"])
    
    # Predicciones y probabilidades para detractores
    detractor_test_set = test_set.drop(['detractor_binary'], axis=1, errors='ignore')
    predictions_detractor = pd.DataFrame(model_detractor.predict(detractor_test_set), index=detractor_test_set.index, columns=["prediction_det"])
    proba_detractor = pd.DataFrame(model_detractor.predict_proba(detractor_test_set)[:, 1], index=detractor_test_set.index, columns=["out_prob_det"])
    
    # Combinar resultados de predicción, manteniendo el índice original
    prediction = pd.concat([id_df, test_set, predictions_promoter, proba_promoter, predictions_detractor, proba_detractor], axis=1)
    
    # SHAP values y explicadores para el modelo promotor
    shap_Explainer_promoter = shap.TreeExplainer(model_promoter)
    shap_values_promoter = shap_Explainer_promoter.shap_values(promoter_test_set)
    feature_names = [i for i in promoter_test_set.columns]
    shap_values_prom = pd.DataFrame(shap_values_promoter, index=promoter_test_set.index, columns=[f"{i}_prom" for i in feature_names])
    shap_values_prom["base_value_prom"] = shap_Explainer_promoter.expected_value
    shap_values_prom["out_value_prom"] = shap_values_prom.sum(axis=1)
    
    # SHAP values y explicadores para el modelo detractor
    shap_Explainer_detractor = shap.TreeExplainer(model_detractor)
    shap_values_detractor = shap_Explainer_detractor.shap_values(detractor_test_set)
    shap_values_det = pd.DataFrame(shap_values_detractor, index=detractor_test_set.index, columns=[f"{i}_det" for i in feature_names])
    shap_values_det["base_value_det"] = shap_Explainer_detractor.expected_value
    shap_values_det["out_value_det"] = shap_values_det.sum(axis=1)
    
    # Combinar SHAP values con predicciones, manteniendo el índice original
    output_df = pd.concat([prediction, shap_values_prom, shap_values_det], axis=1)
    
    # Devolver el dataframe de salida
    return output_df


def from_shap_to_probability_binary(df, features_dummy, label_binary):
    output_df = df.copy()
    
    # Determinar el sufijo basado en el tipo de modelo (promoter o detractor)
    class_suffix = '_prom' if label_binary == 'promoter_binary' else '_det'
    
    # Identificar columnas de SHAP para la clase de interés, asumiendo que ya tienen el sufijo correcto
    shap_columns = [col for col in df.columns if col.endswith(class_suffix)]
    base_value_col = f'base_value{class_suffix}'
    
    # Convertir el valor base a probabilidades y actualizar el nombre de la columna
    output_df[f'base_prob{class_suffix}'] = inv_logit(output_df[base_value_col])
    
    # Convertir valores SHAP a probabilidades sin cambiar los nombres de las columnas
    for col in shap_columns:
        output_df[col] = inv_logit(output_df[col])
    
    # Asegurarse de incluir solo las columnas relevantes en el DataFrame final
    relevant_columns = ['respondent_id', 'date_flight_local'] + shap_columns + [f'base_prob{class_suffix}'] + features_dummy
    output_df = output_df[relevant_columns]
    print(output_df)
    return output_df

def adjust_shap_values_binary(shap_values, base_prob, out_prob):
    """Ajustar los valores SHAP para un modelo binario basado en la distancia."""
    # Calcular la distancia total deseada entre la probabilidad base y la de salida
    total_distance = out_prob - base_prob
    # Calcular la suma total de los valores SHAP
    total_shap = np.sum(shap_values)
    # Calcular el factor de ajuste si la suma total de SHAP no es cero
    adjustment_factor = total_distance / total_shap if total_shap != 0 else 0
    # Ajustar los valores SHAP
    return shap_values * adjustment_factor

def from_shap_to_probability_binary(df, features_dummy, label_binary):
    output_df = df.copy()
    
    # Determinar el sufijo basado en el tipo de modelo (promoter o detractor)
    class_suffix = '_prom' if label_binary == 'promoter_binary' else '_det'
    
    # Identificar columnas de SHAP para la clase de interés, asumiendo que ya tienen el sufijo correcto
    shap_columns = [f'{feature}{class_suffix}' for feature in features_dummy if f'{feature}{class_suffix}' in df.columns]
    base_value_col = f'base_value{class_suffix}'
    out_prob_col = f'out_prob{class_suffix}'

    # Calcular la probabilidad base usando softmax o inv_logit según sea apropiado
    output_df[f'base_prob{class_suffix}'] = inv_logit(output_df[base_value_col])

    for index, row in output_df.iterrows():
        # Extraer los valores SHAP para ajustar
        shap_values = row[shap_columns].values
        # Calcular los valores SHAP ajustados
        adjusted_shap_values = adjust_shap_values_binary(shap_values, row[f'base_prob{class_suffix}'], row[out_prob_col])
        # Actualizar el DataFrame con los valores SHAP ajustados
        output_df.loc[index, shap_columns] = adjusted_shap_values

    # Incluir solo las columnas relevantes en el DataFrame final
    relevant_columns = ['respondent_id', 'date_flight_local'] + shap_columns + [f'base_prob{class_suffix}', out_prob_col] + features_dummy
    print(output_df)
    output_df = output_df[relevant_columns]
    
    return output_df

def predict_and_explain(model_prom, model_det, df, features_dummy):
    """
    Realiza predicciones y genera explicaciones para modelos de promotores y detractores
    para todo el dataframe.

    Args:
    - model_prom: Modelo entrenado para predecir promotores.
    - model_det: Modelo entrenado para predecir detractores.
    - df: DataFrame con los datos.
    - features_dummy: Lista de características utilizadas para las predicciones.

    Returns:
    - Df final con .data, .values, .base_value, y predicciones.
    """
    # 1. Asumiendo que las funciones de cálculo de SHAP y probabilidad ya están implementadas y ajustadas para usar df
    df_contrib = calculate_SHAP_and_probability_binary(model_prom, model_det, df)

    # 3. Convertir valores SHAP a probabilidad
    df_probability_prom = from_shap_to_probability_binary(df_contrib, features_dummy, 'promoter_binary')
    df_probability_det = from_shap_to_probability_binary(df_contrib, features_dummy, 'detractor_binary')

    # 4. Concatenar DataFrames para ambos modelos
    df_probability_prom = df_probability_prom.reset_index(drop=True)
    df_probability_det = df_probability_det.reset_index(drop=True)
    unique_columns_det = [col for col in df_probability_det.columns if col not in df_probability_prom.columns]
    df_probability_binary = pd.concat([df_probability_prom, df_probability_det[unique_columns_det]], axis=1)

    # 5. Calcular columnas NPS con la diferencia entre _prom y _det
    for column in df_probability_binary.columns:
        if '_prom' in column:
            base_name = column.split('_prom')[0]
            det_column = f'{base_name}_det'
            if det_column in df_probability_binary.columns:
                nps_column = f'{base_name}_nps'
                df_probability_binary[nps_column] = df_probability_binary[column] - df_probability_binary[det_column]

    return df_probability_binary

