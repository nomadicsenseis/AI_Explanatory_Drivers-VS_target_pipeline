from subprocess import check_call
from sys import executable

STEP = "TRAIN"

check_call([executable, "-m", "pip", "install", "-r", f"./{STEP.lower()}.txt"])

import argparse
import logging
from json import dumps as jdumps
from os import environ
from pickle import dumps as pkl_dumps

import plots
import train_utils as tutils
import utils
from boto3 import resource
from catboost import CatBoostClassifier
from pandas import read_csv
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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


def cast_variables_types(df):
    categorical_features = df.select_dtypes(include=['object'])
    int_features = df.select_dtypes(include=['int64'])
    float_features = df.select_dtypes(include=['float64'])
    boolean_features = df.select_dtypes(include=['bool'])
    for c in categorical_features:
        df[c] = pd.Categorical(df[c])
    for c in int_features:
        df[c] = df[c].astype('int16', errors='raise')
    for c in float_features:
        df[c] = df[c].astype('float16', errors='raise')
    for c in boolean_features:
        df[c] = df[c].astype('bool', errors='raise')
    return df


def dumpModel(model,metrics_json):
    SAGEMAKER_LOGGER.info(f"Dumping model...")
    fitted_clf_model = pickle.dumps(model)
    s3_resource.Object(
        S3_BUCKET,
        f"{save_path}/model/{config['TRAIN']['MODEL_NAME']}",
    ).put(Body=fitted_clf_model)
    SAGEMAKER_LOGGER.info(f"Dumping metrics...")
    s3_resource.Object(
        S3_BUCKET,
        f"{save_path}/metrics/clf_metrics.json",
    ).put(Body=(bytes(json.dumps(metrics_json).encode("UTF-8"))))


def eval_set(X_set,y_set,model,features,set_name):
    X_set = cast_variables_types(X_set)
    SAGEMAKER_LOGGER.info(f"X {set_name} SHAPE {X_test.shape} ; {y_set.shape}")
    SAGEMAKER_LOGGER.info(f"WARNING X {set_name}: rows with na {X_set[features].isnull().any(axis=1).sum()}")
    missing_rows = X_set[features].isnull().any(axis=1)
    X_set = X_set[~missing_rows]
    y_set = y_set[~missing_rows]
    SAGEMAKER_LOGGER.info(f"X {set_name} SHAPE {X_set.shape} ; {y_set.shape}")
    metrics_set = get_metrics(model, X_set[features], y_set, set_name)
    del X_set
    del y_set
    gc.collect()
    return metrics_set

if __name__ == "__main__":

    # Crea un manejador de archivos que escriba los logs a un archivo
    file_handler = logging.FileHandler('/opt/ml/processing/output/log.txt')
    file_handler.setLevel(logging.INFO)

    # Añade el manejador de archivos al logger
    SAGEMAKER_LOGGER.addHandler(file_handler)
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

    # Paths
    read_path = f"{S3_PATH_WRITE}/01_preprocess_step/train/{year}{month}{day}"
    save_path = f"{S3_PATH_WRITE}/02_train_step/{year}{month}{day}"
    SAGEMAKER_LOGGER.info("userlog: Read date path %s.", read_path)

    # Read data
    X_train = read_csv(f"s3://{S3_BUCKET}/{read_path}/data_train/X_train.csv")
    y_train = read_csv(f"s3://{S3_BUCKET}/{read_path}/data_train/y_train.csv")
    X_train = cast_variables_types(X_train)
    SAGEMAKER_LOGGER.info(f"X_TRAIN SHAPE {X_train.shape} ; {y_train.shape}")
    SAGEMAKER_LOGGER.info(f"WARNING X_TRAIN: rows with na {X_train[features].isnull().any(axis=1).sum()}")
    missing_rows = X_train[features].isnull().any(axis=1)
    X_train = X_train[~missing_rows]
    y_train = y_train[~missing_rows]
    SAGEMAKER_LOGGER.info(f"X_TRAIN SHAPE {X_train.shape} ; {y_train.shape}")

    # Estimator
    SAGEMAKER_LOGGER.info(f"userlog: INPUT COLS: {str(features)}")
    model = CatBoostClassifier(auto_class_weights="Balanced", random_state=42, silent=True)
    model = model.fit(X_train[features], y_train)
    metrics_train = get_metrics(model, X_train[features], y_train, 'train')
    SAGEMAKER_LOGGER.info(f"Trained model: {str(model)}")

    del X_train
    del y_train
    gc.collect()

    X_test = read_csv(f"s3://{S3_BUCKET}/{read_path}/data_test/X_test.csv")
    y_test = read_csv(f"s3://{S3_BUCKET}/{read_path}/data_test/y_test.csv")
    metrics_test = eval_set(X_test, y_test, model, features, 'test')

    X_val = read_csv(f"s3://{S3_BUCKET}/{read_path}/data_val/X_val.csv")
    y_val = read_csv(f"s3://{S3_BUCKET}/{read_path}/data_val/y_val.csv")
    metrics_val = eval_set(X_val, y_val, model, features, 'validation')

    X_oos = read_csv(f"s3://{S3_BUCKET}/{read_path}/data_OOS/X_out_of_sample.csv")
    y_oos = read_csv(f"s3://{S3_BUCKET}/{read_path}/data_OOS/y_out_of_sample.csv")
    metrics_oos = eval_set(X_oos, y_oos, model, features, 'out_of_sample')

    s3_resource = resource("s3")
    SAGEMAKER_LOGGER.info(f"Dumping metrics...")
    clf_metrics = {**metrics_train, **metrics_test, **metrics_val, **metrics_oos}

    SAGEMAKER_LOGGER.info(f"METRICS: {clf_metrics}")
    s3_resource = boto3.resource('s3')

    if not FIRST_TIME:
        # Recuperamos el modelo vigente
        my_bucket = s3_resource.Bucket(S3_BUCKET)
        model_paths = []

        for obj in my_bucket.objects.filter(Prefix=f'{S3_PATH_WRITE}/02_train_step/'):
            match = re.search(r'(\d+)/', obj.key)
            if match:
                model_paths.append(obj.key)

        # En este caso, asumimos que sólo hay una carpeta numérica
        latest_model_path = sorted(model_paths)[-1]
        SAGEMAKER_LOGGER.info(f"latest_model_path: {latest_model_path}")

        prod_model = (
            my_bucket.Object(f"{latest_model_path}").get()
        )
        prod_model = pickle.loads(prod_model["Body"].read())


        def get_number_of_training_variables(model):
            if hasattr(model, 'n_features_in_'):
                return model.n_features_in_
            if hasattr(model, 'n_features_'):
                return model.n_features_
            else:
                return None


        number_of_training_variables = get_number_of_training_variables(model)

        if number_of_training_variables == len(features):
            prod_model_metrics_test = get_metrics(prod_model, X_test[features], y_test, 'test')
            ## Proceso de Checkeo si hay que actualizar modelo
            compare_metrics = ["Recall"]
            min_increase = 0.05
            min_thresholds = {"AUC": 0.9, "Recall": 0.8, "Precision": 0.2, "Accuracy": 0.8, "F1-Score": 0.25}
            result = should_update_model(prod_model_metrics_test, clf_metrics, compare_metrics, min_increase,
                                         min_thresholds)

            print("Actualizar modelo:", result)
            clf_metrics_json = json.dumps(clf_metrics)

            if result:
                dumpModel(model, clf_metrics)
            else:
                path = f"{S3_PATH_WRITE}/02_train_step/not_deployed"
                SAGEMAKER_LOGGER.info(f"Dumping FAILED MODEL metrics...")
                s3_resource.Object(
                    S3_BUCKET,
                    f"{path}/{year}{month}{day}_metrics.json",
                ).put(Body=(bytes(json.dumps(clf_metrics_json).encode("UTF-8"))))
        else:
            dumpModel(model, clf_metrics)
    else:
        dumpModel(model, clf_metrics)
