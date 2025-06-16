# repo.py
from dagster import job, op, asset, repository, Config
import psycopg2, os
import pandas as pd
import pickle
from pathlib import Path
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

class MyOpConfig(Config):
    year: str
    month: str

class MyAssetConfig(Config):
    person_name: str

@asset
def greeting(config: MyAssetConfig) -> str:
    return f"hello {config.person_name}"

def read_dataframe(context, year, month):
    #url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    url = f'/opt/dagster/app/data/yellow_tripdata_{year}-{month:02d}.parquet'
    context.log.info(f"Reading data from {url}")
    df = pd.read_parquet(url)
    context.log.info(f"Successfully read {len(df)} for {year}-{month}")
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    return df

def create_X(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_model(context, X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        #intercept = 0
        #mlflow.log_metric("intercept", intercept)
        context.log.info(f"RMSE: {rmse}")
        #context.log.info(f"Intercept: {intercept}")
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id

#@op
#def configure_pipeline(context, config: MyOpConfig):
##    context.log.info(f"Configuring pipeline for year {config.year} and month {config.month}")
 #   # Here you can set up any necessary configurations or connections
 #   # For example, setting up a database connection or API client
 #   # This is just a placeholder for demonstration purposes
 #   return config.year, config.month
@op
def run_pipeline(context, config: MyOpConfig):
    year = int(config.year)
    month = int(config.month)
    context.log.info(f"Reading train data for {year}-{month}")
    df_train = read_dataframe(context, year=year, month=month)
    context.log.info(f"Filtered records - {len(df_train)} for {year}-{month}")

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    context.log.info(f"Reading validation data for {next_year}-{next_month}")   
    df_val = read_dataframe(context, year=next_year, month=next_month)

    context.log.info(f"Createing feature matrix for train dataset")
    X_train, dv = create_X(df_train)
    context.log.info(f"Createing feature matrix for validation dataset")   
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    context.log.info(f"Train Model")   
    run_id = train_model(context, X_train, y_train, X_val, y_val, dv)
    context.log.info(f"MLflow run_id: {run_id}")
    return run_id

@job
def etl_job():
    #year, month = configure_pipeline()
    run_id = run_pipeline()


@repository
def my_repository():
    return [etl_job]