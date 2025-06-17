import os
import uuid
import pickle
import sys
from datetime import datetime
import mlflow.client
import pandas as pd

import mlflow

from dateutil.relativedelta import relativedelta
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from mlflow import MlflowClient



#input_file = f'https://s3.amazonaws.com/nyc-tlc/trip+data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'

RUN_ID = os.getenv('RUN_ID', '8df7f91d8acf4e61b8bab57232481b2c')



os.environ["AWS_ACCESS_KEY_ID"] = "myuserserviceaccount"
os.environ["AWS_SECRET_ACCESS_KEY"] = "myuserserviceaccountpassword"
os.environ["AWS_ENDPOINT_URL_S3"] = "http://localhost:9000"


# In[4]:


def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids

def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    df['ride_id'] = generate_uuids(len(df))

    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts


def load_model(run_id):
    logged_model = f's3://mlflow/8/{RUN_ID}/artifacts/model'
    mlflow.pyfunc.get_model_dependencies(logged_model)
    model = mlflow.pyfunc.load_model(logged_model)
    return model

def load_vectorizer(run_id):
    client = MlflowClient(tracking_uri='http://localhost:5005')
    dv_path = client.download_artifacts(run_id, 'preprocessor')
    dv_path = os.path.join(dv_path, 'dict_vectorizer.pkl')
    with open(dv_path, 'rb') as f:
        dv = pickle.load(f)
    return dv

def apply_model(input_file, run_id, output_file):

    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)
    dv = load_vectorizer(run_id)
    X = dv.transform(dicts)
    
    model = load_model(run_id)
    y_pred = model.predict(X)
    print(f'Mean prediction: {y_pred.mean():.2f} minutes')
    
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    #df_result['tpep_pickup_datetime'] = df['tpep_pickup_datetime']
    #df_result['PULocationID'] = df['PULocationID']
    #df_result['DOLocationID'] = df['DOLocationID']
    #df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    #df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    #df_result['model_version'] = run_id
    
    df_result.to_parquet(output_file, index=False)

def get_paths(run_date, taxi_type, run_id):
    prev_month = run_date - relativedelta(months=1)
    year = prev_month.year
    month = prev_month.month 

    #input_file = f's3://nyc-tlc/trip data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    #output_file = f's3://nyc-duration-prediction-alexey/taxi_type={taxi_type}/year={year:04d}/month={month:02d}/{run_id}.parquet'
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'

    return input_file, output_file

def ride_duration_prediction(
        taxi_type: str,
        run_id: str,
        run_date: datetime = None):
    #if run_date is None:
    #    ctx = get_run_context()
    #    run_date = ctx.flow_run.expected_start_time
    
    input_file, output_file = get_paths(run_date, taxi_type, run_id)
    dest_path = os.path.dirname(output_file)
    os.makedirs(dest_path, exist_ok=True)
    
    apply_model(
        input_file=input_file,
        run_id=run_id,
        output_file=output_file
    )


def run():
    taxi_type = sys.argv[1] # 'yellow'
    year = int(sys.argv[2]) # 2023
    month = int(sys.argv[3]) # 3

    run_id = sys.argv[4] # '1d94990ac099449696dd3414d5e93bd8' 

    ride_duration_prediction(
        taxi_type=taxi_type,
        run_id=run_id,
        run_date=datetime(year=year, month=month, day=1)
    )


if __name__ == '__main__':
    run()




