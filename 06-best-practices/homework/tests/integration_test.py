from datetime import datetime
import pandas as pd
import os

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

options = {
    'client_kwargs': {
        'endpoint_url': os.getenv('AWS_S3_ENDPOINT_URL', '')
    }
}

data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

input_file = get_input_path(2023, 1)

df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

os.system('python batch.py 2023 01')

output_file = get_output_path(2023, 1)
df_output = pd.read_parquet(
    output_file, storage_options=options)

print('output shape:', df_output.shape)
print(sum(df_output['predicted_duration']))