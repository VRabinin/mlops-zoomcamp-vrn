import batch
from datetime import datetime
import pandas as pd

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df = pd.DataFrame(data, columns=columns)

def test_prepare_data():
    df_result = batch.prepare_data(df, ['PULocationID', 'DOLocationID'])

    assert df_result.shape[0] == 2 # Only the first and the second rows should remain
    assert df_result.shape[1] == 5
    assert df_result['duration'].min() >= 1
    assert df_result['duration'].max() <= 60
 #   assert df_result['PULocationID'].dtype == 'str'
 #   assert df_result['DOLocationID'].dtype == 'str'
 
def test_get_input_path():
    year = 2023
    month = 1
    expected_path = 's3://nyc-duration/in/2023-01.parquet'
    assert batch.get_input_path(year, month) == expected_path
    
def test_get_output_path():
    year = 2023
    month = 1
    expected_path = 's3://nyc-duration/out/2023-01.parquet'
    assert batch.get_output_path(year, month) == expected_path