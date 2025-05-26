import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    mlflow.autolog(disable=False) 
    with mlflow.start_run():
        mlflow.set_tag("developer", "Valery")
#        mlflow.log_param("train-data-path", "data/green_tripdata_2023-01.parquet")
#        mlflow.log_param("valid-data-path", "data/green_tripdata_2023-02.parquet")
#        mlflow.log_param("test-data-path", "data/green_tripdata_2023-02.parquet")
        
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        
        with open('models/random_forest.bin', 'wb') as f_out:
            pickle.dump((rf), f_out)

        rmse = root_mean_squared_error(y_val, y_pred)
#        mlflow.log_metric("rmse", rmse)
#        mlflow.log_metric("min_sample_split", rf.min_samples_split) 
#        mlflow.log_artifact(local_path="models/random_forest.bin", artifact_path="models_pickle")

if __name__ == '__main__':
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-green-taxi-homework2")    
    run_train()
