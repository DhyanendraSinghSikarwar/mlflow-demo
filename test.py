import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
print(mlflow.get_tracking_uri())
# currently it if file, for using artifact storage, it should be set to a remote server
# or run mlflow.set_tracking_uri("http://localhost:5000") to set it to a remote server