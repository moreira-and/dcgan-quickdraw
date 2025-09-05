import mlflow
import os

# Tracking endpoint
mlflow.set_tracking_uri("http://localhost:5000")

# Para artefatos em S3/MinIO via servidor (recomendado), não precisa setar nada extra.
# Se for acessar S3 direto do cliente, exporte:
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin_password_change_me"


def test_experiment_on_server():
    mlflow.set_experiment("demo")

    with mlflow.start_run():
        mlflow.log_param("alpha", 0.1)
        mlflow.log_metric("rmse", 0.234)
        with open("notes.txt", "w") as f:
            f.write("hello artifacts")
        mlflow.log_artifact("notes.txt")
