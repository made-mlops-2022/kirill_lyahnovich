import click

from ml.prediction.predict_pipeline import run_prediction_pipeline
from ml.schemes.predict_config import load_predict_config


@click.command(name="predict")
@click.argument("predict_config_path")
def predict_command(predict_config_path: str):
    predict(predict_config_path)


def predict(predict_config_path: str):
    params = load_predict_config(predict_config_path)
    run_prediction_pipeline(params)


if __name__ == "__main__":
    predict_command()
