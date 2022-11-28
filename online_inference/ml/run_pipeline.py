import click

from ml import fix_seed
from ml.schemes.train_config import load_train_config
from ml.training.training_pipeline import run_training_pipeline


@click.command(name="run_pipeline")
@click.argument("config_path")
def run_pipeline_command(config_path: str):
    run_pipeline(config_path)


def run_pipeline(config_path: str):
    params = load_train_config(config_path)
    fix_seed(params.random_seed)
    run_training_pipeline(params)


if __name__ == "__main__":
    run_pipeline_command()
