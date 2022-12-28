import warnings
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ..schemes.predict_config import PredictParams
from ..schemes.train_config import PipeLineParams

warnings.filterwarnings("ignore")


def load_data_for_prediction(params: PredictParams):
    X = pd.read_csv(params.input_data_path)
    if 'target_column_name' in params.__dict__:
        X = X.drop(params.target_column_name, axis=1)
    return X


def load_n_split_data(params: PipeLineParams) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(params.input_data_path)
    X = train.drop(params.target_column, axis=1)
    y = train[params.target_column]
    stratification = None
    if params.split_stratification:
        stratification = y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params.split_test_size,
        random_state=params.random_seed,
        shuffle=True,
        stratify=stratification
    )
    return X_train, X_test, y_train, y_test


def save_prediction_results(params: PredictParams, X: pd.DataFrame, y_pred):
    X[str(params.target_column_name) + "_pred"] = y_pred
    X.to_csv(params.output_data_path, index=False)
