from dataclasses import dataclass
from typing import List

import yaml
from marshmallow_dataclass import class_schema


@dataclass()
class RFCParams:
    max_depth: List[int]
    n_estimators: List[int]
    bootstrap: List[bool]
    criterion: List[str]


@dataclass()
class SVCParams:
    C: List[float]
    tol: List[float]
    shrinking: List[bool]
    kernel: List[str]


@dataclass()
class AdaBoostParams:
    n_estimators: List[int]
    algorithm: List[str]
    learning_rate: List[float]


@dataclass()
class ParamGrid:
    RandomForestClassifier: RFCParams
    SVC: SVCParams
    AdaBoostClassifier: AdaBoostParams


@dataclass()
class PipeLineParams:
    input_data_path: str
    output_model_path: str
    target_column: str
    split_test_size: float
    split_stratification: bool
    randomized_search_iters: int
    cv_n_folds: int
    random_seed: int
    tuning_grid: ParamGrid


PipeLineParamsSchema = class_schema(PipeLineParams)


def load_train_config(path: str) -> PipeLineParamsSchema:
    with open(path, "r") as input_stream:
        schema = PipeLineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
