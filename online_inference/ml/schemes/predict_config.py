from dataclasses import dataclass
from typing import List, Optional

import yaml
from marshmallow_dataclass import class_schema


@dataclass()
class PredictParams:
    input_data_path: str
    output_data_path: str
    saved_model_path: str
    target_column_name: str


PredictParamsSchema = class_schema(PredictParams)


def load_predict_config(path: str) -> PredictParams:
    with open(path, "r") as input_stream:
        schema = PredictParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
