from dataclasses import dataclass
from typing import List, Optional

import yaml
from marshmallow_dataclass import class_schema


@dataclass()
class InferenceParams:
    saved_model_path: str
    target_column_name: Optional[str]


InferenceParamsSchema = class_schema(InferenceParams)


def load_inference_config(path: str) -> InferenceParamsSchema:
    with open(path, "r") as input_stream:
        schema = InferenceParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
