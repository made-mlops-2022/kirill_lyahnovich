import pickle
from typing import Optional, Union

from sklearn.base import ClassifierMixin

from ml.schemes.predict_config import PredictParams
from ml.schemes.train_config import PipeLineParams
from ml.schemes.inference_config import InferenceParams


def save_model(params: PipeLineParams, model: ClassifierMixin):
    path = params.output_model_path
    with open(path, 'wb') as file:
        pickle.dump(model, file)


def load_model(params: Union[PredictParams, InferenceParams]) -> Optional[ClassifierMixin]:
    model = None
    with open(params.saved_model_path, 'rb') as file:
        model = pickle.load(file)
    return model
