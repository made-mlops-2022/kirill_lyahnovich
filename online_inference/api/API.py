from fastapi import FastAPI

from api.inferencer import Inferencer
from ml.schemes.inference_config import InferenceParams


class API(FastAPI):

    def __init__(self, inference_params : InferenceParams):
        FastAPI.__init__(self)
        self.inferencer = Inferencer(inference_params)
