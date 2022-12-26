from ml.models.serializing import load_model
from ml.schemes.inference_config import InferenceParams
from .models import HealthInfo
import time


class Inferencer:
    def __init__(self, inference_config: InferenceParams):
        time.sleep(20)
        self.model = load_model(inference_config)
        self.target_column_name = inference_config.target_column_name

    def predict(self, health_info: HealthInfo):
        response = dict()
        try:
            x = health_info.__dict__.values()
            print(list(x))
            y = self.model.predict([list(x)])[0]
            prediction = {self.target_column_name: str(y)}

            response['status'] = 'ok'
            response[self.target_column_name] = str(y)
        except Exception as e:
            response['status'] = 'error'
            #response['message'] = str(e)
        finally:
            return response
