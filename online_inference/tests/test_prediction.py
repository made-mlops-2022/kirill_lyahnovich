from ml.predict import predict
import pytest

def test_prediction():
    predict("configs/predict/predict_config.yml")


if __name__ == "__main__":
    test_prediction()
