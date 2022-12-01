import math
import pytest
import pandas as pd
import requests
import json

URL = 'http://127.0.0.1:8000/api/predict'


def test_prediction():
    predictions = pd.read_csv('data/predictions/predictions.csv')
    inputs = predictions.drop(columns=['condition_pred'], axis=1)
    true_predicts = list(predictions['condition_pred'])
    api_predictions = []
    records = json.loads(inputs.to_json(orient='records'))
    for body in records:
        response = requests.api.post(url=URL, json=body)
        api_predictions.append(int(response.json()['condition']))

    for true_pred, api_pred in zip(true_predicts, api_predictions):
        assert math.isclose(true_pred, api_pred)


if __name__ == "__main__":
    test_prediction()
