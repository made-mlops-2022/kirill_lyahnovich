from ml.data.IO import load_data_for_prediction, save_prediction_results
from ..models.serializing import load_model
from ..schemes.predict_config import PredictParams


def run_prediction_pipeline(params: PredictParams):
    X = load_data_for_prediction(params)
    model = load_model(params)
    y_pred = model.predict(X)
    save_prediction_results(params, X, y_pred)
