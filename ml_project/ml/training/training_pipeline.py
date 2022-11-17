from ml.data.IO import load_n_split_data
from ml.models.basic_models import get_basic_models
from ml.training.hyperparameters_tuning import fine_tune_models, select_best_model
from ..models.serializing import save_model
from ml.schemes.train_config import PipeLineParams


def run_training_pipeline(params: PipeLineParams):
    X_train, X_test, y_train, y_test = load_n_split_data(params)
    basic_models = get_basic_models(params)
    best_models, best_scores = fine_tune_models(params, basic_models, X_train, y_train)
    best_model = select_best_model(best_models, X_train, X_test, y_train, y_test)
    save_model(params, best_model)
