import time
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
)

from ml.schemes.train_config import PipeLineParams
from .cross_validation import get_cv


def hyperparameter_tune(params: PipeLineParams,
                        base_model: ClassifierMixin,
                        n_iter: int,
                        cv,
                        X,
                        y) -> Tuple[dict, float]:
    start_time = time.time()
    hyperparams_grid = params.tuning_grid.__dict__[base_model.__class__.__name__].__dict__
    optimal_model = RandomizedSearchCV(
        base_model,
        param_distributions=hyperparams_grid,
        n_iter=params.randomized_search_iters,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        random_state=params.random_seed,
    )

    optimal_model.fit(X, y)

    scores = cross_val_score(optimal_model, X, y, cv=cv, n_jobs=-1, scoring="f1")
    stop_time = time.time()

    print("====================")
    print(f"Updated Parameters for {str(base_model.__class__.__name__)}")
    print(
        "Cross Val Mean: {:.3f}, Cross Val Stdev: {:.3f}".format(
            scores.mean(), scores.std()
        )
    )
    print("Best Score: {:.3f}".format(optimal_model.best_score_))
    print("Best Parameters: {}".format(optimal_model.best_params_))
    print(
        "Elapsed Time:", time.strftime("%H:%M:%S", time.gmtime(stop_time - start_time))
    )
    print("====================")

    return optimal_model.best_params_, optimal_model.best_score_


def fine_tune_models(params: PipeLineParams, models, X, y):
    best_models, best_scores = [], []
    # fine tuning models
    for model in models:
        model_name = model.__class__.__name__
        if model_name not in params.tuning_grid.__dict__:
            continue
        cv = get_cv(params)
        best_params, best_score = hyperparameter_tune(
            params, model, 20, cv, X, y
        )
        model.set_params(**best_params)
        model.fit(X, y)
        best_models.append(model)
        best_scores.append(best_score)
    return best_models, best_scores


def select_best_model(best_models: List[ClassifierMixin],
                      X_train: pd.DataFrame,
                      X_test: pd.DataFrame,
                      y_train: pd.DataFrame,
                      y_test: pd.DataFrame) -> ClassifierMixin:
    best_test_scores = []
    print("====================")
    for model in best_models:
        y_test_pred = model.predict(X_test)
        test_score = f1_score(y_test, y_test_pred)
        print(f"Test score for {model.__class__.__name__}: {test_score}")
        best_test_scores.append(test_score)
    print("====================")
    best_model = best_models[np.argmax(best_test_scores)]
    print(f"Best model on test is {best_model.__class__.__name__}")

    # fit best model on whole dataset
    best_model.fit(X_train.append(X_test), y_train.append(y_test))

    return best_model
