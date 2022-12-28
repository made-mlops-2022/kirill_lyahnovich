from typing import List

from sklearn.base import ClassifierMixin
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml.schemes.train_config import PipeLineParams


def get_basic_models(params: PipeLineParams) -> List[ClassifierMixin]:
    models = [
        GradientBoostingClassifier(random_state=params.random_seed),
        KNeighborsClassifier(),
        DecisionTreeClassifier(random_state=params.random_seed),
        SVC(),
        RandomForestClassifier(random_state=params.random_seed),
        AdaBoostClassifier(random_state=params.random_seed),
        MLPClassifier(random_state=params.random_seed),
        GaussianNB()
    ]
    return models
