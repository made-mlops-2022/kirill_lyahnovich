from sklearn.model_selection import KFold

from ml.schemes.train_config import PipeLineParams


def get_cv(params: PipeLineParams):
    cv = KFold(params.cv_n_folds, shuffle=True, random_state=params.random_seed)
