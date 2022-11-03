import pytest


def test_loading():
    from ml.schemes.train_config import load_train_config
    configs = load_train_config("configs/train/default_config.yaml")
    print(str(configs.__dict__))


if __name__ == "__main__":
    test_loading()
