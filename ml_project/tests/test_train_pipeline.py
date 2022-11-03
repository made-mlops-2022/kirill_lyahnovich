from ml.run_pipeline import run_pipeline


def test_train_pipeline():
    run_pipeline("../configs/train/fast_config.yaml")


if __name__ == "__main__":
    test_train_pipeline()
