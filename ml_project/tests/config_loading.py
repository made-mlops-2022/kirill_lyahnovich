def test_loading(path="../configs/default_config.yaml"):
    from ml.schemes.train_config import load_train_config
    configs = load_train_config(path)
    print(str(configs.__dict__))


if __name__ == "__main__":
    PATH = "../configs/train/default_config.yaml"
    test_loading(PATH)
