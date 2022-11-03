import os
import random
import warnings
import numpy as np

warnings.filterwarnings("ignore")
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
