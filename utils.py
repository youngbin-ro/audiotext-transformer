import pickle
import random
import torch
import numpy as np


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
