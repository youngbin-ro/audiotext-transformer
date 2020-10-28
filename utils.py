import pickle
import random
import torch
import json
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
