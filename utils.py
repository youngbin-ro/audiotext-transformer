import pickle
import random
import torch
import json
import numpy as np
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


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


def get_optimizer_and_scheduler(args, model):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, verbose=True)
    if args.warmup_steps != 0:
        scheduler = GradualWarmupScheduler(
            optimizer=optimizer,
            multiplier=1,
            total_epoch=args.warmup_steps,
            after_scheduler=scheduler
        )
    return optimizer, scheduler
