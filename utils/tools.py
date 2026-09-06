import os
import random
import numpy as np
import torch


def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.enabled = True


class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-5, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        else:
            if self.mode == 'min':
                improvement = score < (self.best_score - self.min_delta)
            else:
                improvement = score > (self.best_score + self.min_delta)
            if improvement:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        return self.early_stop, self.best_epoch
