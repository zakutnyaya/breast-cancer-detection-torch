import os
import random
import numpy as np
import torch
import torch.optim as optim


def save_model(model, path):
    """
    Helper to save torch model weights
    """
    with path.open(mode="wb") as f:
        torch.save(model.state_dict(), f)


def seed_everything(seed):
    """
    Helper to fix randomness
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))


def get_scheduler(optimizer, dataloader_len, sheduler_type, **kwargs):
    if sheduler_type == 'ReduceLROnPlateau':
        patience, factor = kwargs['patience'], kwargs['factor']
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=patience, verbose=True, factor=factor
        )
    elif sheduler_type == 'CyclicLR':
        min_lr, lr, step_size = kwargs['min_lr'], kwargs['lr'], kwargs['step_size']
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=min_lr, max_lr=lr, step_size_up=step_size, mode="triangular2"
        )
    elif sheduler_type == 'OneCycleLR':
        lr, epochs = kwargs['lr'], kwargs['epochs']
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=dataloader_len, epochs=epochs
        )
    return scheduler
