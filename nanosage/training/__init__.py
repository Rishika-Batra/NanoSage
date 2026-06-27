from .dataset import TextDataset
from .trainer import Trainer, train, train_epoch, evaluate, load_checkpoint
from .scheduler import (
    get_lr,
    get_cosine_lr_with_warmup,   # backward-compat alias
    CosineWarmupScheduler,
    plot_lr_schedule,
)

__all__ = [
    # Dataset
    "TextDataset",
    # Trainer — class API
    "Trainer",
    # Trainer — functional API
    "train",
    "train_epoch",
    "evaluate",
    "load_checkpoint",
    # Scheduler
    "get_lr",
    "get_cosine_lr_with_warmup",
    "CosineWarmupScheduler",
    "plot_lr_schedule",
]
