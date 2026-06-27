"""
NanoSage — a tiny GPT-style language model from scratch.

Quickstart
----------
>>> from nanosage import NanoSageLM, NanoSageConfig, BPETokenizer
>>> config = NanoSageConfig(vocab_size=8000, block_size=256, n_layer=4,
...                         n_head=4, n_embd=128)
>>> model = NanoSageLM(config)
"""

from nanosage.model.config import NanoSageConfig, ModelConfig, TrainingConfig, GenerationConfig
from nanosage.model.transformer import NanoSageLM
from nanosage.tokenizer.bpe import BPETokenizer
from nanosage.inference.generate import (
    generate,
    greedy_decode,
    sample_decode,
    beam_search,
    batch_generate,
    GenerationConfig as SamplingConfig,   # alias to avoid collision with ModelConfig.GenerationConfig
)
from nanosage.inference.chat import NanoSageChat
from nanosage.training.dataset import TextDataset, get_dataloaders
from nanosage.training.trainer import Trainer, train, evaluate, load_checkpoint
from nanosage.training.scheduler import (
    CosineWarmupScheduler,
    get_lr,
    get_cosine_lr_with_warmup,
    plot_lr_schedule,
)

__version__ = "0.1.0"
__author__  = "NanoSage Contributors"

__all__ = [
    # Model
    "NanoSageLM",
    "NanoSageConfig",
    "ModelConfig",
    "TrainingConfig",
    "GenerationConfig",
    # Tokenizer
    "BPETokenizer",
    # Inference
    "generate",
    "greedy_decode",
    "sample_decode",
    "beam_search",
    "batch_generate",
    "SamplingConfig",
    "NanoSageChat",
    # Training — dataset
    "TextDataset",
    "get_dataloaders",
    # Training — trainer
    "Trainer",
    "train",
    "evaluate",
    "load_checkpoint",
    # Training — scheduler
    "CosineWarmupScheduler",
    "get_lr",
    "get_cosine_lr_with_warmup",
    "plot_lr_schedule",
]
