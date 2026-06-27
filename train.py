import argparse
import os
import sys
import time
import math
import torch
from torch.utils.data import DataLoader

from nanosage.tokenizer.bpe import BPETokenizer
from nanosage.model.config import ModelConfig, TrainingConfig
from nanosage.model.transformer import NanoSageLM
from nanosage.training.dataset import (
    download_pretrain_data,
    prepare_pretrain_data,
    get_dataloaders,
)
from nanosage.training.scheduler import CosineWarmupScheduler
import nanosage.training.trainer as trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain NanoSage GPT-style LLM")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    # Auxiliary utility flags
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cuda/mps/cpu)")
    parser.add_argument("--data_path", type=str, default="nanosage/data/raw/tinystories.txt", help="Path to raw txt dataset")
    parser.add_argument("--max_steps", type=int, default=2000, help="Max steps per epoch")
    parser.add_argument("--tokenizer_path", type=str, default="nanosage/checkpoints/tokenizer.json", help="Path to save tokenizer state")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1) Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 2) Initialize ModelConfig and TrainingConfig
    model_config = ModelConfig()
    training_config = TrainingConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    # 3) Download and prepare data if not already done
    raw_path = args.data_path
    if not os.path.exists(raw_path):
        print(f"Raw dataset not found at {raw_path}. Downloading...")
        download_pretrain_data(output_path=raw_path)
    
    # 4) Train BPETokenizer on the dataset and save it
    tokenizer = BPETokenizer()
    tokenizer_path = args.tokenizer_path
    if os.path.exists(tokenizer_path):
        print(f"Loading existing BPE tokenizer from {tokenizer_path}...")
        tokenizer.load(tokenizer_path)
    else:
        print("Training BPE tokenizer from scratch...")
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        tokenizer.train(raw_text, vocab_size=model_config.vocab_size)
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
        
    # Prepare tokenized splits if they don't exist
    train_bin = "nanosage/data/processed/train_tokens.npy"
    val_bin = "nanosage/data/processed/val_tokens.npy"
    if not os.path.exists(train_bin) or not os.path.exists(val_bin):
        print("Tokenizing raw dataset into train/val splits...")
        prepare_pretrain_data(tokenizer, raw_path=raw_path, train_out=train_bin, val_out=val_bin)

    # 5) Build NanoSage model, print parameter count
    model = NanoSageLM(model_config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with vocabulary size: {model_config.vocab_size}")
    print(f"Total model parameters: {param_count / 1e6:.2f}M")

    # 6) Initialize AdamW optimizer
    optimizer = model.configure_optimizers(
        weight_decay=training_config.weight_decay,
        learning_rate=training_config.learning_rate,
        betas=(0.9, 0.95),
        device_type=device.type
    )

    # 7) Initialize CosineWarmupScheduler
    # Build loaders first to calculate total steps
    import dataclasses
    config_dict = dataclasses.asdict(training_config)
    config_dict['context_length'] = model_config.context_length
    config_dict['device'] = device.type
    config_dict['data_path'] = raw_path
    config_dict['train_bin'] = train_bin
    config_dict['val_bin'] = val_bin
    
    train_loader, val_loader = get_dataloaders(config_dict, tokenizer)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * training_config.max_epochs
    
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=min(training_config.warmup_steps, total_steps // 10),
        max_steps=total_steps,
        max_lr=training_config.learning_rate,
        min_lr=training_config.learning_rate * 0.1
    )

    # 8) Load checkpoint if --resume is given
    if args.resume:
        trainer.load_checkpoint(model, optimizer, args.resume)
        # Pass resume to config_dict so trainer.train starts from the correct step/history
        config_dict['resume'] = args.resume

    # Pass pre-built objects and configure training loop
    config_dict['optimizer'] = optimizer
    config_dict['scheduler'] = scheduler
    config_dict['train_loader'] = train_loader
    config_dict['val_loader'] = val_loader
    config_dict['max_iters'] = total_steps
    
    # 9) Call trainer.train()
    t_start = time.time()
    history = trainer.train(model, tokenizer, config_dict)
    total_time = time.time() - t_start

    # 10) Save final model weights to checkpoints/nanosage_final.pt
    final_path = os.path.join(training_config.checkpoint_dir, "nanosage_final.pt")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model.config,
    }, final_path)
    print(f"Final model weights saved to: {final_path}")

    # 11) Print training summary
    final_train_loss = history["train_loss"][-1] if history["train_loss"] else float("nan")
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else float("nan")
    final_perplexity = history["perplexity"][-1] if history["perplexity"] else float("nan")

    # Format time
    h, rem = divmod(int(total_time), 3600)
    m, s = divmod(rem, 60)
    time_str = f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")

    print("\n" + "="*50)
    print("               TRAINING SUMMARY")
    print("="*50)
    print(f"  • Total Time       : {time_str} ({total_time:.2f}s)")
    print(f"  • Final Train Loss : {final_train_loss:.4f}")
    print(f"  • Final Val Loss   : {final_val_loss:.4f}")
    print(f"  • Final Perplexity : {final_perplexity:.2f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
