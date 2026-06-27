import argparse
import os
import sys
import time
import math
import torch
from torch.utils.data import DataLoader, Dataset

from nanosage.tokenizer.bpe import BPETokenizer
from nanosage.model.config import ModelConfig
from nanosage.model.transformer import NanoSageLM
from nanosage.inference.generate import generate
import nanosage.training.trainer as trainer
from nanosage.training.scheduler import CosineWarmupScheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Instruction Finetuning for NanoSage")
    parser.add_argument("--checkpoint", type=str, default="nanosage/checkpoints/nanosage_final.pt", help="Path to pretrained model checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Finetuning epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Finetuning learning rate")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/mps/cpu")
    return parser.parse_args()

def format_alpaca_example(example):
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output_text = example.get("output", "").strip()
    
    prompt = "### Instruction:\n" + instruction + "\n\n"
    if input_text:
        prompt += "### Input:\n" + input_text + "\n\n"
    prompt += "### Response:\n"
    
    response = output_text + "<|endoftext|>"
    return prompt, response

class InstructionDataset(Dataset):
    def __init__(self, examples, tokenizer, block_size=512):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_token_id = tokenizer.special_tokens.get("<|endoftext|>", 256)
        self.samples = []
        
        for ex in examples:
            prompt, response = format_alpaca_example(ex)
            prompt_ids = tokenizer.encode(prompt)
            response_ids = tokenizer.encode(response)
            
            if not prompt_ids or not response_ids:
                continue
                
            full_ids = prompt_ids + response_ids
            prompt_len = len(prompt_ids)
            total_len = len(full_ids)
            
            if total_len > block_size:
                full_ids = full_ids[:block_size + 1]
                total_len = len(full_ids)
                prompt_len = min(prompt_len, block_size)
            
            N = total_len
            if N <= block_size:
                input_ids = full_ids[:-1] + [self.pad_token_id] * (block_size - N + 1)
                target_ids = [-1] * (prompt_len - 1) + full_ids[prompt_len:] + [-1] * (block_size - N + 1)
            else:
                input_ids = full_ids[:-1]
                target_ids = [-1] * (prompt_len - 1) + full_ids[prompt_len:]
                
            self.samples.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_ids, dtype=torch.long)
            ))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def generate_sample(model, tokenizer, device):
    prompt = "### Instruction:\nWhat is artificial intelligence?\n\n### Response:\n"
    prompt_ids = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    eos_id = tokenizer.special_tokens.get("<|endoftext|>")
    output_tensor = generate(
        model=model,
        idx=prompt_tensor,
        max_new_tokens=64,
        temperature=0.7,
        top_k=50,
        eos_token_id=eos_id
    )
    generated_tokens = output_tensor[0, len(prompt_ids):].tolist()
    response_text = tokenizer.decode(generated_tokens)
    return response_text.replace("<|endoftext|>", "").strip()

def main():
    args = parse_args()
    
    # Auto-detect device
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
    
    # 1) Load pretrained weights
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        # Fallback to best_model.pt
        checkpoint_path = "nanosage/checkpoints/best_model.pt"
        
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {args.checkpoint} or fallback best_model.pt.")
        
    print(f"Loading pretrained weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['model_config']
    
    model = NanoSageLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # 2) Load BPE Tokenizer
    tokenizer = BPETokenizer()
    tokenizer_path = "nanosage/checkpoints/tokenizer.json"
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
        print(f"Loaded tokenizer from {tokenizer_path}")
    else:
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
    # 3) Download instruction dataset
    from datasets import load_dataset
    print("Downloading 'yahma/alpaca-cleaned' from HuggingFace...")
    ds = load_dataset("yahma/alpaca-cleaned", split="train", trust_remote_code=True)
    examples = [ds[i] for i in range(min(5000, len(ds)))]
    print(f"Downloaded {len(examples)} examples.")
    
    # 4) Prepare dataset and loader
    print("Tokenizing alpaca examples into InstructionDataset...")
    train_dataset = InstructionDataset(examples, tokenizer, block_size=config.block_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Print shapes of first sample for validation
    first_x, first_y = train_dataset[0]
    ignored = (first_y == -1).sum().item()
    active = (first_y != -1).sum().item()
    print(f"First dataset sample:")
    print(f"  Input tokens shape  : {first_x.shape}")
    print(f"  Target tokens shape : {first_y.shape}")
    print(f"  Ignored prompt tokens: {ignored}")
    print(f"  Active loss tokens   : {active}")
    
    # 5) Optimizer and Scheduler
    optimizer = model.configure_optimizers(
        weight_decay=0.01,
        learning_rate=args.lr,
        betas=(0.9, 0.95),
        device_type=device.type
    )
    
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=100,
        max_steps=total_steps,
        max_lr=args.lr,
        min_lr=args.lr * 0.1
    )
    
    # 6) Finetune loop
    print(f"Starting instruction finetuning...")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Total steps  : {total_steps}")
    
    step_offset = 0
    t_start = time.time()
    
    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f" Epoch {epoch + 1} / {args.epochs}")
        print(f"{'='*50}")
        
        avg_loss, steps_taken = trainer.train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config={
                'grad_clip': 1.0,
                'grad_accumulation_steps': 1,
                'log_interval': 20
            },
            device=device,
            step_offset=step_offset
        )
        step_offset += steps_taken
        print(f"\nEpoch {epoch + 1} Complete. Average loss: {avg_loss:.4f}")
        
        # Generate sample response to track progress
        print("\nProgress generation sample:")
        sample_response = generate_sample(model, tokenizer, device)
        print(f"  [Q]: What is artificial intelligence?")
        print(f"  [A]: {sample_response}\n")
        
    total_time = time.time() - t_start
    print(f"Finetuning completed in {total_time:.2f} seconds.")
    
    # 7) Save finetuned weights
    instruct_path = "nanosage/checkpoints/nanosage_instruct.pt"
    os.makedirs(os.path.dirname(instruct_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model.config,
    }, instruct_path)
    print(f"Finetuned model saved to: {instruct_path}")

if __name__ == "__main__":
    main()
