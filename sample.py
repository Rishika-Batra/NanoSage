import argparse
import os
import torch

from nanosage.tokenizer.bpe import BPETokenizer
from nanosage.model.transformer import NanoSageLM
from nanosage.inference.generate import generate

def parse_args():
    parser = argparse.ArgumentParser(description="Generate sample completions from a trained NanoSage model")
    
    # Path arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--tokenizer_path", type=str, default="nanosage/checkpoints/tokenizer.json", help="Path to BPE tokenizer state")
    
    # Generation settings
    parser.add_argument("--prompt", type=str, default="\n", help="Prompt to start generation from")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of tokens to generate per response")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature (0.0 = greedy, higher = more random)")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling threshold")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p (nucleus) sampling threshold")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    
    parser.add_argument("--device", type=str, default="auto", help="Execution device (cuda/mps/cpu)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1) Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
        
    print(f"Using device: {device}")
        
    # 2) Load BPE Tokenizer
    tokenizer = BPETokenizer()
    if os.path.exists(args.tokenizer_path):
        tokenizer.load(args.tokenizer_path)
        print(f"Loaded tokenizer from: {args.tokenizer_path}")
    else:
        raise FileNotFoundError(f"Tokenizer file not found at: {args.tokenizer_path}. Run training or supply correct path.")

    # 3) Load model checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint['model_config']
    
    model = NanoSageLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded. Vocabulary size: {config.vocab_size}, Context size: {config.block_size}")

    eos_id = tokenizer.special_tokens.get("<|endoftext|>")
    
    # 4) Generate samples
    prompt_tokens = tokenizer.encode(args.prompt)
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    print(f"\nGenerating {args.num_samples} samples with prompt: {repr(args.prompt)}")
    print(f"Sampling settings: temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print("=" * 60)
    
    for i in range(args.num_samples):
        print(f"\n--- Sample {i + 1} ---")
        output_tensor = generate(
            model=model,
            idx=prompt_tensor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=eos_id
        )
        
        # Decode the complete sequence (prompt + completion)
        generated_tokens = output_tensor[0].tolist()
        generated_text = tokenizer.decode(generated_tokens)
        print(generated_text)
        
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
