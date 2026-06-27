#!/usr/bin/env python3
"""
chat.py — Interactive Command Line Chat Interface for NanoSage.

Provides a colorful terminal interface to chat with a fine-tuned instruction model.
Supports multi-turn dialogue context up to 3 turns, generation parameters,
and custom command shortcuts.
"""

import argparse
import os
import sys
import time
import torch

from nanosage.tokenizer.bpe import BPETokenizer
from nanosage.model.transformer import NanoSageLM
from nanosage.inference.generate import sample_decode, GenerationConfig
from nanosage.inference.chat import NanoSageChat

# ANSI Color codes for a premium, styled CLI experience
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_BLUE = "\033[34m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_MAGENTA = "\033[35m"
C_YELLOW = "\033[33m"
C_GRAY = "\033[90m"
C_RED = "\033[31m"


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive command-line chat for NanoSage")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint. Defaults to checking checkpoints/nanosage_instruct.pt"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="nanosage/checkpoints/tokenizer.json",
        help="Path to tokenizer.json file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on (auto/cuda/mps/cpu)"
    )
    return parser.parse_args()


def print_welcome_banner():
    banner = f"""{C_CYAN}{C_BOLD}
   ╔══════════════════════════════╗
   ║     🧠 NanoSage Chat v1.0    ║
   ║   Your tiny AI assistant     ║
   ╚══════════════════════════════╝{C_RESET}
    """
    print(banner)
    print(f"{C_GRAY}Interactive instruction session started.{C_RESET}")
    print(f"{C_GRAY}Supported slash commands:{C_RESET}")
    print(f"  {C_CYAN}/clear{C_RESET}   → Clear conversation history")
    print(f"  {C_CYAN}/config{C_RESET}  → Show current generation parameters")
    print(f"  {C_CYAN}/temp N{C_RESET}  → Set temperature to N (e.g. /temp 0.7)")
    print(f"  {C_CYAN}/quit{C_RESET}    → Exit chat session (or type /exit)")
    print(f"{C_GRAY}───────────────────────────────────────────────────{C_RESET}\n")


def get_default_checkpoint():
    """Searches for model checkpoints in standard paths in descending priority."""
    paths = [
        "checkpoints/nanosage_instruct.pt",
        "nanosage/checkpoints/nanosage_instruct.pt",
        "nanosage/checkpoints/best_model.pt",
        "nanosage/checkpoints/latest_model.pt"
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def main():
    args = parse_args()

    # 1. Device Selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # 2. Checkpoint Selection & Loading
    checkpoint_path = args.checkpoint or get_default_checkpoint()
    if not checkpoint_path:
        print(f"{C_RED}{C_BOLD}Error: No model checkpoint found!{C_RESET}")
        print("Please train a model or place a checkpoint at 'nanosage/checkpoints/nanosage_instruct.pt'.")
        sys.exit(1)

    print(f"{C_GRAY}Loading checkpoint from: {C_BOLD}{checkpoint_path}{C_RESET} on {C_BOLD}{device}{C_RESET}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"{C_RED}Failed to load checkpoint: {e}{C_RESET}")
        sys.exit(1)

    # 3. Model Initialization
    model_cfg = checkpoint["model_config"]
    model = NanoSageLM(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    print(f"{C_GREEN}Model successfully loaded! (Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M){C_RESET}")

    # 4. Tokenizer Loading
    tokenizer_path = args.tokenizer
    if not os.path.exists(tokenizer_path):
        # Fallback check
        tokenizer_path = "checkpoints/tokenizer.json"
        if not os.path.exists(tokenizer_path):
            print(f"{C_RED}Error: Tokenizer file not found at {args.tokenizer} or checkpoints/tokenizer.json{C_RESET}")
            sys.exit(1)

    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    print(f"{C_GREEN}Tokenizer loaded from {tokenizer_path} (Vocab size: {len(tokenizer.vocab)}).{C_RESET}\n")

    # 5. Dialogue & Generation Settings Setup
    chat_manager = NanoSageChat(max_history=3)
    
    # Initialize default sampling config
    eos_id = tokenizer.special_tokens.get("<|endoftext|>")
    gen_config = GenerationConfig(
        max_new_tokens=128,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=eos_id
    )

    print_welcome_banner()

    # 6. Interaction Loop
    while True:
        try:
            # Styled User prompt
            user_input = input(f"{C_BLUE}{C_BOLD}User › {C_RESET}").strip()
            if not user_input:
                continue

            # Command Handling
            if user_input.startswith("/"):
                cmd_parts = user_input.split()
                cmd = cmd_parts[0].lower()

                if cmd in ("/quit", "/exit"):
                    print(f"\n{C_MAGENTA}Goodbye! Thanks for chatting with NanoSage.{C_RESET}\n")
                    break

                elif cmd == "/clear":
                    chat_manager.clear()
                    print(f"{C_GREEN}Conversation history cleared (last 3 turns reset).{C_RESET}\n")
                    continue

                elif cmd == "/config":
                    print(f"\n{C_CYAN}{C_BOLD}Current Generation Configuration:{C_RESET}")
                    print(f"  • {C_BOLD}max_new_tokens{C_RESET}     : {gen_config.max_new_tokens}")
                    print(f"  • {C_BOLD}temperature{C_RESET}        : {gen_config.temperature:.2f}")
                    print(f"  • {C_BOLD}top_k{C_RESET}              : {gen_config.top_k}")
                    print(f"  • {C_BOLD}top_p{C_RESET}              : {gen_config.top_p}")
                    print(f"  • {C_BOLD}repetition_penalty{C_RESET} : {gen_config.repetition_penalty}")
                    print()
                    continue

                elif cmd == "/temp":
                    if len(cmd_parts) < 2:
                        print(f"{C_YELLOW}Usage: /temp <value> (e.g. /temp 0.7){C_RESET}\n")
                        continue
                    try:
                        new_temp = float(cmd_parts[1])
                        if new_temp < 0.0:
                            print(f"{C_RED}Error: Temperature must be non-negative.{C_RESET}\n")
                            continue
                        gen_config.temperature = new_temp
                        print(f"{C_GREEN}Temperature successfully set to {new_temp:.2f}{C_RESET}\n")
                    except ValueError:
                        print(f"{C_RED}Error: Temperature must be a numeric value.{C_RESET}\n")
                    continue

                else:
                    print(f"{C_YELLOW}Unknown command: {cmd}. Available: /clear, /config, /temp, /quit{C_RESET}\n")
                    continue

            # Format the instruction with history
            prompt = chat_manager.get_formatted_prompt(user_input)

            # Generate response & measure latency
            t0 = time.perf_counter()
            response_text = sample_decode(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                config=gen_config,
                device=device
            )
            elapsed = time.perf_counter() - t0

            # Decode / measure tokens
            response_tokens = tokenizer.encode(response_text)
            num_tokens = len(response_tokens)
            tokens_per_sec = num_tokens / max(elapsed, 1e-6)

            # Print Response Header
            print(f"{C_MAGENTA}{C_BOLD}Assistant › {C_RESET}", end="", flush=True)

            # Typing effect: print response character-by-character with a small delay
            for char in response_text:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(0.015)
            print()

            # Display performance metadata
            print(f"{C_GRAY}[Generated {num_tokens} tokens in {elapsed:.2f}s | Speed: {tokens_per_sec:.2f} tok/sec]{C_RESET}\n")

            # Update conversation history
            chat_manager.add_turn(user_input, response_text)

        except KeyboardInterrupt:
            print(f"\n{C_YELLOW}Session interrupted by user. Exiting...{C_RESET}\n")
            break
        except Exception as e:
            print(f"\n{C_RED}Error occurred during generation: {e}{C_RESET}\n")


if __name__ == "__main__":
    main()
