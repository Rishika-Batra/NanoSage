#!/usr/bin/env python3
"""
tests/test_nanosage.py — Integration and Unit Tests for NanoSage.

Tests tokenizer roundtrips, model dimensions and gradients, dataset and
dataloader batch formatting, and text generation decoding strategies.
"""

import os
import sys
import tempfile
import torch
from torch.utils.data import DataLoader

# Add repository root to system path for clean imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nanosage.tokenizer.bpe import BPETokenizer
from nanosage.model.config import ModelConfig
from nanosage.model.transformer import NanoSageLM
from nanosage.training.dataset import TextDataset
from nanosage.inference.generate import greedy_decode, sample_decode, GenerationConfig

# ANSI color codes for execution feedback
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_GREEN = "\033[32m"
C_RED = "\033[31m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print standard colorized test outcome status."""
    status = f"{C_GREEN}{C_BOLD}PASSED{C_RESET}" if passed else f"{C_RED}{C_BOLD}FAILED{C_RESET}"
    print(f"[{status}] {C_BOLD}{test_name}{C_RESET}")
    if details:
        # Indent details for cleaner alignment
        indented_details = "\n".join("         " + line for line in details.strip().split("\n"))
        print(indented_details)
    print()


def test_bpe_tokenizer() -> bool:
    """Tests tokenizer training, encoding, decoding roundtrips, and file save/load."""
    print(f"{C_BLUE}Running BPETokenizer test...{C_RESET}")
    try:
        tokenizer = BPETokenizer()
        sample_text = (
            "Once upon a time in a faraway land, there was a little robot "
            "who loved to code transformer models in PyTorch from scratch."
        )

        # Train with a tiny target vocabulary size
        tokenizer.train(sample_text, vocab_size=265, verbose=False)

        # 1. Roundtrip test
        encoded = tokenizer.encode(sample_text)
        decoded = tokenizer.decode(encoded)
        roundtrip_passed = (decoded == sample_text)

        # 2. Save and Load verification
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_json = os.path.join(tmpdir, "tokenizer.json")
            tokenizer.save(temp_json)

            loaded_tokenizer = BPETokenizer()
            loaded_tokenizer.load(temp_json)
            loaded_encoded = loaded_tokenizer.encode(sample_text)
            loaded_decoded = loaded_tokenizer.decode(loaded_encoded)

            save_load_passed = (
                loaded_encoded == encoded and
                loaded_decoded == sample_text and
                len(loaded_tokenizer.vocab) == len(tokenizer.vocab)
            )

        passed = roundtrip_passed and save_load_passed
        details = (
            f"Roundtrip check (Decoded == Original) : {roundtrip_passed}\n"
            f"Save/Load check (Loaded == Original)   : {save_load_passed}\n"
            f"Final Vocab Size                       : {len(tokenizer.vocab)} tokens"
        )
        print_result("BPETokenizer Validation", passed, details)
        return passed
    except Exception as e:
        print_result("BPETokenizer Validation", False, f"Exception occurred: {e}")
        return False


def test_model() -> bool:
    """Tests model initialization parameter count, forward pass shapes, and backprop gradient flow."""
    print(f"{C_BLUE}Running NanoSageLM Model test...{C_RESET}")
    try:
        # Standard configuration chosen to result in ~12.4M parameters
        config = ModelConfig(
            vocab_size=8000,
            embedding_dim=320,
            num_layers=6,
            num_heads=8,
            context_length=512,
            ffn_hidden_dim=1280,
            use_rope=True,
            use_rmsnorm=True
        )
        model = NanoSageLM(config)

        # 1. Parameter count check (should be ~10-15M)
        param_count = sum(p.numel() for p in model.parameters())
        param_count_m = param_count / 1e6
        param_count_passed = (10.0 <= param_count_m <= 15.0)

        # 2. Forward pass shape check (pass targets to get full logits back)
        batch_size = 2
        seq_len = 64
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        logits, _ = model(x, targets=targets)
        expected_shape = (batch_size, seq_len, config.vocab_size)
        shape_passed = (logits.shape == expected_shape)

        # 3. Gradient flow check
        model.train()
        loss = logits.sum()
        loss.backward()

        grad_passed = True
        none_grads = []
        zero_grads = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    grad_passed = False
                    none_grads.append(name)
                elif torch.all(param.grad == 0.0):
                    grad_passed = False
                    zero_grads.append(name)

        passed = param_count_passed and shape_passed and grad_passed
        details = (
            f"Param Count  : {param_count_m:.2f}M (Expected: 10-15M) -> {'OK' if param_count_passed else 'OUT-OF-BOUNDS'}\n"
            f"Output Shape : {logits.shape} (Expected: {expected_shape}) -> {'OK' if shape_passed else 'INCORRECT'}\n"
            f"Grad Flow    : {'OK' if grad_passed else f'FAILED (None grads: {len(none_grads)}, Zero grads: {len(zero_grads)})'}"
        )
        if none_grads:
            details += f"\n  - Parameters with missing gradients: {none_grads[:5]}"
        if zero_grads:
            details += f"\n  - Parameters with zero gradients: {zero_grads[:5]}"

        print_result("NanoSageLM Model Validation", passed, details)
        return passed
    except Exception as e:
        print_result("NanoSageLM Model Validation", False, f"Exception occurred: {e}")
        return False


def test_dataset() -> bool:
    """Tests TextDataset and DataLoader shapes and verifies there are no NaNs in batches."""
    print(f"{C_BLUE}Running Dataset/DataLoader test...{C_RESET}")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "temp_corpus.txt")
            with open(file_path, "w") as f:
                # Write a sufficiently long text corpus for windowed slices
                f.write("Hello world, this is a clean test corpus. Code is running fine. " * 50)

            tokenizer = BPETokenizer()
            tokenizer.train("Hello world, this is a clean test corpus. Code is running fine. ", vocab_size=260)

            # Instantiating dataset using context_length and tokenizer parameters
            dataset = TextDataset(file_path, context_length=16, tokenizer=tokenizer)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

            # Retrieve one batch
            x, y = next(iter(dataloader))

            # Verify dimensions
            shape_passed = (x.shape == (4, 16)) and (y.shape == (4, 16))
            # Verify no NaNs are present in the tensors
            nan_passed = (not torch.isnan(x).any()) and (not torch.isnan(y).any())

        passed = shape_passed and nan_passed
        details = (
            f"Input Batch Shape  : {x.shape} (Expected: (4, 16))\n"
            f"Target Batch Shape : {y.shape} (Expected: (4, 16))\n"
            f"NaN values check   : No NaNs found -> {nan_passed}"
        )
        print_result("Dataset and DataLoader Validation", passed, details)
        return passed
    except Exception as e:
        print_result("Dataset and DataLoader Validation", False, f"Exception occurred: {e}")
        return False


def test_generation() -> bool:
    """Tests that greedy decoding generates strings and stochastic sampling generates diverse texts."""
    print(f"{C_BLUE}Running Text Generation test...{C_RESET}")
    try:
        tokenizer = BPETokenizer()
        # Train BPE tokenizer briefly to register base characters
        tokenizer.train("Once upon a time there was a tiny robot.", vocab_size=260)
        
        # Match model's vocab_size to tokenizer's vocab_size to prevent out-of-vocab token index lookup errors
        vocab_size = len(tokenizer.vocab)

        # Create a tiny configuration for extremely fast compilation
        config = ModelConfig(
            vocab_size=vocab_size,
            embedding_dim=128,
            num_layers=2,
            num_heads=2,
            context_length=64
        )
        model = NanoSageLM(config)

        prompt = "Once upon"

        # 1. Greedy decoding output verification
        greedy_output = greedy_decode(model, tokenizer, prompt, max_tokens=10)
        greedy_passed = isinstance(greedy_output, str) and len(greedy_output) > 0

        # 2. Stochastic sampling diversity verification
        # High temperature increases output diversity to ensure distinct sequence outcomes
        gen_config = GenerationConfig(
            max_new_tokens=10,
            temperature=2.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0
        )

        stochastic_samples = []
        for _ in range(5):
            res = sample_decode(model, tokenizer, prompt, config=gen_config)
            stochastic_samples.append(res)

        unique_samples = set(stochastic_samples)
        stochastic_passed = (len(unique_samples) > 1)

        passed = greedy_passed and stochastic_passed
        details = (
            f"Greedy decode output length : {len(greedy_output)} characters (Passed: {greedy_passed})\n"
            f"Unique stochastic outputs    : {len(unique_samples)} distinct strings out of 5 runs (Passed: {stochastic_passed})\n"
            f"Sample outputs               : {list(unique_samples)[:2]}"
        )
        print_result("Text Generation Decoding Validation", passed, details)
        return passed
    except Exception as e:
        print_result("Text Generation Decoding Validation", False, f"Exception occurred: {e}")
        return False


def main():
    print("==================================================")
    print("           NANOSAGE INTEGRATION TEST SUITE        ")
    print("==================================================")
    print()

    t1 = test_bpe_tokenizer()
    t2 = test_model()
    t3 = test_dataset()
    t4 = test_generation()

    print("==================================================")
    if all([t1, t2, t3, t4]):
        print(f"{C_GREEN}{C_BOLD}ALL TESTS PASSED! 🎉{C_RESET}")
        sys.exit(0)
    else:
        print(f"{C_RED}{C_BOLD}SOME TESTS FAILED! ❌{C_RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
