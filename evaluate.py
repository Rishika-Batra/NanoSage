"""
evaluate.py — Evaluation script for NanoSage.

Computes:
  • Train loss & perplexity
  • Validation loss & perplexity
  • Token-level accuracy
  • Bits-per-character (BPC)

Usage
-----
  python evaluate.py --checkpoint nanosage/checkpoints/best_model.pt
  python evaluate.py --checkpoint nanosage/checkpoints/best_model.pt \\
                     --data_path nanosage/data/raw/tinystories.txt \\
                     --split val --max_batches 200
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nanosage.tokenizer.bpe import BPETokenizer
from nanosage.model.transformer import NanoSageLM
from nanosage.training.dataset import TextDataset, TRAIN_BIN_PATH, VAL_BIN_PATH


# ── helpers ──────────────────────────────────────────────────────────────────

def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _fmt_time(secs: float) -> str:
    secs = int(secs)
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")


def _bar(value: float, max_val: float, width: int = 20) -> str:
    filled = int(width * value / max(max_val, 1e-9))
    return "█" * filled + "░" * (width - filled)


# ── core evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_loader(
    model: NanoSageLM,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    split_name: str = "split",
) -> dict:
    """
    Evaluate model over *loader* and return a metrics dict.

    Returns
    -------
    {
        "loss":        float   cross-entropy loss
        "perplexity":  float
        "accuracy":    float   token-level accuracy (ignores -1 targets)
        "bpc":         float   bits-per-character (using log2)
        "num_tokens":  int     total non-masked tokens evaluated
        "num_batches": int
    }
    """
    model.eval()

    total_loss    = 0.0
    correct_toks  = 0
    total_toks    = 0
    num_batches   = 0

    try:
        from tqdm import tqdm
        pbar = tqdm(loader, desc=f"  Evaluating [{split_name}]",
                    unit="batch", dynamic_ncols=True, colour="green",
                    total=max_batches)
    except ImportError:
        pbar = loader

    for batch_idx, (x, y) in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)
        logits, _ = model(x, y)

        # Re-compute loss so we can also extract per-token accuracy
        # (y uses -1 as ignore mask, matching ignore_index=-1 in model.forward)
        logits_flat = logits.view(-1, logits.size(-1))   # (B*T, vocab)
        y_flat      = y.view(-1)                          # (B*T,)

        loss = F.cross_entropy(logits_flat, y_flat, ignore_index=-1, reduction="sum")
        mask = (y_flat != -1)
        n    = mask.sum().item()

        if n > 0:
            total_loss   += loss.item()
            total_toks   += n
            # Token accuracy
            preds   = logits_flat.argmax(dim=-1)
            correct_toks += (preds[mask] == y_flat[mask]).sum().item()

        num_batches += 1

    avg_loss   = total_loss / max(total_toks, 1)
    perplexity = math.exp(min(avg_loss, 20.0))
    accuracy   = correct_toks / max(total_toks, 1)
    bpc        = avg_loss / math.log(2)           # nats → bits

    model.train()
    return {
        "loss":        avg_loss,
        "perplexity":  perplexity,
        "accuracy":    accuracy,
        "bpc":         bpc,
        "num_tokens":  total_toks,
        "num_batches": num_batches,
    }


# ── pretty printing ───────────────────────────────────────────────────────────

def print_metrics(name: str, m: dict, elapsed: float) -> None:
    W = 56
    print(f"\n  ┌{'─'*(W-2)}┐")
    print(f"  │{'  ' + name:^{W-2}}│")
    print(f"  ├{'─'*(W-2)}┤")
    print(f"  │  {'Loss':<20}  {m['loss']:>10.4f}                │")
    print(f"  │  {'Perplexity':<20}  {m['perplexity']:>10.2f}                │")
    print(f"  │  {'Token Accuracy':<20}  {m['accuracy']*100:>9.2f} %             │")
    print(f"  │  {'Bits-per-char':<20}  {m['bpc']:>10.4f}                │")
    print(f"  │  {'Tokens evaluated':<20}  {m['num_tokens']:>10,}                │")
    print(f"  │  {'Batches':<20}  {m['num_batches']:>10,}                │")
    print(f"  │  {'Eval time':<20}  {_fmt_time(elapsed):>10}                │")
    print(f"  └{'─'*(W-2)}┘")


def print_comparison(train_m: dict | None, val_m: dict) -> None:
    if train_m is None:
        return
    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │            TRAIN vs VAL COMPARISON             │")
    print(f"  ├──────────────┬──────────────┬──────────────────┤")
    print(f"  │ Metric       │     Train    │     Val          │")
    print(f"  ├──────────────┼──────────────┼──────────────────┤")
    print(f"  │ Loss         │ {train_m['loss']:>10.4f}   │ {val_m['loss']:>10.4f}       │")
    print(f"  │ Perplexity   │ {train_m['perplexity']:>10.2f}   │ {val_m['perplexity']:>10.2f}       │")
    print(f"  │ Accuracy     │ {train_m['accuracy']*100:>9.2f} %  │ {val_m['accuracy']*100:>9.2f} %      │")
    print(f"  │ BPC          │ {train_m['bpc']:>10.4f}   │ {val_m['bpc']:>10.4f}       │")
    gap = val_m["perplexity"] - train_m["perplexity"]
    status = "✅ tight" if abs(gap) < 5 else ("⚠️  overfit?" if gap > 0 else "✅ good")
    print(f"  ├──────────────┴──────────────┴──────────────────┤")
    print(f"  │  PPL gap (val - train): {gap:+.2f}  {status:<18} │")
    print(f"  └─────────────────────────────────────────────────┘")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a NanoSage checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--tokenizer_path", type=str,
                        default="nanosage/checkpoints/tokenizer.json",
                        help="Path to BPE tokenizer state")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Raw .txt to evaluate on (default: use cached .npy splits)")
    parser.add_argument("--split", type=str, default="both",
                        choices=["train", "val", "both"],
                        help="Which split(s) to evaluate")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Cap evaluation at N batches (useful for quick checks)")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = _auto_device() if args.device == "auto" else torch.device(args.device)

    # ── Banner ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("          NanoSage — Model Evaluation")
    print("═" * 60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {device}")
    print(f"  Split      : {args.split}")
    if args.max_batches:
        print(f"  Max batches: {args.max_batches}")
    print("═" * 60)

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"\n  Loading checkpoint…")
    ckpt   = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["model_config"]

    model = NanoSageLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model parameters : {n_params:.2f}M")
    print(f"  Vocab size       : {config.vocab_size}")
    print(f"  Context length   : {config.block_size}")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = BPETokenizer()
    if os.path.exists(args.tokenizer_path):
        tokenizer.load(args.tokenizer_path)
        print(f"  Tokenizer        : {args.tokenizer_path}")

    # ── Build datasets ────────────────────────────────────────────────────────
    context_length = config.block_size

    def _make_loader(is_val: bool) -> DataLoader:
        if args.data_path:
            ds = TextDataset(
                args.data_path, tokenizer=tokenizer,
                block_size=context_length,
                val_split=0.1, is_val=is_val
            )
        else:
            bin_path = VAL_BIN_PATH if is_val else TRAIN_BIN_PATH
            if not os.path.exists(bin_path):
                raise FileNotFoundError(
                    f"Pre-tokenized file not found: {bin_path}\n"
                    "Run train.py first, or pass --data_path <file.txt>"
                )
            ds = TextDataset(bin_path, context_length=context_length)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    train_metrics = None
    val_metrics   = None

    print()
    if args.split in ("train", "both"):
        loader = _make_loader(is_val=False)
        t0 = time.perf_counter()
        train_metrics = evaluate_loader(model, loader, device,
                                        max_batches=args.max_batches,
                                        split_name="Train")
        print_metrics("Training Split", train_metrics, time.perf_counter() - t0)

    if args.split in ("val", "both"):
        loader = _make_loader(is_val=True)
        t0 = time.perf_counter()
        val_metrics = evaluate_loader(model, loader, device,
                                      max_batches=args.max_batches,
                                      split_name="Val")
        print_metrics("Validation Split", val_metrics, time.perf_counter() - t0)

    if train_metrics and val_metrics:
        print_comparison(train_metrics, val_metrics)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
