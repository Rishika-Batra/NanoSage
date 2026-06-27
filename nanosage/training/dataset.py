"""
nanosage/training/dataset.py

Utilities for downloading, preparing, and serving the TinyStories pre-training
dataset.

Public API
----------
download_pretrain_data()            -- fetch & clean from HuggingFace, save .txt
prepare_pretrain_data(tokenizer)    -- tokenize & split → numpy arrays on disk
TextDataset                         -- torch Dataset over a token array
get_dataloaders(config, tokenizer)  -- returns (train_loader, val_loader)
"""

import os
import re
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_DIR      = os.path.join("nanosage", "data")
_RAW_DIR       = os.path.join(_DATA_DIR, "raw")
_PROCESSED_DIR = os.path.join(_DATA_DIR, "processed")

RAW_TXT_PATH       = os.path.join(_RAW_DIR,       "tinystories.txt")
TRAIN_BIN_PATH     = os.path.join(_PROCESSED_DIR, "train_tokens.npy")
VAL_BIN_PATH       = os.path.join(_PROCESSED_DIR, "val_tokens.npy")

# ---------------------------------------------------------------------------
# 1.  download_pretrain_data
# ---------------------------------------------------------------------------

def _clean_story(text: str) -> str:
    """
    Light cleaning applied to every story:
      - Collapse runs of whitespace / blank lines to a single newline/space.
      - Normalise "smart" quotes to plain ASCII equivalents.
      - Strip leading / trailing whitespace.
    """
    # Normalise unicode quotes → plain ASCII
    text = text.replace("\u2018", "'").replace("\u2019", "'")   # ' '
    text = text.replace("\u201c", '"').replace("\u201d", '"')   # " "
    text = text.replace("\u2013", "-").replace("\u2014", "-")   # en / em dash

    # Collapse multiple blank lines → single newline
    text = re.sub(r"\n{2,}", "\n", text)

    # Collapse multiple spaces / tabs → single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def download_pretrain_data(
    num_stories: int = 50_000,
    output_path: str = RAW_TXT_PATH,
    hf_dataset: str = "roneneldan/TinyStories",
) -> str:
    """
    Download the first *num_stories* stories from ``roneneldan/TinyStories``
    on HuggingFace, clean them, join with ``<|endoftext|>`` separators, and
    write the result to *output_path*.

    Returns the output path.

    Parameters
    ----------
    num_stories : int
        How many stories to take from the train split (default 50 000).
    output_path : str
        Where to save the combined .txt file.
    hf_dataset : str
        HuggingFace dataset identifier.
    """
    try:
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError(
            "The `datasets` and `tqdm` packages are required. "
            "Install them with:  pip install datasets tqdm"
        ) from exc

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Downloading '{hf_dataset}' from HuggingFace …")
    ds = load_dataset(hf_dataset, split="train", trust_remote_code=True)

    # Cap at available number of examples
    total_available = len(ds)
    num_stories = min(num_stories, total_available)
    print(f"Using {num_stories:,} stories out of {total_available:,} available.")

    stories: list[str] = []
    for i in tqdm(range(num_stories), desc="Cleaning stories", unit="story"):
        raw = ds[i].get("text", "")
        cleaned = _clean_story(raw)
        if cleaned:                     # skip empty stories
            stories.append(cleaned)

    separator = "\n<|endoftext|>\n"
    combined  = separator.join(stories)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined)

    # ---- Statistics --------------------------------------------------------
    total_chars  = len(combined)
    total_words  = len(combined.split())
    total_stories = len(stories)

    print("\n── Dataset Statistics ──────────────────────────────────────")
    print(f"  Stories saved : {total_stories:>12,}")
    print(f"  Total words   : {total_words:>12,}")
    print(f"  Total chars   : {total_chars:>12,}")
    print(f"  Saved to      : {output_path}")
    print("────────────────────────────────────────────────────────────\n")

    return output_path


# ---------------------------------------------------------------------------
# 2.  prepare_pretrain_data
# ---------------------------------------------------------------------------

def prepare_pretrain_data(
    tokenizer,
    raw_path: str  = RAW_TXT_PATH,
    train_out: str = TRAIN_BIN_PATH,
    val_out:   str = VAL_BIN_PATH,
    val_split: float = 0.10,
) -> tuple[str, str]:
    """
    Tokenize the raw text and persist token-id arrays as numpy uint16 files.

    Parameters
    ----------
    tokenizer  : BPETokenizer (or any object with an ``encode`` method)
    raw_path   : path to the raw .txt file produced by ``download_pretrain_data``
    train_out  : destination path for the training token array (.npy)
    val_out    : destination path for the validation token array (.npy)
    val_split  : fraction reserved for validation (default 10 %)

    Returns
    -------
    (train_out, val_out) — the two file paths written to disk.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # graceful fallback

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw data not found at '{raw_path}'. "
            "Run download_pretrain_data() first."
        )

    os.makedirs(_PROCESSED_DIR, exist_ok=True)

    print(f"Reading raw text from: {raw_path}")
    with open(raw_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Tokenizing … (this may take a minute for large files)")
    ids = tokenizer.encode(text)

    arr = np.array(ids, dtype=np.uint16)
    n   = len(arr)

    split_idx   = int(n * (1.0 - val_split))
    train_arr   = arr[:split_idx]
    val_arr     = arr[split_idx:]

    np.save(train_out, train_arr)
    np.save(val_out,   val_arr)

    print("\n── Tokenisation Statistics ─────────────────────────────────")
    print(f"  Total tokens  : {n:>12,}")
    print(f"  Train tokens  : {len(train_arr):>12,}  ({100*(1-val_split):.0f} %)")
    print(f"  Val   tokens  : {len(val_arr):>12,}  ({100*val_split:.0f} %)")
    print(f"  Train saved   : {train_out}")
    print(f"  Val   saved   : {val_out}")
    print("────────────────────────────────────────────────────────────\n")

    return train_out, val_out


# ---------------------------------------------------------------------------
# 3.  TextDataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """
    PyTorch Dataset for causal language modelling.

    Wraps a 1-D integer token array (either a numpy array loaded from disk, a
    pre-built torch.Tensor, or a raw .txt file that will be tokenized on the
    fly) and returns overlapping ``(input_ids, target_ids)`` pairs of length
    ``context_length``.

    The target is the input shifted by one position — standard next-token
    prediction.

    Parameters
    ----------
    source : str | np.ndarray | torch.Tensor
        * ``str``          → interpreted as a path.
          - If it ends in ``.npy`` the numpy array is loaded directly.
          - Otherwise the file is read as UTF-8 text and tokenized with
            ``tokenizer`` (which must be supplied).
        * ``np.ndarray``   → used as-is (expected dtype int / uint16).
        * ``torch.Tensor`` → used as-is (expected dtype torch.long).
    context_length : int
        Number of tokens per sample (a.k.a. block_size).
    tokenizer : optional
        Required only when *source* is a raw .txt path.
    val_split : float
        Validation fraction; only used when *source* is a .txt file.
    is_val : bool
        When True the validation slice is used; otherwise the training slice.

    Backward-compatible shim
    ------------------------
    The old signature ``TextDataset(data_path, tokenizer, block_size, …)``
    is still accepted — positional argument detection picks the right branch.
    """

    def __init__(
        self,
        source,
        context_length: int | None = None,
        tokenizer=None,
        val_split: float = 0.10,
        is_val: bool = False,
        # ── legacy keyword aliases ──────────────────────────────────────
        block_size: int | None = None,      # alias for context_length
        data_path: str | None = None,       # alias for source when str
        token_cache_path: str | None = None,
    ):
        # ------------------------------------------------------------------
        # Resolve aliases (backward compat with old TextDataset signature)
        # ------------------------------------------------------------------
        if data_path is not None and source is None:
            source = data_path
        if block_size is not None and context_length is None:
            context_length = block_size

        if context_length is None:
            raise ValueError("context_length (or block_size) must be provided.")

        self.context_length = context_length

        # ------------------------------------------------------------------
        # Load / build the token tensor
        # ------------------------------------------------------------------
        if isinstance(source, (np.ndarray, torch.Tensor)):
            # Direct array hand-off
            all_tokens = (
                torch.from_numpy(source.astype(np.int64))
                if isinstance(source, np.ndarray)
                else source.long()
            )

        elif isinstance(source, str):
            if source.endswith(".npy"):
                # Pre-tokenized numpy binary
                print(f"Loading token array from: {source}")
                arr = np.load(source)
                all_tokens = torch.from_numpy(arr.astype(np.int64))

            else:
                # Raw .txt — tokenize on the fly (with optional cache)
                cache_hit = token_cache_path and os.path.exists(token_cache_path)

                if cache_hit:
                    print(f"Loading tokenized cache from: {token_cache_path}")
                    all_tokens = torch.load(token_cache_path, weights_only=True)
                else:
                    if tokenizer is None:
                        raise ValueError(
                            "A tokenizer must be supplied when source is a raw .txt file."
                        )
                    with open(source, "r", encoding="utf-8") as fh:
                        text = fh.read()
                    print("Tokenizing raw dataset …")
                    ids = tokenizer.encode(text)
                    all_tokens = torch.tensor(ids, dtype=torch.long)

                    if token_cache_path:
                        os.makedirs(os.path.dirname(token_cache_path), exist_ok=True)
                        torch.save(all_tokens, token_cache_path)
                        print(f"Cached tokenized dataset to: {token_cache_path}")

                # Train / val split for on-the-fly tokenized path
                n         = len(all_tokens)
                split_idx = int(n * (1.0 - val_split))
                all_tokens = all_tokens[split_idx:] if is_val else all_tokens[:split_idx]
                label = "Validation" if is_val else "Training"
                print(f"{label} dataset size: {len(all_tokens):,} tokens")

        else:
            raise TypeError(
                f"source must be a file path (str), np.ndarray, or torch.Tensor; "
                f"got {type(source)}"
            )

        if len(all_tokens) <= context_length:
            raise ValueError(
                f"Dataset has only {len(all_tokens):,} tokens which is ≤ "
                f"context_length ({context_length}). Use more data or reduce "
                f"context_length."
            )

        self.data = all_tokens

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Number of non-overlapping (stride-1) samples."""
        return len(self.data) - self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx     : idx + self.context_length]
        y = self.data[idx + 1 : idx + self.context_length + 1]
        return x, y


# ---------------------------------------------------------------------------
# 4.  get_dataloaders
# ---------------------------------------------------------------------------

def get_dataloaders(
    config: dict,
    tokenizer=None,
) -> tuple[DataLoader, DataLoader]:
    """
    Build and return ``(train_loader, val_loader)``.

    Config keys recognised
    ----------------------
    Required
    ~~~~~~~~
    block_size / context_length : int   — sequence length per sample

    Optional — data sources
    ~~~~~~~~~~~~~~~~~~~~~~~
    train_bin  : str  — path to train .npy  (default TRAIN_BIN_PATH)
    val_bin    : str  — path to val   .npy  (default VAL_BIN_PATH)
    data_path  : str  — raw .txt to tokenize on-the-fly (fallback)
    val_split  : float — train/val split if using raw .txt (default 0.10)

    Optional — loader settings
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    batch_size   : int   (default 32)
    num_workers  : int   (default 0 on Windows / MPS, else 4)
    pin_memory   : bool  (default True when CUDA is available)
    """

    context_length = config.get("context_length") or config.get("block_size")
    if context_length is None:
        raise ValueError("config must contain 'block_size' or 'context_length'.")

    batch_size  = config.get("batch_size",  32)
    val_split   = config.get("val_split",   0.10)

    # Decide num_workers sensibly per platform
    default_workers = 0 if (sys.platform == "win32" or config.get("device") == "mps") else 4
    num_workers = config.get("num_workers", default_workers)

    pin_memory = config.get("pin_memory", torch.cuda.is_available())

    # ------------------------------------------------------------------
    # Prefer pre-tokenized .npy files; fall back to raw .txt
    # ------------------------------------------------------------------
    train_bin = config.get("train_bin", TRAIN_BIN_PATH)
    val_bin   = config.get("val_bin",   VAL_BIN_PATH)

    if os.path.exists(train_bin) and os.path.exists(val_bin):
        print(f"Loading pre-tokenized splits:\n  train → {train_bin}\n  val   → {val_bin}")
        train_ds = TextDataset(train_bin, context_length=context_length)
        val_ds   = TextDataset(val_bin,   context_length=context_length)

    else:
        raw_path = config.get("data_path")
        if not raw_path:
            raise FileNotFoundError(
                f"Neither pre-tokenized splits ({train_bin}, {val_bin}) nor a "
                f"'data_path' key in config were found."
            )
        if tokenizer is None:
            raise ValueError(
                "tokenizer must be provided when loading from a raw .txt file."
            )
        print(f"Pre-tokenized splits not found; loading raw text from: {raw_path}")
        train_ds = TextDataset(
            raw_path, context_length=context_length,
            tokenizer=tokenizer, val_split=val_split, is_val=False,
        )
        val_ds = TextDataset(
            raw_path, context_length=context_length,
            tokenizer=tokenizer, val_split=val_split, is_val=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print(
        f"\n── DataLoader Summary ───────────────────────────────────────\n"
        f"  Train samples : {len(train_ds):>10,}   batches: {len(train_loader):>8,}\n"
        f"  Val   samples : {len(val_ds):>10,}   batches: {len(val_loader):>8,}\n"
        f"  Batch size    : {batch_size}\n"
        f"  Context length: {context_length}\n"
        f"  num_workers   : {num_workers}\n"
        f"────────────────────────────────────────────────────────────\n"
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    # Allow running from the repo root:  python -m nanosage.training.dataset
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from nanosage.tokenizer.bpe import BPETokenizer

    TOKENIZER_PATH = os.path.join("nanosage", "checkpoints", "tokenizer.json")

    # ── Step 1: download (skip if file already exists) ──────────────────────
    if not os.path.exists(RAW_TXT_PATH):
        print("=== Step 1: Downloading TinyStories ===")
        download_pretrain_data(num_stories=50_000)
    else:
        print(f"Raw data already present at {RAW_TXT_PATH}; skipping download.")

    # ── Step 2: train / load tokenizer ─────────────────────────────────────
    tokenizer = BPETokenizer()
    if os.path.exists(TOKENIZER_PATH):
        print(f"Loading tokenizer from {TOKENIZER_PATH}")
        tokenizer.load(TOKENIZER_PATH)
    else:
        print("Training tokenizer from scratch …")
        with open(RAW_TXT_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        tokenizer.train(text, vocab_size=4096, verbose=True)
        tokenizer.save(TOKENIZER_PATH)

    # ── Step 3: prepare (skip if .npy files already exist) ──────────────────
    if not (os.path.exists(TRAIN_BIN_PATH) and os.path.exists(VAL_BIN_PATH)):
        print("=== Step 3: Tokenizing & saving splits ===")
        prepare_pretrain_data(tokenizer)
    else:
        print("Pre-tokenized splits already exist; skipping prepare step.")

    # ── Step 4: build loaders and print a sample batch ──────────────────────
    print("=== Step 4: Building DataLoaders & sampling a batch ===")
    cfg = {"block_size": 128, "batch_size": 8}
    train_loader, val_loader = get_dataloaders(cfg)

    x_batch, y_batch = next(iter(train_loader))
    print(f"Sample batch — x shape: {x_batch.shape}  y shape: {y_batch.shape}")
    print(f"  x[0][:10] = {x_batch[0, :10].tolist()}")
    print(f"  y[0][:10] = {y_batch[0, :10].tolist()}")
    print("\nAll checks passed ✓")
