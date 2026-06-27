"""
nanosage/inference/generate.py

All text-generation strategies for NanoSage.

Public API
----------
generate(model, idx, max_new_tokens, ...)         -- original low-level tensor API (kept for backward compat)
greedy_decode(model, tokenizer, prompt, ...)      -- greedy argmax decoding
sample_decode(model, tokenizer, prompt, config)   -- temperature + top-k + top-p + repetition penalty
beam_search(model, tokenizer, prompt, ...)        -- beam search (beam_width beams)
batch_generate(model, tokenizer, prompts, config) -- parallel batch generation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F


# ============================================================================
# Config dataclass for sample_decode / batch_generate
# ============================================================================

@dataclass
class GenerationConfig:
    """
    Sampling hyper-parameters consumed by sample_decode and batch_generate.

    Attributes
    ----------
    max_new_tokens      : maximum tokens to generate
    temperature         : logit temperature (1.0 = unchanged, <1 = sharper, >1 = flatter)
    top_k               : keep only top-k logits before sampling (None = disabled)
    top_p               : nucleus probability mass cutoff (None = disabled)
    repetition_penalty  : penalise tokens that already appear in context (1.0 = off)
    eos_token_id        : stop generation when this token is produced
    """
    max_new_tokens:     int   = 128
    temperature:        float = 0.8
    top_k:              Optional[int]   = 50
    top_p:              Optional[float] = 0.9
    repetition_penalty: float = 1.1
    eos_token_id:       Optional[int] = None


# ============================================================================
# Internal helpers
# ============================================================================

def _encode(tokenizer, text: str) -> torch.Tensor:
    """Encode a string prompt → 1-D LongTensor."""
    ids = tokenizer.encode(text)
    return torch.tensor(ids, dtype=torch.long)


def _decode_ids(tokenizer, ids) -> str:
    """Decode a list/tensor of token ids → string, stripping the EOS marker."""
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    text = tokenizer.decode(ids)
    return text.replace("<|endoftext|>", "").strip()


def _crop(idx: torch.Tensor, block_size: int) -> torch.Tensor:
    """Crop sequence to model's context window."""
    return idx if idx.size(1) <= block_size else idx[:, -block_size:]


def _apply_repetition_penalty(logits: torch.Tensor, context_ids: torch.Tensor,
                               penalty: float) -> torch.Tensor:
    """
    Divide logits of tokens already present in *context_ids* by *penalty*
    (penalty > 1 → discourages repetition; 1.0 → no-op).
    """
    if penalty == 1.0:
        return logits
    for token_id in context_ids.unique():
        logits[token_id] /= penalty
    return logits


def _filter_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits below the k-th highest value."""
    if k <= 0:
        return logits
    threshold = torch.topk(logits, min(k, logits.size(-1))).values[-1]
    logits[logits < threshold] = float("-inf")
    return logits


def _filter_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering — keep the smallest set of tokens whose
    cumulative probability mass ≥ p."""
    if p >= 1.0 or p <= 0.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens beyond the nucleus
    remove = cum_probs > p
    # Shift right so the first token that breaches the threshold is kept
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits[remove] = float("-inf")
    # Scatter back to original order
    logits.scatter_(0, sorted_indices, sorted_logits)
    return logits


# ============================================================================
# 0.  Original low-level generate() — kept for backward compatibility
# ============================================================================

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None,
             top_p=None, eos_token_id=None):
    """
    Low-level autoregressive generation from a token tensor.

    Parameters
    ----------
    model          : NanoSageLM
    idx            : LongTensor of shape (B, T) — prompt token ids
    max_new_tokens : int
    temperature    : float  (0 → greedy)
    top_k          : int | None
    top_p          : float | None
    eos_token_id   : int | None

    Returns
    -------
    LongTensor of shape (B, T + generated) with the full sequence.
    """
    model.eval()
    block_size = model.config.block_size

    for _ in range(max_new_tokens):
        idx_cond = _crop(idx, block_size)
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]       # (B, vocab_size)

        if temperature == 0.0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None and 0.0 < top_p < 1.0:
                s_logits, s_idx = torch.sort(logits, descending=True, dim=-1)
                cum_p = torch.cumsum(F.softmax(s_logits, dim=-1), dim=-1)
                remove = cum_p > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                remove_orig = remove.scatter(dim=-1, index=s_idx, src=remove)
                logits[remove_orig] = float("-inf")

            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

        if eos_token_id is not None and (idx_next == eos_token_id).all():
            break

    return idx


# ============================================================================
# 1.  greedy_decode
# ============================================================================

@torch.no_grad()
def greedy_decode(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 128,
    device: Optional[torch.device] = None,
) -> str:
    """
    Greedy decoding — always pick the single highest-probability next token.

    This is the fastest and most deterministic strategy, but tends to produce
    repetitive or bland text.

    Parameters
    ----------
    model      : NanoSageLM
    tokenizer  : BPETokenizer
    prompt     : str — input text
    max_tokens : int — maximum new tokens to generate
    device     : torch.device (defaults to model's current device)

    Returns
    -------
    str : the generated continuation (prompt **included**)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    block_size = model.config.block_size
    eos_id     = tokenizer.special_tokens.get("<|endoftext|>")

    idx = _encode(tokenizer, prompt).unsqueeze(0).to(device)   # (1, T)
    prompt_len = idx.size(1)

    for _ in range(max_tokens):
        idx_cond     = _crop(idx, block_size)
        logits, _    = model(idx_cond)
        logits       = logits[:, -1, :]          # (1, vocab)
        idx_next     = torch.argmax(logits, dim=-1, keepdim=True)  # (1, 1)
        idx          = torch.cat((idx, idx_next), dim=1)

        if eos_id is not None and idx_next.item() == eos_id:
            break

    return _decode_ids(tokenizer, idx[0, prompt_len:])


# ============================================================================
# 2.  sample_decode
# ============================================================================

@torch.no_grad()
def sample_decode(
    model,
    tokenizer,
    prompt: str,
    config: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None,
) -> str:
    """
    Stochastic sampling with temperature, top-k, top-p, and repetition penalty.

    Parameters
    ----------
    model      : NanoSageLM
    tokenizer  : BPETokenizer
    prompt     : str
    config     : GenerationConfig — sampling parameters (uses defaults if None)
    device     : torch.device

    Returns
    -------
    str : generated continuation (prompt **not** included)
    """
    if config is None:
        config = GenerationConfig()
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    block_size = model.config.block_size
    eos_id     = config.eos_token_id or tokenizer.special_tokens.get("<|endoftext|>")

    idx = _encode(tokenizer, prompt).unsqueeze(0).to(device)   # (1, T)
    prompt_len = idx.size(1)

    for _ in range(config.max_new_tokens):
        idx_cond = _crop(idx, block_size)
        logits, _ = model(idx_cond)
        logits    = logits[0, -1, :].clone()     # (vocab,)  — work with 1-D

        # Repetition penalty
        logits = _apply_repetition_penalty(logits, idx[0], config.repetition_penalty)

        # Temperature
        temp = max(config.temperature, 1e-8)
        logits = logits / temp

        # Top-k
        if config.top_k is not None and config.top_k > 0:
            logits = _filter_top_k(logits, config.top_k)

        # Top-p
        if config.top_p is not None:
            logits = _filter_top_p(logits, config.top_p)

        probs    = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).unsqueeze(0)   # (1, 1)
        idx      = torch.cat((idx, idx_next), dim=1)

        if eos_id is not None and idx_next.item() == eos_id:
            break

    return _decode_ids(tokenizer, idx[0, prompt_len:])


# ============================================================================
# 3.  beam_search
# ============================================================================

@torch.no_grad()
def beam_search(
    model,
    tokenizer,
    prompt: str,
    beam_width: int = 3,
    max_tokens: int = 128,
    length_penalty: float = 0.7,
    device: Optional[torch.device] = None,
) -> str:
    """
    Beam search — maintain the *beam_width* most probable partial sequences,
    return the one with the highest length-normalised log-probability.

    Parameters
    ----------
    model          : NanoSageLM
    tokenizer      : BPETokenizer
    prompt         : str
    beam_width     : int — number of beams (3 default)
    max_tokens     : int
    length_penalty : float — exponent on sequence length for score normalisation
                     (< 1 → favour shorter sequences, > 1 → favour longer)
    device         : torch.device

    Returns
    -------
    str : best decoded continuation (prompt **not** included)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    block_size = model.config.block_size
    eos_id     = tokenizer.special_tokens.get("<|endoftext|>")

    prompt_ids = _encode(tokenizer, prompt).to(device)    # (T,)
    prompt_len = prompt_ids.size(0)

    # Each beam: [token_ids_tensor, cumulative_log_prob, is_done]
    beams: list[list] = [[prompt_ids, 0.0, False]]

    for step in range(max_tokens):
        if all(b[2] for b in beams):
            break                          # all beams finished

        candidates: list[list] = []

        for seq, score, done in beams:
            if done:
                candidates.append([seq, score, True])
                continue

            idx_cond   = _crop(seq.unsqueeze(0), block_size)
            logits, _  = model(idx_cond)
            logits     = logits[0, -1, :]                # (vocab,)
            log_probs  = F.log_softmax(logits, dim=-1)   # (vocab,)

            # Expand this beam into top-beam_width candidates
            top_lp, top_ids = torch.topk(log_probs, beam_width)
            for lp, tok in zip(top_lp.tolist(), top_ids.tolist()):
                new_seq   = torch.cat([seq, torch.tensor([tok], device=device)])
                new_score = score + lp
                is_done   = (tok == eos_id)
                candidates.append([new_seq, new_score, is_done])

        # Keep top beam_width by length-penalised score
        def _penalised(cand):
            seq_len = cand[0].size(0) - prompt_len
            return cand[1] / max(seq_len, 1) ** length_penalty

        candidates.sort(key=_penalised, reverse=True)
        beams = candidates[:beam_width]

    # Pick the beam with highest length-normalised score
    def _final_score(b):
        seq_len = b[0].size(0) - prompt_len
        return b[1] / max(seq_len, 1) ** length_penalty

    best_seq = max(beams, key=_final_score)[0]
    return _decode_ids(tokenizer, best_seq[prompt_len:])


# ============================================================================
# 4.  batch_generate
# ============================================================================

@torch.no_grad()
def batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    config: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None,
) -> List[str]:
    """
    Generate continuations for *multiple prompts* in a single batched forward pass.

    Prompts are left-padded to the same length so they can be stacked into
    a single (B, T) tensor.  This is significantly faster than calling
    sample_decode() in a loop when the model fits in GPU memory.

    Parameters
    ----------
    model     : NanoSageLM
    tokenizer : BPETokenizer
    prompts   : list[str]
    config    : GenerationConfig (uses defaults if None)
    device    : torch.device

    Returns
    -------
    list[str] : one decoded continuation per prompt (prompts **not** included)
    """
    if config is None:
        config = GenerationConfig()
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    block_size = model.config.block_size
    eos_id     = config.eos_token_id or tokenizer.special_tokens.get("<|endoftext|>")
    pad_id     = eos_id if eos_id is not None else 0

    # Encode all prompts
    encoded: list[list[int]] = [tokenizer.encode(p) for p in prompts]
    prompt_lens = [len(e) for e in encoded]
    max_len     = max(prompt_lens)

    # Left-pad so all sequences are the same length
    padded = [
        [pad_id] * (max_len - len(e)) + e
        for e in encoded
    ]
    idx = torch.tensor(padded, dtype=torch.long, device=device)   # (B, max_len)

    # Track which sequences have hit EOS
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=device)

    for _ in range(config.max_new_tokens):
        if finished.all():
            break

        idx_cond  = _crop(idx, block_size)
        logits, _ = model(idx_cond)
        logits    = logits[:, -1, :]                   # (B, vocab)

        # Apply temperature
        temp = max(config.temperature, 1e-8)
        logits = logits / temp

        # Top-k (vectorised over batch)
        if config.top_k is not None and config.top_k > 0:
            k = min(config.top_k, logits.size(-1))
            threshold = torch.topk(logits, k, dim=-1).values[:, -1:]   # (B, 1)
            logits[logits < threshold] = float("-inf")

        # Top-p (per-sample nucleus)
        if config.top_p is not None and 0.0 < config.top_p < 1.0:
            s_logits, s_idx = torch.sort(logits, descending=True, dim=-1)
            cum_p = torch.cumsum(F.softmax(s_logits, dim=-1), dim=-1)
            remove = cum_p > config.top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0]  = False
            remove_orig = remove.scatter(dim=-1, index=s_idx, src=remove)
            logits[remove_orig] = float("-inf")

        probs    = F.softmax(logits, dim=-1)               # (B, vocab)
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

        # For finished sequences, output the pad token (won't be decoded)
        idx_next[finished] = pad_id

        idx      = torch.cat((idx, idx_next), dim=1)

        # Mark newly finished sequences
        if eos_id is not None:
            finished |= (idx_next.squeeze(-1) == eos_id)

    results = []
    for i, plen in enumerate(prompt_lens):
        # Trim left-padding from prompt side
        generated = idx[i, max_len:].tolist()
        # Remove any trailing EOS tokens
        if eos_id is not None and eos_id in generated:
            generated = generated[:generated.index(eos_id)]
        results.append(tokenizer.decode(generated).replace("<|endoftext|>", "").strip())

    return results


# ============================================================================
# Smoke-test / demo
# ============================================================================

if __name__ == "__main__":
    import os, sys, textwrap, time
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    import torch
    from nanosage.tokenizer.bpe import BPETokenizer
    from nanosage.model.transformer import NanoSageLM

    # ── Config ───────────────────────────────────────────────────────────────
    CKPT  = "nanosage/checkpoints/best_model.pt"
    TPATH = "nanosage/checkpoints/tokenizer.json"
    PROMPT = "Once upon a time"
    MAX_TOKENS = 64

    device = (
        torch.device("cuda")  if torch.cuda.is_available() else
        torch.device("mps")   if torch.backends.mps.is_available() else
        torch.device("cpu")
    )

    # ── Load model & tokenizer ────────────────────────────────────────────────
    print(f"Loading model from {CKPT} on {device} …")
    ckpt       = torch.load(CKPT, map_location=device, weights_only=False)
    model_cfg  = ckpt["model_config"]
    model      = NanoSageLM(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    tokenizer = BPETokenizer()
    tokenizer.load(TPATH)
    print(f"Vocab size: {model_cfg.vocab_size}  |  Context: {model_cfg.block_size}\n")

    cfg = GenerationConfig(
        max_new_tokens=MAX_TOKENS,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    # ── Helper ────────────────────────────────────────────────────────────────
    W = 70   # display column width
    def banner(title):
        print(f"\n{'─'*W}")
        print(f"  {title}")
        print(f"{'─'*W}")

    def show(label, text, elapsed):
        wrapped = textwrap.fill(text.strip(), width=W - 4, subsequent_indent="    ")
        print(f"\n  ▶ {label}  ({elapsed*1000:.0f} ms)\n    {wrapped}\n")

    print(f'  Prompt: "{PROMPT}"\n')

    # ── 1. Greedy decode ──────────────────────────────────────────────────────
    banner("1. Greedy Decode")
    t0 = time.perf_counter()
    out_greedy = greedy_decode(model, tokenizer, PROMPT, max_tokens=MAX_TOKENS, device=device)
    show("Greedy", out_greedy, time.perf_counter() - t0)

    # ── 2. Sample decode ──────────────────────────────────────────────────────
    banner("2. Sample Decode  (temp=0.8, top_k=50, top_p=0.9, rep_penalty=1.1)")
    t0 = time.perf_counter()
    out_sample = sample_decode(model, tokenizer, PROMPT, config=cfg, device=device)
    show("Sample", out_sample, time.perf_counter() - t0)

    # ── 3. Beam search ────────────────────────────────────────────────────────
    banner("3. Beam Search  (beam_width=3)")
    t0 = time.perf_counter()
    out_beam = beam_search(model, tokenizer, PROMPT, beam_width=3, max_tokens=MAX_TOKENS, device=device)
    show("Beam", out_beam, time.perf_counter() - t0)

    # ── 4. Batch generate ─────────────────────────────────────────────────────
    banner("4. Batch Generate  (3 prompts at once)")
    batch_prompts = [
        "Once upon a time",
        "The little robot",
        "In a faraway land",
    ]
    t0 = time.perf_counter()
    out_batch = batch_generate(model, tokenizer, batch_prompts, config=cfg, device=device)
    elapsed_batch = time.perf_counter() - t0
    for p, o in zip(batch_prompts, out_batch):
        wrapped = textwrap.fill(o.strip(), width=W - 10, subsequent_indent="           ")
        print(f"\n  [{p!r}]\n  → {wrapped}")
    print(f"\n  Total batch time: {elapsed_batch*1000:.0f} ms\n")

    # ── Side-by-side comparison table ─────────────────────────────────────────
    print(f"\n{'═'*W}")
    print(f'  SIDE-BY-SIDE COMPARISON — prompt: "{PROMPT}"')
    print(f"{'═'*W}")
    rows = [("Greedy",  out_greedy),
            ("Sample",  out_sample),
            ("Beam(3)", out_beam)]
    for name, txt in rows:
        preview = (txt.strip()[:55] + "…") if len(txt.strip()) > 55 else txt.strip()
        print(f"  {name:<10}: {preview}")
    print(f"{'═'*W}\n")
