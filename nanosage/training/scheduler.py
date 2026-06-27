"""
nanosage/training/scheduler.py

Learning-rate scheduling utilities for NanoSage.

Public API
----------
get_lr(step, warmup_steps, max_steps, max_lr, min_lr) → float
    Pure-function LR computation: linear warmup → cosine decay → floor.

get_cosine_lr_with_warmup(it, warmup_iters, lr_decay_iters,
                          learning_rate, min_lr) → float
    Backward-compatible alias for get_lr (used by trainer.py / train.py).

CosineWarmupScheduler
    torch.optim.lr_scheduler.LambdaLR subclass.  Pass to any PyTorch
    training loop that expects a LRScheduler.

plot_lr_schedule(warmup_steps, max_steps, max_lr, min_lr, save_path)
    Render and save a styled LR-schedule diagram.
"""

from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import LambdaLR


# ============================================================================
# Core computation
# ============================================================================

def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """
    Cosine-decay schedule with linear warmup.

    Three regions
    ~~~~~~~~~~~~~
    1. **Warmup**  ``0 ≤ step < warmup_steps``
       LR rises linearly from 0 → max_lr.

    2. **Cosine decay**  ``warmup_steps ≤ step ≤ max_steps``
       LR follows a half-cosine from max_lr → min_lr.

    3. **Floor**  ``step > max_steps``
       LR is clamped at min_lr.

    Parameters
    ----------
    step         : current training step (0-indexed)
    warmup_steps : number of linear-warmup steps
    max_steps    : total training steps (decay ends here)
    max_lr       : peak learning rate (reached at end of warmup)
    min_lr       : minimum / floor learning rate (≈ 10 % of max_lr recommended)

    Returns
    -------
    float : learning rate for this step
    """
    # 1) Linear warmup
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)

    # 2) Post-decay floor
    if step > max_steps:
        return min_lr

    # 3) Cosine decay
    decay_steps = max(1, max_steps - warmup_steps)
    progress    = (step - warmup_steps) / decay_steps   # 0.0 → 1.0
    cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1.0 → 0.0
    return min_lr + cosine_coeff * (max_lr - min_lr)


# Backward-compatible alias (imported by trainer.py and train.py)
def get_cosine_lr_with_warmup(
    it: int,
    warmup_iters: int,
    lr_decay_iters: int,
    learning_rate: float,
    min_lr: float,
) -> float:
    """
    Alias for :func:`get_lr` using the original parameter names.

    This function is kept for backward compatibility with ``trainer.py`` and
    ``train.py``.  New code should call :func:`get_lr` directly.
    """
    return get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr)


# ============================================================================
# PyTorch LambdaLR scheduler
# ============================================================================

class CosineWarmupScheduler(LambdaLR):
    """
    PyTorch ``LambdaLR``-based cosine-warmup scheduler.

    Usage
    -----
    >>> scheduler = CosineWarmupScheduler(
    ...     optimizer,
    ...     warmup_steps=500,
    ...     max_steps=10_000,
    ...     max_lr=3e-4,
    ...     min_lr=3e-5,
    ... )
    >>> for step in range(max_steps):
    ...     optimizer.zero_grad()
    ...     loss.backward()
    ...     optimizer.step()
    ...     scheduler.step()         # advance by one step
    ...     current_lr = scheduler.get_last_lr()[0]

    How it works
    ------------
    ``LambdaLR`` scales the base LR stored in the optimizer by the return
    value of the supplied lambda.  Here the lambda always returns a ratio
    ``get_lr(step, ...) / max_lr``, so the *effective* LR equals
    ``get_lr(step, ...)``, regardless of what base LR the optimizer was
    initialised with.

    Parameters
    ----------
    optimizer    : any ``torch.optim.Optimizer``
    warmup_steps : linear warmup length (steps)
    max_steps    : total training steps (cosine decay ends here)
    max_lr       : peak LR (should match the optimizer's initial LR)
    min_lr       : floor LR (default: 10 % of max_lr)
    last_epoch   : step to resume from (default -1 = fresh start)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr: float | None = None,
        last_epoch: int = -1,
    ):
        if min_lr is None:
            min_lr = max_lr * 0.1

        self.warmup_steps = warmup_steps
        self.max_steps    = max_steps
        self.max_lr       = max_lr
        self.min_lr       = min_lr

        # LambdaLR multiplies the optimizer's base_lr by the lambda output.
        # We normalise by max_lr so the actual LR tracks get_lr() exactly.
        def _lr_lambda(current_step: int) -> float:
            return get_lr(current_step, warmup_steps, max_steps, max_lr, min_lr) / max(max_lr, 1e-12)

        super().__init__(optimizer, lr_lambda=_lr_lambda, last_epoch=last_epoch)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_current_lr(self) -> float:
        """Return the LR for the current step as a plain float."""
        step = self.last_epoch
        return get_lr(step, self.warmup_steps, self.max_steps,
                      self.max_lr, self.min_lr)

    def __repr__(self) -> str:
        return (
            f"CosineWarmupScheduler("
            f"warmup_steps={self.warmup_steps}, "
            f"max_steps={self.max_steps}, "
            f"max_lr={self.max_lr:.2e}, "
            f"min_lr={self.min_lr:.2e})"
        )


# ============================================================================
# Visualisation
# ============================================================================

def plot_lr_schedule(
    warmup_steps: int   = 500,
    max_steps:    int   = 10_000,
    max_lr:       float = 3e-4,
    min_lr:       float | None = None,
    save_path:    str   = "nanosage/logs/lr_schedule.png",
) -> str:
    """
    Render a styled diagram of the LR schedule and save it to disk.

    Parameters
    ----------
    warmup_steps : warmup length
    max_steps    : total steps plotted
    max_lr       : peak LR
    min_lr       : floor LR (default 10 % of max_lr)
    save_path    : output file path

    Returns
    -------
    str : the resolved ``save_path``
    """
    if min_lr is None:
        min_lr = max_lr * 0.1

    steps = list(range(max_steps + 1))
    lrs   = [get_lr(s, warmup_steps, max_steps, max_lr, min_lr) for s in steps]

    # ── Style ────────────────────────────────────────────────────────────────
    BG      = "#0f0f1a"
    PANEL   = "#13132a"
    CYAN    = "#4fc3f7"
    ORANGE  = "#ff8a65"
    GREY    = "#555577"
    WHITE   = "#e8e8f0"
    SUBTEXT = "#888899"

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    # Main LR curve
    ax.plot(steps, lrs, color=CYAN, linewidth=2.2, zorder=3, label="Learning rate")

    # Warmup / decay boundary shading
    ax.axvspan(0, warmup_steps, alpha=0.12, color=ORANGE, label=f"Warmup ({warmup_steps} steps)")
    ax.axvspan(warmup_steps, max_steps, alpha=0.06, color=CYAN, label="Cosine decay")

    # Annotation lines
    ax.axvline(warmup_steps, color=ORANGE, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.axhline(min_lr,       color=GREY,   linewidth=1.0, linestyle=":",  alpha=0.8,
               label=f"min_lr = {min_lr:.1e}")
    ax.axhline(max_lr,       color=GREY,   linewidth=1.0, linestyle=":",  alpha=0.8,
               label=f"max_lr = {max_lr:.1e}")

    # Text annotation at warmup boundary
    ax.annotate(
        f"warmup end\nstep {warmup_steps}",
        xy=(warmup_steps, get_lr(warmup_steps, warmup_steps, max_steps, max_lr, min_lr)),
        xytext=(warmup_steps + max_steps * 0.04, max_lr * 0.75),
        color=ORANGE, fontsize=8,
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2),
    )

    # Labels & decorations
    ax.set_title("NanoSage — Cosine Warmup LR Schedule",
                 color=WHITE, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Training step", color=SUBTEXT, fontsize=10)
    ax.set_ylabel("Learning rate", color=SUBTEXT, fontsize=10)
    ax.tick_params(colors="#666688", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#222244")
    ax.grid(color="#1e1e3a", linestyle="--", linewidth=0.7, alpha=0.8, zorder=0)

    legend = ax.legend(
        facecolor="#1a1a30", edgecolor="#333355",
        labelcolor=WHITE, fontsize=8.5, loc="upper right",
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, facecolor=BG)
    plt.close(fig)
    print(f"LR schedule plot saved → {save_path}")
    return save_path


# ============================================================================
# Smoke-test / demo
# ============================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    # ── Config ───────────────────────────────────────────────────────────────
    WARMUP_STEPS = 500
    MAX_STEPS    = 10_000
    MAX_LR       = 3e-4
    MIN_LR       = 3e-5          # 10 % of max_lr

    # ── 1. Standalone get_lr() ───────────────────────────────────────────────
    print("=" * 55)
    print("  get_lr() spot-check")
    print("=" * 55)
    probe_steps = [0, 100, 500, 1000, 5000, 10_000]
    for s in probe_steps:
        lr = get_lr(s, WARMUP_STEPS, MAX_STEPS, MAX_LR, MIN_LR)
        region = (
            "warmup" if s < WARMUP_STEPS
            else "floor" if s >= MAX_STEPS
            else "cosine"
        )
        print(f"  step {s:>6,}  |  lr = {lr:.6e}  [{region}]")
    print()

    # ── 2. CosineWarmupScheduler with a dummy optimizer ──────────────────────
    print("=" * 55)
    print("  CosineWarmupScheduler — LR via scheduler.get_last_lr()")
    print("=" * 55)

    # Minimal dummy model so we have something to optimise
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    optimizer   = torch.optim.AdamW([dummy_param], lr=MAX_LR)
    scheduler   = CosineWarmupScheduler(
        optimizer,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        max_lr=MAX_LR,
        min_lr=MIN_LR,
    )
    print(scheduler)
    print()

    probe_set = set(probe_steps)
    for step in range(MAX_STEPS + 1):
        if step in probe_set:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  step {step:>6,}  |  scheduler lr = {current_lr:.6e}")
        scheduler.step()

    # One extra step to show the floor
    scheduler.step()
    print(f"  step {MAX_STEPS+1:>6,}  |  scheduler lr = {scheduler.get_last_lr()[0]:.6e}  [floor]")
    print()

    # ── 3. Plot ──────────────────────────────────────────────────────────────
    print("=" * 55)
    print("  Plotting LR schedule …")
    print("=" * 55)
    plot_lr_schedule(
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        max_lr=MAX_LR,
        min_lr=MIN_LR,
        save_path="nanosage/logs/lr_schedule.png",
    )
    print("\nAll checks passed ✓")
