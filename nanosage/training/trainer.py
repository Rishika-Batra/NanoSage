"""
nanosage/training/trainer.py

Provides two complementary APIs:

Functional API (new)
--------------------
  train_epoch(model, loader, optimizer, scheduler, config)
      → avg_train_loss (float)

  evaluate(model, loader, device)
      → {"val_loss": float, "perplexity": float}

  train(model, tokenizer, config)
      → history dict  {"train_loss", "val_loss", "perplexity", "steps"}

  load_checkpoint(model, optimizer, path)
      → (step, history)

Class API (original — used by train.py)
----------------------------------------
  Trainer(model, train_dataset, val_dataset, train_config, checkpoint=None)
      .train()
      .estimate_loss()
      .save_checkpoint(name, val_loss)
      .plot_loss_curve()
"""

from __future__ import annotations

import math
import os
import time
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .scheduler import get_cosine_lr_with_warmup

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_device() -> torch.device:
    """Return CUDA > MPS > CPU, whichever is available first."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _fmt_time(seconds: float) -> str:
    """Format seconds → 'Xh Ym Zs' human-readable string."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _plot_loss_curve(
    train_records: list[tuple[int, float]],
    val_records:   list[tuple[int, float]],
    save_path: str = "nanosage/logs/loss_curve.png",
) -> None:
    """Render and save a styled train / val loss curve."""
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    if train_records:
        steps, vals = zip(*train_records)
        ax.plot(steps, vals, color="#4fc3f7", linewidth=2,
                label="Train loss", alpha=0.9)

    if val_records:
        steps, vals = zip(*val_records)
        ax.plot(steps, vals, color="#ff8a65", linewidth=2,
                linestyle="--", label="Val loss", alpha=0.9)

    ax.set_title("NanoSage — Training Progress",
                 color="white", fontsize=14, pad=12)
    ax.set_xlabel("Step", color="#aaaaaa")
    ax.set_ylabel("Cross-Entropy Loss", color="#aaaaaa")
    ax.tick_params(colors="#888888")
    ax.spines[:].set_color("#333355")
    ax.grid(color="#222244", linestyle="--", linewidth=0.6, alpha=0.7)
    legend = ax.legend(facecolor="#1a1a2e", edgecolor="#333355",
                       labelcolor="white", fontsize=10)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


# ============================================================================
# FUNCTIONAL API
# ============================================================================

def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,                          # callable(step) → lr  OR  LRScheduler
    config:    dict,
    *,
    device:    torch.device | None = None,
    step_offset: int = 0,
    pbar=None,                          # optional external tqdm bar
) -> tuple[float, int]:
    """
    Run one pass over *loader*, applying gradient accumulation and clipping.

    Parameters
    ----------
    model        : NanoSageLM (or any nn.Module with forward(x, y) → (logits, loss))
    loader       : DataLoader over the training split
    optimizer    : AdamW (or any torch optimizer)
    scheduler    : callable(global_step) → float  **or**  a torch LRScheduler.
                   If callable, its return value is applied directly to all
                   param groups each micro-batch.  If it is a torch LRScheduler
                   object, ``.step()`` is called once per optimizer step.
    config       : dict with keys:
                     grad_clip              (float,  default 1.0)
                     grad_accumulation_steps(int,    default 1)
                     log_interval           (int,    default 10)
    step_offset  : global step count at the start of this epoch (for scheduler)
    pbar         : optional tqdm progress bar to update

    Returns
    -------
    (avg_loss, steps_taken)
    """
    if device is None:
        device = _auto_device()

    model.train()

    grad_clip   = config.get("grad_clip",               1.0)
    accum_steps = config.get("grad_accumulation_steps", 1)
    log_interval= config.get("log_interval",            10)

    total_loss   = 0.0
    total_batches= 0
    global_step  = step_offset

    loader_iter  = iter(loader)
    done         = False

    while not done:
        optimizer.zero_grad(set_to_none=True)
        micro_loss_accum = 0.0

        # ── Gradient accumulation micro-steps ──────────────────────────
        for micro in range(accum_steps):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                done = True
                break

            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            loss    = loss / accum_steps
            micro_loss_accum += loss.item()
            loss.backward()

        if micro_loss_accum == 0.0:
            break                       # loader exhausted on first micro-step

        # ── Apply LR from scheduler ────────────────────────────────────
        if callable(scheduler) and not isinstance(
            scheduler, torch.optim.lr_scheduler.LRScheduler
        ):
            lr = scheduler(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        # ── Gradient clip + optimizer step ─────────────────────────────
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        if isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            scheduler.step()

        total_loss    += micro_loss_accum
        total_batches += 1
        global_step   += 1

        if pbar is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{micro_loss_accum:.4f}",
                lr=f"{current_lr:.2e}",
                refresh=False,
            )
            pbar.update(1)
        elif total_batches % log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  step {global_step:6d} | "
                f"loss {micro_loss_accum:.4f} | "
                f"lr {current_lr:.3e}"
            )

    avg_loss = total_loss / max(total_batches, 1)
    return avg_loss, global_step - step_offset


# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:  nn.Module,
    loader: DataLoader,
    *,
    device: torch.device | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """
    Evaluate *model* on *loader* without gradient computation.

    Parameters
    ----------
    model       : NanoSageLM
    loader      : DataLoader over the validation split
    device      : inference device (auto-detected if None)
    max_batches : if set, only evaluate this many batches (fast estimate)

    Returns
    -------
    {"val_loss": float, "perplexity": float}
    """
    if device is None:
        device = _auto_device()

    model.eval()

    total_loss   = 0.0
    total_batches= 0

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss    += loss.item()
        total_batches += 1

    avg_loss   = total_loss / max(total_batches, 1)
    perplexity = math.exp(min(avg_loss, 20.0))   # clamp to avoid overflow

    model.train()
    return {"val_loss": avg_loss, "perplexity": perplexity}


# ---------------------------------------------------------------------------

def train(
    model,
    tokenizer,
    config: dict,
) -> dict:
    """
    Full training loop.

    Parameters
    ----------
    model     : NanoSageLM
    tokenizer : BPETokenizer (used only if dataset construction needs it)
    config    : dict with keys:

      Data / loader
        train_loader  DataLoader  — pre-built, OR provide dataset config keys
        val_loader    DataLoader  — pre-built, OR provide dataset config keys
        block_size    int         context length
        batch_size    int         (default 32)
        data_path     str         raw .txt path (if loaders not pre-built)

      Optimisation
        learning_rate float  (default 3e-4)
        min_lr        float  (default 1e-5)
        weight_decay  float  (default 0.1)
        betas         tuple  (default (0.9, 0.95))
        warmup_iters  int    (default 100)
        max_iters     int    total training steps (default 2000)
        grad_clip     float  (default 1.0)
        grad_accumulation_steps int (default 1)

      Logging / checkpointing
        eval_interval   int  steps between evaluations (default 200)
        eval_iters      int  batches per eval (default None → full val set)
        save_interval   int  steps between checkpoints (default 500)
        checkpoint_dir  str  (default "nanosage/checkpoints")
        log_dir         str  (default "nanosage/logs")
        device          str  (default auto-detected)
        resume          str  path to .pt checkpoint to resume from (optional)

    Returns
    -------
    history : {
        "steps":      list[int],
        "train_loss": list[float],
        "val_loss":   list[float],
        "perplexity": list[float],
    }
    """
    # ── Device ─────────────────────────────────────────────────────────────
    _dev_str = config.get("device", "auto")
    if _dev_str == "auto":
        device = _auto_device()
    else:
        device = torch.device(_dev_str)
    print(f"[train] Using device: {device}")

    model.to(device)

    # ── Dirs ────────────────────────────────────────────────────────────────
    ckpt_dir = config.get("checkpoint_dir", "nanosage/checkpoints")
    log_dir  = config.get("log_dir",        "nanosage/logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    # ── Optimiser ───────────────────────────────────────────────────────────
    lr           = config.get("learning_rate", 3e-4)
    min_lr       = config.get("min_lr",        1e-5)
    weight_decay = config.get("weight_decay",  0.1)
    betas        = config.get("betas",         (0.9, 0.95))
    max_iters    = config.get("max_iters",     1000)
    warmup_iters = config.get("warmup_iters",  100)

    optimizer = config.get("optimizer")
    if optimizer is None:
        optimizer = model.configure_optimizers(weight_decay, lr, betas, device.type)

    # ── LR scheduler (cosine with warmup) ──────────────────────────────────
    scheduler = config.get("scheduler")
    if scheduler is None:
        def _scheduler(step: int) -> float:
            return get_cosine_lr_with_warmup(step, warmup_iters, max_iters, lr, min_lr)

    # ── DataLoaders ─────────────────────────────────────────────────────────
    if "train_loader" in config and "val_loader" in config:
        train_loader = config["train_loader"]
        val_loader   = config["val_loader"]
    else:
        from .dataset import get_dataloaders
        train_loader, val_loader = get_dataloaders(config, tokenizer)

    # ── Optionally resume ────────────────────────────────────────────────────
    history: dict = {"steps": [], "train_loss": [], "val_loss": [], "perplexity": []}
    global_step = 0

    if config.get("resume"):
        global_step, history = load_checkpoint(model, optimizer, config["resume"])
        if isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            scheduler.last_epoch = global_step
        print(f"[train] Resumed from step {global_step}.")

    # ── Hyper-params for the loop ─────────────────────────────────────────
    eval_interval = config.get("eval_interval",  200)
    eval_iters    = config.get("eval_iters",      None)
    save_interval = config.get("save_interval",   500)
    accum_steps   = config.get("grad_accumulation_steps", 1)
    grad_clip     = config.get("grad_clip",        1.0)
    log_interval  = config.get("log_interval",     10)

    # ── Main loop ────────────────────────────────────────────────────────────
    model.train()
    train_iter    = iter(train_loader)
    t_loop_start  = time.perf_counter()
    t_step_start  = time.perf_counter()

    print(f"\n{'─'*60}")
    print(f"  NanoSage training  |  {max_iters:,} steps  |  device: {device}")
    print(f"{'─'*60}\n")

    # Build a tqdm bar if available
    pbar = None
    if _TQDM_AVAILABLE:
        pbar = tqdm(
            total=max_iters,
            initial=global_step,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            colour="cyan",
        )

    while global_step < max_iters:
        # ── LR ──────────────────────────────────────────────────────────
        if isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = _scheduler(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

        # ── Forward / backward / accumulate ─────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        micro_loss_accum = 0.0

        for _ in range(accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            scaled  = loss / accum_steps
            micro_loss_accum += scaled.item()
            scaled.backward()

        # ── Clip + step ──────────────────────────────────────────────────
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            scheduler.step()

        global_step += 1

        # ── ETA / step-time ──────────────────────────────────────────────
        t_now      = time.perf_counter()
        step_time  = t_now - t_step_start
        t_step_start = t_now
        elapsed    = t_now - t_loop_start
        steps_done = global_step
        steps_left = max_iters - global_step
        eta_secs   = (elapsed / max(steps_done, 1)) * steps_left

        if pbar is not None:
            pbar.set_postfix(
                loss=f"{micro_loss_accum:.4f}",
                lr=f"{current_lr:.2e}",
                eta=_fmt_time(eta_secs),
                refresh=False,
            )
            pbar.update(1)
        elif global_step % log_interval == 0:
            print(
                f"step {global_step:6d}/{max_iters} | "
                f"loss {micro_loss_accum:.4f} | "
                f"lr {current_lr:.3e} | "
                f"step {step_time*1000:.1f}ms | "
                f"ETA {_fmt_time(eta_secs)}"
            )

        # ── Evaluate ─────────────────────────────────────────────────────
        is_eval = (
            global_step % eval_interval == 0
            or global_step == max_iters
        )
        if is_eval:
            metrics = evaluate(model, val_loader, device=device, max_batches=eval_iters)
            history["steps"].append(global_step)
            history["train_loss"].append(micro_loss_accum)
            history["val_loss"].append(metrics["val_loss"])
            history["perplexity"].append(metrics["perplexity"])

            msg = (
                f"\n  ── eval @ step {global_step:,} ──\n"
                f"     train loss : {micro_loss_accum:.4f}\n"
                f"     val   loss : {metrics['val_loss']:.4f}\n"
                f"     perplexity : {metrics['perplexity']:.2f}\n"
                f"     ETA        : {_fmt_time(eta_secs)}\n"
            )
            if pbar is not None:
                pbar.write(msg)
            else:
                print(msg)

            # Save best
            best_val = min(history["val_loss"])
            if metrics["val_loss"] <= best_val:
                _save_checkpoint(
                    model, optimizer, global_step, history,
                    os.path.join(ckpt_dir, "best_model.pt"),
                )

            # Plot after every eval
            train_records = list(zip(history["steps"], history["train_loss"]))
            val_records   = list(zip(history["steps"], history["val_loss"]))
            _plot_loss_curve(
                train_records, val_records,
                save_path=os.path.join(log_dir, "loss_curve.png"),
            )

        # ── Periodic checkpoint ───────────────────────────────────────────
        if save_interval > 0 and global_step % save_interval == 0:
            _save_checkpoint(
                model, optimizer, global_step, history,
                os.path.join(ckpt_dir, "latest_model.pt"),
            )

    # ── Final checkpoint ─────────────────────────────────────────────────
    _save_checkpoint(
        model, optimizer, global_step, history,
        os.path.join(ckpt_dir, "latest_model.pt"),
    )

    if pbar is not None:
        pbar.close()

    total_time = time.perf_counter() - t_loop_start
    print(f"\nTraining complete in {_fmt_time(total_time)}.")
    return history


# ---------------------------------------------------------------------------

def _save_checkpoint(model, optimizer, step, history, path: str) -> None:
    """Internal helper — saves a full checkpoint dict."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "step":                step,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "model_config":        model.config,
        "history":             history,
        # Legacy keys (so --resume still works with the Trainer class)
        "val_loss":    history["val_loss"][-1]   if history["val_loss"]   else float("inf"),
        "train_losses":list(zip(history["steps"], history["train_loss"])) if history["steps"] else [],
        "val_losses":  list(zip(history["steps"], history["val_loss"]))   if history["steps"] else [],
    }
    torch.save(ckpt, path)
    print(f"  ✓ checkpoint saved → {path}")


def load_checkpoint(
    model,
    optimizer,
    path: str,
) -> tuple[int, dict]:
    """
    Load a checkpoint written by ``train()`` or ``Trainer``.

    Parameters
    ----------
    model     : NanoSageLM instance (weights will be loaded in-place)
    optimizer : optimizer instance (state will be restored in-place)
    path      : path to the .pt checkpoint file

    Returns
    -------
    (step, history)
        step    — the global step at which the checkpoint was saved
        history — {"steps", "train_loss", "val_loss", "perplexity"}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    step = ckpt.get("step", 0)

    # Try the new "history" key first; fall back to legacy parallel lists
    if "history" in ckpt:
        history = ckpt["history"]
    else:
        train_records = ckpt.get("train_losses", [])
        val_records   = ckpt.get("val_losses",   [])
        steps      = [r[0] for r in train_records]
        train_loss = [r[1] for r in train_records]
        val_loss   = [r[1] for r in val_records]
        history = {
            "steps":      steps,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "perplexity": [math.exp(min(v, 20.0)) for v in val_loss],
        }
        # If step wasn't explicitly stored, infer from loss history
        if step == 0 and steps:
            step = steps[-1] + 1

    print(
        f"[load_checkpoint] Loaded '{path}' "
        f"(step {step}, val_loss {ckpt.get('val_loss', 'N/A')})"
    )
    return step, history


# ============================================================================
# CLASS API  (original Trainer — kept for backward compatibility with train.py)
# ============================================================================

class Trainer:
    """
    Original class-based training engine.  Used directly by ``train.py``.
    Delegates internally to the functional helpers where possible.
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        train_config: dict,
        checkpoint=None,
    ):
        """
        Args:
            model          : NanoSageLM
            train_dataset  : TextDataset for training
            val_dataset    : TextDataset for validation
            train_config   : dict (see Functional API docstring for keys)
            checkpoint     : optional raw checkpoint dict (from torch.load)
        """
        self.model  = model
        self.config = train_config
        self.device = torch.device(train_config["device"])

        model.to(self.device)

        pin = self.device.type == "cuda"
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_config["batch_size"],
            shuffle=True,
            pin_memory=pin,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=train_config["batch_size"],
            shuffle=False,
            pin_memory=pin,
        )

        betas = train_config.get("betas", (0.9, 0.95))
        self.optimizer = model.configure_optimizers(
            train_config["weight_decay"],
            train_config["learning_rate"],
            betas,
            self.device.type,
        )

        # Training stats
        self.train_losses: list[tuple[int, float]] = []
        self.val_losses:   list[tuple[int, float]] = []
        self.best_val_loss = float("inf")
        self.resume_iter   = 0

        if checkpoint is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.best_val_loss = checkpoint.get("val_loss", float("inf"))
            self.train_losses  = checkpoint.get("train_losses", [])
            self.val_losses    = checkpoint.get("val_losses",   [])
            if self.train_losses:
                self.resume_iter = self.train_losses[-1][0] + 1
            print(
                f"Restored optimizer state. "
                f"Resuming from iter {self.resume_iter} "
                f"(best val loss so far: {self.best_val_loss:.4f})"
            )

        os.makedirs(train_config["checkpoint_dir"], exist_ok=True)
        os.makedirs("nanosage/logs", exist_ok=True)

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        """Estimate train and val loss over ``eval_iters`` batches each."""
        out = {}
        self.model.eval()
        for split, loader in [("train", self.train_loader), ("val", self.val_loader)]:
            n_iters = self.config["eval_iters"]
            losses  = torch.zeros(n_iters)
            it      = iter(loader)
            for k in range(n_iters):
                try:
                    X, Y = next(it)
                except StopIteration:
                    it = iter(loader)
                    X, Y = next(it)
                X, Y = X.to(self.device), Y.to(self.device)
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def save_checkpoint(self, name: str, val_loss: float) -> None:
        """Save model + optimizer state to ``checkpoint_dir/<name>.pt``."""
        ckpt = {
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config":         self.model.config,
            "train_config":         self.config,
            "val_loss":             val_loss,
            "train_losses":         self.train_losses,
            "val_losses":           self.val_losses,
        }
        path = os.path.join(self.config["checkpoint_dir"], f"{name}.pt")
        torch.save(ckpt, path)
        print(f"Saved checkpoint to: {path}")

    def plot_loss_curve(self) -> None:
        """Render and save the loss curve to ``nanosage/logs/loss_curve.png``."""
        _plot_loss_curve(
            self.train_losses,
            self.val_losses,
            save_path="nanosage/logs/loss_curve.png",
        )

    def train(self) -> None:
        """Run the full training loop (iter-based, not epoch-based)."""
        self.model.train()

        max_iters     = self.config["max_iters"]
        eval_interval = self.config["eval_interval"]
        accum_steps   = self.config.get("grad_accumulation_steps", 1)
        grad_clip     = self.config.get("grad_clip", 1.0)

        iter_num   = self.resume_iter
        train_iter = iter(self.train_loader)

        # tqdm bar
        pbar = None
        if _TQDM_AVAILABLE:
            pbar = tqdm(
                total=max_iters,
                initial=iter_num,
                desc="Training",
                unit="step",
                dynamic_ncols=True,
                colour="cyan",
            )

        t0 = time.perf_counter()
        t_loop_start = t0

        print("Training starting…")

        while iter_num < max_iters:
            # 1) LR
            lr = get_cosine_lr_with_warmup(
                iter_num,
                self.config["warmup_iters"],
                self.config["max_iters"],
                self.config["learning_rate"],
                self.config["min_lr"],
            )
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # 2) Forward / backward / accumulate
            self.optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0

            for _ in range(accum_steps):
                try:
                    X, Y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    X, Y = next(train_iter)

                X, Y = X.to(self.device), Y.to(self.device)
                _, loss = self.model(X, Y)
                scaled   = loss / accum_steps
                loss_accum += scaled.item()
                scaled.backward()

            # 3) Clip + step
            if grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()

            # Timing / ETA
            t1       = time.perf_counter()
            dt       = t1 - t0
            t0       = t1
            elapsed  = t1 - t_loop_start
            steps_done = iter_num + 1
            eta_secs = (elapsed / steps_done) * (max_iters - steps_done)

            if pbar is not None:
                pbar.set_postfix(
                    loss=f"{loss_accum:.4f}",
                    lr=f"{lr:.2e}",
                    eta=_fmt_time(eta_secs),
                    refresh=False,
                )
                pbar.update(1)
            elif iter_num % 10 == 0:
                print(
                    f"step {iter_num}: loss {loss_accum:.4f} | "
                    f"lr {lr:.4e} | "
                    f"step {dt*1000:.1f}ms | "
                    f"ETA {_fmt_time(eta_secs)}"
                )

            # 4) Evaluate + checkpoint
            is_eval = (
                iter_num % eval_interval == 0
                or iter_num == max_iters - 1
            )
            if is_eval:
                losses = self.estimate_loss()
                ppl    = math.exp(min(losses["val"], 20.0))
                msg = (
                    f"\n  ── eval @ step {iter_num:,} ──\n"
                    f"     train loss : {losses['train']:.4f}\n"
                    f"     val   loss : {losses['val']:.4f}\n"
                    f"     perplexity : {ppl:.2f}\n"
                    f"     ETA        : {_fmt_time(eta_secs)}\n"
                )
                if pbar is not None:
                    pbar.write(msg)
                else:
                    print(msg)

                self.train_losses.append((iter_num, losses["train"]))
                self.val_losses.append((iter_num, losses["val"]))

                if losses["val"] < self.best_val_loss:
                    self.best_val_loss = losses["val"]
                    self.save_checkpoint("best_model", losses["val"])

                self.save_checkpoint("latest_model", losses["val"])
                self.plot_loss_curve()

            iter_num += 1

        if pbar is not None:
            pbar.close()

        total = time.perf_counter() - t_loop_start
        print(f"\nTraining complete in {_fmt_time(total)}.")
