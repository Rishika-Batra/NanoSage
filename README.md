# NanoSage 🧠

A clean, minimal, educational implementation of a **LLaMA-style GPT** autoregressive language model built from scratch in PyTorch.

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

```text
  _   _                 ____                  
 | \ | | __ _ _ __   ___/ ___|  __ _  __ _  ___ 
 |  \| |/ _` | '_ \ / _ \___ \ / _` |/ _` |/ _ \
 | |\  | (_| | | | | (_) |__) | (_| | (_| |  __/
 |_| \_|\__,_|_| |_|\___/____/ \__,_|\__, |\___|
                                     |___/      
```

NanoSage covers the complete LLM lifecycle: **pretraining → instruction finetuning → evaluation → interactive inference** — all written in pure, readable Python without heavy frameworks.

---

## 📋 Table of Contents

- [What is NanoSage?](#-what-is-nanosage)
- [Architecture Details](#-architecture-details)
- [Quick Start](#-quick-start)
- [Full Usage Guide](#-full-usage-guide)
  - [1. Pretraining](#1-pretraining)
  - [2. Instruction Finetuning](#2-instruction-finetuning)
  - [3. Evaluation](#3-evaluation)
  - [4. Standalone Generation](#4-standalone-generation)
  - [5. Interactive Chat](#5-interactive-chat)
- [Inference API](#-inference-api)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Sample Results](#-sample-results)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🎯 What is NanoSage?

NanoSage is designed to demystify Large Language Models (LLMs) by providing a clean, modular, and educational codebase. It replaces traditional absolute pos-embeddings and standard LayerNorms with modern **LLaMA/Gemma-style components** (RoPE, RMSNorm, SwiGLU) and walks you through:

1. **Custom Tokenization**: Learn subword merges via Byte Pair Encoding (BPE).
2. **Pretraining**: Train the model on the `TinyStories` dataset using a Cosine Warmup learning rate scheduler.
3. **Instruction Finetuning**: Adapt the model to respond to prompts using Stanford Alpaca instruction datasets with loss masking (training only on response tokens).
4. **Evaluation**: Compute perplexity and bits-per-character (BPC).
5. **Interactive Chat**: Run a multi-turn chat assistant with streaming text typing effects and performance tracking.

---

## 🏗️ Architecture Details

NanoSage implements a modernized decoder-only transformer:

| Component | Default Value / Implementation | Description |
| :--- | :--- | :--- |
| **Model Type** | Decoder-Only GPT | Autoregressive text generator |
| **Vocabulary Size** | 8,000 | Custom subwords trained with scratch BPE |
| **Embedding Dimension** | 256 | Dimension of token and representation spaces |
| **Transformer Layers** | 6 | Number of self-attention block repetitions |
| **Attention Heads** | 8 | Multi-head causal self-attention |
| **Context Length** | 512 | Maximum sequence token capacity |
| **Positional Encoding** | **RoPE** (Rotary Positional Embeddings) | Rotate key/query vectors based on distance |
| **Normalization** | **RMSNorm** | Root Mean Square Normalization (no mean subtraction) |
| **Activation Block** | **SwiGLU** | Gated Swish FeedForward projection (`SiLU(xW) * xV`) |

---

## 🚀 Quick Start

Get NanoSage up and running in **four simple commands**:

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/NanoSage.git && cd NanoSage

# 2. Install requirements
pip install -r requirements.txt

# 3. Start pretraining (TinyStories dataset will download automatically)
python train.py --epochs 1 --batch_size 32 --lr 3e-4

# 4. Launch the interactive chat shell
python chat.py
```

*Requirements: Python 3.9+, PyTorch 2.0+, datasets, tqdm, matplotlib, numpy.*

---

## 🛠️ Full Usage Guide

### 1. Pretraining

Pretraining trains the model on `TinyStories` (50,000 short stories). It automatically splits text, trains a custom BPE tokenizer, builds the model, and plots loss curves.

```bash
# Pretrain for 5 epochs
python train.py --epochs 5 --batch_size 32 --lr 3e-4
```

To resume pretraining from a specific checkpont:
```bash
python train.py --resume nanosage/checkpoints/latest_model.pt
```

Checkpoints are saved to `nanosage/checkpoints/`. The final pretrain weights are written to `nanosage/checkpoints/nanosage_final.pt`.

---

### 2. Instruction Finetuning

Downloads `yahma/alpaca-cleaned` (5,000 instruction-following examples) and finetunes the model. 

```bash
python finetune.py --epochs 3 --batch_size 16 --lr 1e-4
```

To run instruction finetuning on top of a custom pretrained checkpoint:
```bash
python finetune.py --checkpoint nanosage/checkpoints/nanosage_final.pt
```

Finetuning employs **loss masking**: it formats input examples according to the Alpaca template:
```text
### Instruction:
{instruction}

### Input:
{input} (skipped if empty)

### Response:
{output}<|endoftext|>
```
Gradients are calculated **only** on the response tokens (indicated by the `loss_mask`), preventing the model from wasting capacity trying to predict instructions.

---

### 3. Evaluation

Assess model performance (loss, perplexity, bits-per-character, and token accuracy) across train/val splits:

```bash
python evaluate.py --checkpoint nanosage/checkpoints/nanosage_instruct.pt
```

---

### 4. Standalone Generation

To generate text completions using the command line:

```bash
# Stochastic sampling with repetition penalty
python sample.py --checkpoint nanosage/checkpoints/best_model.pt --prompt "Once upon a time" --temperature 0.8 --top_p 0.9

# Greedy / deterministic decoding
python sample.py --checkpoint nanosage/checkpoints/best_model.pt --prompt "Once upon a time" --temperature 0.0
```

---

### 5. Interactive Chat

Launch the interactive chat shell to talk to the model with multi-turn history tracking (up to the last 3 turns) and an animated typing effect:

```bash
python chat.py
```

#### Slash Commands:
* `/clear` — Clears conversation history.
* `/config` — Displays current generation config.
* `/temp N` — Dynamically changes temperature (e.g. `/temp 0.6`).
* `/quit` — Gracefully exits the shell.

---

## 🧠 Inference API

You can import NanoSage components and decode strategies directly in your Python code:

```python
import torch
from nanosage import NanoSageLM, BPETokenizer
from nanosage import greedy_decode, sample_decode, beam_search, batch_generate, SamplingConfig

# 1. Load Model & Tokenizer
ckpt = torch.load("nanosage/checkpoints/best_model.pt", weights_only=False)
model = NanoSageLM(ckpt["model_config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

tokenizer = BPETokenizer()
tokenizer.load("nanosage/checkpoints/tokenizer.json")

# 2. Greedy Decode
greedy_text = greedy_decode(model, tokenizer, "Once upon a time", max_tokens=64)

# 3. Sample Decode (Temperature, Top-K, Top-P, Repetition Penalty)
cfg = SamplingConfig(temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.1)
sample_text = sample_decode(model, tokenizer, "Once upon a time", config=cfg)

# 4. Beam Search (width = 3)
beam_text = beam_search(model, tokenizer, "Once upon a time", beam_width=3, max_tokens=64)

# 5. Parallel Batch Generation
results = batch_generate(
    model, 
    tokenizer, 
    ["Once upon a time", "The small robot"], 
    config=cfg
)
```

---

## 📁 Project Structure

```text
NanoSage/
├── nanosage/
│   ├── __init__.py           # Package facade exporting model/training/inference APIs
│   ├── data/
│   │   ├── raw/              # Raw data text downloads
│   │   └── processed/        # Tokenized .npy train/validation splits
│   ├── tokenizer/
│   │   └── bpe.py            # Byte Pair Encoding tokenizer trained from scratch
│   ├── model/
│   │   ├── config.py         # ModelConfig, TrainingConfig, GenerationConfig structures
│   │   ├── attention.py      # RoPE, RMSNorm, SwiGLU, and CausalSelfAttention classes
│   │   └── transformer.py    # NanoSageLM full autoregressive transformer definition
│   ├── training/
│   │   ├── dataset.py        # Dataset loaders and pretraining formatters
│   │   ├── trainer.py        # Unified training engine loop
│   │   └── scheduler.py      # CosineWarmupScheduler with LR plot generation
│   ├── inference/
│   │   ├── generate.py       # Greedy, stochastic sampling, beam search, batch generation
│   │   └── chat.py           # Dialogue turn and Alpaca history formatting
│   ├── checkpoints/          # Saved model weights and configurations
│   └── logs/                 # Plotted training curves and schedules
├── train.py                  # CLI pretraining entrypoint
├── finetune.py               # CLI instruction-finetuning script
├── evaluate.py               # Evaluation benchmark execution script
├── sample.py                 # Standalone generation script
├── chat.py                   # Multi-turn interactive terminal interface
├── requirements.txt          # Python package requirements
└── README.md                 # Project documentation
```

---

## 📖 How It Works

- **`bpe.py`**: Converts raw characters to bytes, counts adjacent pair frequencies, and iteratively merges the most frequent pairs up to the target vocabulary size.
- **`attention.py`**:
  - `RoPE`: Precomputes rotation matrices and applies position-dependent rotations to query and key embeddings.
  - `RMSNorm`: Normalizes vectors using their Root Mean Square value, leaving scale variance to be learned.
  - `SwiGLU`: Splits linear projections in two, applying the SiLU activation function to one half before merging with the other.
- **`scheduler.py`**: Adjusts learning rate: linearly warm up over initial steps, decay down to 10% maximum rate using a cosine wave, and clamp to the floor rate.
- **`trainer.py`**: Performs forward & backward propagation, clips gradients to prevent exploding weights, handles training metrics, and logs progress bars.

---

## 💬 Sample Results

Interaction log from the chat terminal:

```text
User › What is artificial intelligence?
Assistant › Artificial intelligence is when computers or robots are programmed to think, learn, and solve problems like a person.
[Generated 23 tokens in 0.35s | Speed: 65.71 tok/sec]

User › Give me an example.
Assistant › A robot helper cleaning up toys because it has learned what they look like is an example of smart machines helping us.
[Generated 27 tokens in 0.41s | Speed: 65.85 tok/sec]
```

---

## 🗺️ Roadmap

- [ ] **Flash Attention Support**: Native speedups in multi-head attention using PyTorch's scaled dot product.
- [ ] **KV Caching**: Cache key-value states during decoding steps to accelerate generation throughput.
- [ ] **Quantization**: Support float16/bfloat16 precision and quantized representations (INT8).
- [ ] **DPO / RLHF**: Direct Preference Optimization alignment training pipeline.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 💖 Acknowledgements

- Sebastian Raschka's wonderful book [Build an LLM from Scratch](https://github.com/rasbt/LLMs-from-scratch)
- The TinyStories dataset (Eldan & Li, 2023)
- Stanford Alpaca clean instruction set
