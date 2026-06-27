import math
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch.nn import functional as F

from .attention import CausalSelfAttention, RMSNorm, FeedForward
from .config import NanoSageConfig

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        if getattr(config, "use_rmsnorm", False):
            self.ln_1 = RMSNorm(config.n_embd)
            self.ln_2 = RMSNorm(config.n_embd)
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
            self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        
        self.attn = CausalSelfAttention(config)
        
        if getattr(config, "ffn_hidden_dim", None) is not None:
            self.mlp = FeedForward(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x):
        # Pre-LayerNorm architecture with residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class NanoSageLM(nn.Module):
    def __init__(self, config: NanoSageConfig):
        super().__init__()
        self.config = config

        if getattr(config, "use_rmsnorm", False):
            ln_f = RMSNorm(config.n_embd)
        else:
            ln_f = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)

        transformer_dict = {
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": ln_f,
        }
        if not getattr(config, "use_rope", False):
            transformer_dict["wpe"] = nn.Embedding(config.block_size, config.n_embd)

        self.transformer = nn.ModuleDict(transformer_dict)
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: tie transformer token embeddings weight and language modeling head weight
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled initialization to residual projections (per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Log parameter count
        print(f"Total parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, "wpe"):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize weights following GPT-2 distribution rules."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        
        if hasattr(self.transformer, "wpe"):
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos) # (T, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)

        if targets is not None:
            # If targets are passed, compute loss as well
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference optimization: only calculate logits for the final token
            logits = self.lm_head(x[:, [-1], :]) # (B, 1, vocab_size)
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Start with all parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Split parameters into decayable (weight matrices in linears/embeddings) and non-decayable (biases, layernorms)
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # Use fused AdamW if supported (requires CUDA and PyTorch 2.0+)
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Decayed parameter tensors: {len(decay_params)} ({num_decay_params:,} parameters)")
        print(f"Non-decayed parameter tensors: {len(nodecay_params)} ({num_nodecay_params:,} parameters)")
        
        fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW: {use_fused}")
        
        return optimizer
