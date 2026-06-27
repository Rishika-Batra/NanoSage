import math
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch.nn import functional as F

class RoPE(nn.Module):
    """
    Rotary Position Embeddings (RoPE) class.
    Precomputes rotary cosine and sine embeddings for a given dimension and max sequence length,
    and applies them to Query and Key tensors.
    """
    def __init__(self, dim: int, max_seq_len: int = 1024, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # inv_freq = 1.0 / (theta ** (2i / dim)) for i in [0, 1, ..., dim/2 - 1]
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute frequency grid: (max_seq_len, dim // 2)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqss = torch.outer(t, inv_freq)  # (max_seq_len, dim // 2)

        # Duplicate frequency entries to match split-half shape: (max_seq_len, dim)
        emb = torch.cat((freqss, freqss), dim=-1)  # (max_seq_len, dim)

        # Precompute cos and sin embeddings
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the head dimension in half, negate the second half, and concatenate.
        Used to perform rotary matrix multiplications.
        """
        half_dim = self.dim // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """
        Apply Rotary Position Embeddings to query or key tensors.
        Args:
            x (Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim)
            seq_len (int): Length of sequence to apply embeddings for (default: from tensor shape)
        """
        if seq_len is None:
            seq_len = x.shape[2]
        
        # Slice cached arrays up to current sequence length
        # Unsqueeze to align shapes: (1, 1, seq_len, head_dim) for broadcasting
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(1)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(1)
        
        return (x * cos) + (self.rotate_half(x) * sin)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) class.
    Alternative to LayerNorm which scales the hidden states without subtraction of mean.
    Formula: x / rms(x) * weight
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform root-mean-square normalization.
        Args:
            x (Tensor): Input tensor of shape (..., dim)
        """
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    """
    Swish Gated Linear Unit (SwiGLU) activation class.
    Consolidates the gate projection (W) and value projection (V) into a single 
    larger linear layer projection mapping dim -> 2 * hidden_dim, and applies:
    SiLU(xW) * (xV)
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, 2 * out_features, bias=bias)
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform SwiGLU activation.
        Args:
            x (Tensor): Input tensor of shape (..., in_features)
        """
        # Single projection, then chunk along the last dimension
        gate, val = self.linear(x).chunk(2, dim=-1)
        return F.silu(gate) * val


class FeedForward(nn.Module):
    """
    SwiGLU-based FeedForward block (often called MLP in modern LLMs like LLaMA/Gemma).
    Projects embedding_dim to ffn_hidden_dim via SwiGLU, then projects back.
    """
    def __init__(self, config):
        super().__init__()
        bias = getattr(config, "bias", True)
        n_embd = getattr(config, "n_embd", getattr(config, "embedding_dim", None))
        ffn_hidden_dim = getattr(config, "ffn_hidden_dim", None)
        dropout = getattr(config, "dropout", 0.0)

        self.swiglu = SwiGLU(n_embd, ffn_hidden_dim, bias=bias)
        self.c_proj = nn.Linear(ffn_hidden_dim, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feedforward network block.
        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, n_embd)
        """
        x = self.swiglu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism with optional RoPE (Rotary Position Embeddings).
    Uses standard scale dot-product attention with support for Flash Attention where supported.
    """
    def __init__(self, config):
        super().__init__()
        n_embd = getattr(config, "n_embd", getattr(config, "embedding_dim", None))
        n_head = getattr(config, "n_head", getattr(config, "num_heads", None))
        block_size = getattr(config, "block_size", getattr(config, "context_length", None))
        bias = getattr(config, "bias", True)
        dropout = getattr(config, "dropout", 0.0)

        assert n_embd % n_head == 0, "embedding dim must be divisible by num_heads"
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        
        # Projections for Q, K, V
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Configure RoPE
        self.use_rope = getattr(config, "use_rope", False)
        if self.use_rope:
            head_dim = n_embd // n_head
            self.rope = RoPE(dim=head_dim, max_seq_len=block_size)
            
        # Check if Flash Attention is supported in the current PyTorch version
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        if not self.flash:
            # Fallback causal mask buffer
            self.register_buffer(
                "bias", 
                torch.tril(torch.ones(block_size, block_size))
                .view(1, 1, block_size, block_size)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for causal self-attention.
        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, n_embd)
        """
        B, T, C = x.size() # Batch size, sequence length, embedding dimension
        
        # Query, Key, Value projection
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape to (Batch, Heads, Seq_Len, Head_Dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        
        # Apply RoPE if enabled
        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)
            
        if self.flash:
            # Flash Attention computes causal scale dot-product attention efficiently
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0.0, 
                is_causal=True
            )
        else:
            # Manual scaled dot-product attention calculation with causal masking
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            
        # Re-assemble head outputs side-by-side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection and residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


if __name__ == "__main__":
    print("Testing attention.py components...")
    
    # 1. Test RoPE
    print("\n--- Testing RoPE ---")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 32
    rope = RoPE(dim=head_dim, max_seq_len=64)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    q_rope = rope(q)
    k_rope = rope(k)
    print(f"Input Q shape:  {q.shape}")
    print(f"Output Q shape: {q_rope.shape}")
    assert q_rope.shape == q.shape, "RoPE output shape mismatch"
    print("RoPE check succeeded!")
    
    # 2. Test RMSNorm
    print("\n--- Testing RMSNorm ---")
    norm_dim = 128
    rmsnorm = RMSNorm(dim=norm_dim)
    x_norm = torch.randn(batch_size, seq_len, norm_dim)
    out_norm = rmsnorm(x_norm)
    print(f"Input shape:  {x_norm.shape}")
    print(f"Output shape: {out_norm.shape}")
    assert out_norm.shape == x_norm.shape, "RMSNorm output shape mismatch"
    # Verify normalization properties
    var_check = out_norm.pow(2).mean(-1)
    print(f"Mean squared normalized values (should be close to 1.0): {var_check[0, 0].item():.4f}")
    assert torch.allclose(var_check, torch.ones_like(var_check), atol=1e-4), "RMSNorm logic check failed"
    print("RMSNorm check succeeded!")
    
    # 3. Test SwiGLU
    print("\n--- Testing SwiGLU ---")
    swiglu = SwiGLU(in_features=64, out_features=128)
    x_swi = torch.randn(batch_size, seq_len, 64)
    out_swi = swiglu(x_swi)
    print(f"Input shape:  {x_swi.shape}")
    print(f"Output shape: {out_swi.shape}")
    assert out_swi.shape == (batch_size, seq_len, 128), "SwiGLU output shape mismatch"
    print("SwiGLU check succeeded!")
    
    # 4. Test FeedForward
    print("\n--- Testing FeedForward ---")
    class DummyConfig:
        n_embd = 64
        ffn_hidden_dim = 256
        dropout = 0.1
        bias = True
    config_ff = DummyConfig()
    ffn = FeedForward(config_ff)
    x_ff = torch.randn(batch_size, seq_len, 64)
    out_ff = ffn(x_ff)
    print(f"Input shape:  {x_ff.shape}")
    print(f"Output shape: {out_ff.shape}")
    assert out_ff.shape == x_ff.shape, "FeedForward output shape mismatch"
    print("FeedForward check succeeded!")
    
    # 5. Test CausalSelfAttention
    print("\n--- Testing CausalSelfAttention ---")
    class DummyAttentionConfig:
        n_embd = 64
        n_head = 4
        dropout = 0.1
        block_size = 64
        bias = True
        use_rope = True
    config_attn = DummyAttentionConfig()
    attn = CausalSelfAttention(config_attn)
    x_attn = torch.randn(batch_size, seq_len, 64)
    out_attn = attn(x_attn)
    print(f"Input shape:  {x_attn.shape}")
    print(f"Output shape: {out_attn.shape}")
    assert out_attn.shape == x_attn.shape, "CausalSelfAttention output shape mismatch"
    print("CausalSelfAttention check succeeded!")
    print("\nAll components tested successfully! 🎉")
