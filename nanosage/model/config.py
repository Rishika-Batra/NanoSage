from dataclasses import dataclass, field
import sys

@dataclass
class ModelConfig:
    vocab_size: int = 8000
    embedding_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    context_length: int = 512
    dropout: float = 0.1
    ffn_hidden_dim: int = 1024  # 4x embedding_dim
    use_rope: bool = True
    use_rmsnorm: bool = True

    # Compatibility properties for older codebase structure
    @property
    def n_embd(self) -> int:
        return self.embedding_dim

    @property
    def n_layer(self) -> int:
        return self.num_layers

    @property
    def n_head(self) -> int:
        return self.num_heads

    @property
    def block_size(self) -> int:
        return self.context_length

    @property
    def bias(self) -> bool:
        return True


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 5
    warmup_steps: int = 500
    grad_clip: float = 1.0
    eval_interval: int = 500
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints/"
    log_dir: str = "logs/"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2


# Keeping NanoSageConfig for backwards compatibility
@dataclass
class NanoSageConfig(ModelConfig):
    """
    Backward-compatibility class matching the previous NanoSageConfig API.
    Maps old parameters (n_embd, n_layer, n_head, block_size, bias) to ModelConfig.
    """
    def __init__(
        self,
        vocab_size: int = 8000,
        block_size: int = 512,
        n_layer: int = 6,
        n_head: int = 8,
        n_embd: int = 256,
        dropout: float = 0.1,
        bias: bool = True,
        ffn_hidden_dim: int = 1024,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=n_embd,
            num_layers=n_layer,
            num_heads=n_head,
            context_length=block_size,
            dropout=dropout,
            ffn_hidden_dim=ffn_hidden_dim,
            use_rope=use_rope,
            use_rmsnorm=use_rmsnorm,
        )
        self.bias_val = bias

    @property
    def bias(self) -> bool:
        return getattr(self, "bias_val", True)


def print_section(title: str, config_obj):
    # Detect if output is interactive tty to support colors safely
    has_color = sys.stdout.isatty()
    
    cyan = "\033[96m" if has_color else ""
    yellow = "\033[93m" if has_color else ""
    green = "\033[92m" if has_color else ""
    blue = "\033[94m" if has_color else ""
    magenta = "\033[95m" if has_color else ""
    reset = "\033[0m" if has_color else ""
    bold = "\033[1m" if has_color else ""
    
    border = f"{cyan}*==================================================*{reset}"
    print(border)
    print(f"{cyan}|{reset} {bold}{title.center(48)}{reset} {cyan}|{reset}")
    print(border)
    
    # Sort keys to make the layout predictable and neat
    for key, val in sorted(config_obj.__dict__.items()):
        if key.startswith('_') or key == 'bias_val':
            continue
        
        # Color values based on their types
        if isinstance(val, bool):
            val_str = f"{green}{val}{reset}"
        elif isinstance(val, (int, float)):
            val_str = f"{blue}{val}{reset}"
        else:
            val_str = f"{yellow}\"{val}\"{reset}"
            
        # Pad the string considering the ANSI escape sequence length if color is active
        # Using format widths on raw values, then replacing them in colored strings
        raw_key_val = f"  • {key:<20} : {val}"
        # Visual printing alignment
        val_padding = 24 - len(key)
        val_padding = max(val_padding, 1)
        print(f"{cyan}|{reset}   • {key:<22}: {val_str:<{32 + (len(val_str) - len(str(val)))}} {cyan}|{reset}")
        
    print(border)
    print()


if __name__ == "__main__":
    import os
    if os.name == 'nt' and sys.stdout.isatty():
        os.system('color')
        
    has_color = sys.stdout.isatty()
    magenta = "\033[95m" if has_color else ""
    bold = "\033[1m" if has_color else ""
    reset = "\033[0m" if has_color else ""
    
    print(f"\n{bold}{magenta}🚀 NanoSage Configurations 🚀{reset}\n")
    print_section("Model Configuration", ModelConfig())
    print_section("Training Configuration", TrainingConfig())
    print_section("Generation Configuration", GenerationConfig())
