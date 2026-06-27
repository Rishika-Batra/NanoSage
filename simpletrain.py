import torch
from torch.utils.data import Dataset, DataLoader
import sys, os
sys.path.insert(0, '.')
from nanosage.model.transformer import NanoSageLM
from nanosage.model.config import NanoSageConfig
from nanosage.tokenizer.bpe import BPETokenizer

MAX_STEPS = 2000
BLOCK_SIZE = 256
BATCH_SIZE = 8
LR = 3e-4
DEVICE = "cpu"

tokenizer = BPETokenizer()
tokenizer.load("nanosage/checkpoints/tokenizer.json")
print(f"✅ Tokenizer loaded! Vocab: {len(tokenizer.vocab)}")

print("Tokenizing data...")
with open("nanosage/data/raw/tinystories_tiny.txt", "r") as f:
    text = f.read()
tokens = tokenizer.encode(text)
print(f"Total tokens: {len(tokens):,}")

class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size
    def __len__(self):
        return len(self.tokens) - self.block_size
    def __getitem__(self, i):
        x = self.tokens[i:i+self.block_size]
        y = self.tokens[i+1:i+self.block_size+1]
        return x, y

dataset = TextDataset(tokens, BLOCK_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

config = NanoSageConfig(
    vocab_size=len(tokenizer.vocab),
    block_size=BLOCK_SIZE,
    n_layer=4,
    n_head=4,
    n_embd=128,
    ffn_hidden_dim=512
)
model = NanoSageLM(config).to(DEVICE)
print(f"✅ Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

print(f"\nTraining for {MAX_STEPS} steps...")
model.train()
step = 0
for x, y in loader:
    if step >= MAX_STEPS:
        break
    x, y = x.to(DEVICE), y.to(DEVICE)
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 100 == 0:
        print(f"step {step}: loss {loss.item():.4f}")
    step += 1

os.makedirs("nanosage/checkpoints", exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "model_config": config
}, "nanosage/checkpoints/nanosage_v2.pt")
print("✅ Saved to nanosage/checkpoints/nanosage_v2.pt")
