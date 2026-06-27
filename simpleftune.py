import torch
from torch.utils.data import Dataset, DataLoader
import sys, os
sys.path.insert(0, '.')
from nanosage.model.transformer import NanoSageLM
from nanosage.tokenizer.bpe import BPETokenizer
from datasets import load_dataset

EPOCHS = 3
BLOCK_SIZE = 256
BATCH_SIZE = 8
LR = 1e-4
DEVICE = "cpu"

tokenizer = BPETokenizer()
tokenizer.load("nanosage/checkpoints/tokenizer.json")

ckpt = torch.load("nanosage/checkpoints/nanosage_v2.pt", map_location="cpu", weights_only=False)
config = ckpt['model_config']
model = NanoSageLM(config)
model.load_state_dict(ckpt['model_state_dict'])
model.to(DEVICE)
print(f"✅ Model loaded!")

print("Downloading Alpaca dataset...")
ds = load_dataset("yahma/alpaca-cleaned", split="train")
examples = list(ds)[:2000]

class InstructDataset(Dataset):
    def __init__(self, examples, tokenizer, block_size):
        self.samples = []
        for ex in examples:
            inp = ex.get("input", "").strip()
            if inp:
                prompt = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{inp}\n\n### Response:\n{ex['output']}"
            else:
                prompt = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
            tokens = tokenizer.encode(prompt)[:block_size]
            if len(tokens) > 10:
                self.samples.append(tokens)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        t = self.samples[i]
        pad = [0] * (BLOCK_SIZE - len(t))
        x = torch.tensor(t[:-1] + pad, dtype=torch.long)
        y = torch.tensor(t[1:] + pad, dtype=torch.long)
        return x, y

print("Preparing dataset...")
dataset = InstructDataset(examples, tokenizer, BLOCK_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"✅ {len(dataset)} examples ready")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for step, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        if step % 50 == 0:
            print(f"Epoch {epoch+1} step {step}: loss {loss.item():.4f}")
    print(f"✅ Epoch {epoch+1} done. Avg loss: {total_loss/len(loader):.4f}")

torch.save({
    "model_state_dict": model.state_dict(),
    "model_config": config
}, "nanosage/checkpoints/nanosage_v2_instruct.pt")
print("✅ Saved to nanosage/checkpoints/nanosage_v2_instruct.pt")
