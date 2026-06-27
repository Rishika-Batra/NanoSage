import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')
from nanosage.model.transformer import NanoSageLM
from nanosage.tokenizer.bpe import BPETokenizer

print("Loading NanoSage...")
ckpt = torch.load("nanosage/checkpoints/nanosage_v2_instruct.pt", map_location="cpu", weights_only=False)
config = ckpt['model_config']
model = NanoSageLM(config)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"✅ Model loaded!")

tokenizer = BPETokenizer()
tokenizer.load("nanosage/checkpoints/tokenizer.json")
print(f"✅ Tokenizer loaded!")

def generate(prompt, max_tokens=150, temperature=0.7, top_k=40):
    tokens = tokenizer.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        for _ in range(max_tokens):
            if x.shape[1] >= config.block_size:
                break
            logits, _ = model(x)
            logits = logits[:, -1, :] / temperature
            top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_vals[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)
    new_tokens = x[0].tolist()[len(tokens):]
    return tokenizer.decode(new_tokens)

print("\n╔══════════════════════════════╗")
print("║     🧠 NanoSage Chat v2.0    ║")
print("║   Your tiny AI assistant     ║")
print("╚══════════════════════════════╝\n")

while True:
    user = input("You › ").strip()
    if user.lower() in ["/quit", "/exit", "quit", "exit"]:
        print("Goodbye!")
        break
    prompt = f"### Instruction:\n{user}\n\n### Response:\n"
    response = generate(prompt)
    print(f"NanoSage › {response}\n")
