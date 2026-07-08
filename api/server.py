import sys
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
sys.path.insert(0, '.')
from nanosage.model.transformer import NanoSageLM
from nanosage.tokenizer.bpe import BPETokenizer
from nanosage.inference.generate import sample_decode, GenerationConfig
from nanosage.inference.chat import NanoSageChat

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None
config = None
gen_config = None

@app.on_event("startup")
def startup_event():
    global model, tokenizer, config, gen_config
    print("Loading NanoSage...")
    ckpt = torch.load("nanosage/checkpoints/nanosage_instruct.pt", map_location="cpu", weights_only=False)
    config = ckpt["model_config"]
    model = NanoSageLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded!")
    tokenizer = BPETokenizer()
    tokenizer.load("nanosage/checkpoints/tokenizer.json")
    eos_id = tokenizer.special_tokens.get("<|endoftext|>")
    gen_config = GenerationConfig(
        max_new_tokens=128,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=eos_id
    )
    print(f"Tokenizer loaded! Vocab: {len(tokenizer.vocab)}")

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.get("/health")
def health():
    return {
        "status": "online",
        "model": "NanoSage",
        "params": f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M"
    }

@app.post("/chat")
def chat(req: ChatRequest):
    chat_manager = NanoSageChat(max_history=3)
    for turn in req.history[-3:]:
        chat_manager.add_turn(turn.get("user", ""), turn.get("assistant", ""))
    prompt = chat_manager.get_formatted_prompt(req.message)
    response = sample_decode(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        config=gen_config,
        device=torch.device("cpu")
    )
    response = response.replace("\x00", "").strip()
    return {"response": response}
