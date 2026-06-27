import json
import os
import torch
from threading import Thread
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel

app = FastAPI(title="NanoSage LLM API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

model = None
tokenizer = None
device = None
model_info = {}

SYSTEM_PROMPT = "You are NanoSage, a helpful and intelligent AI assistant."
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "nanosage-tinyllama/final"
STOP_STRINGS = ["###", "<|endoftext|>", "NanoSage:"]

def clean_response(text):
    text = text.strip()
    # Strip leading "NanoSage: " prefix
    if text.lower().startswith("nanosage:"):
        text = text[text.index(":")+1:].strip()
    for stop in STOP_STRINGS:
        if stop in text:
            text = text[:text.index(stop)].strip()
    return text

def format_prompt(message, history=None):
    prompt = "### System:\n" + SYSTEM_PROMPT + "\n\n"
    if history:
        for turn in history:
            if "user" in turn and "assistant" in turn:
                prompt += "### Instruction:\n" + turn["user"] + "\n\n### Response:\n" + turn["assistant"] + "\n\n"
            elif turn.get("role") == "user":
                prompt += "### Instruction:\n" + turn["content"] + "\n\n"
            elif turn.get("role") == "assistant":
                prompt += "### Response:\n" + turn["content"] + "\n\n"
    prompt += "### Instruction:\n" + message + "\n\n### Response:\n"
    return prompt

@app.on_event("startup")
def startup_event():
    global model, tokenizer, device, model_info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print("Using device:", device)

    lora_path = None
    for p in [LORA_PATH, "../" + LORA_PATH]:
        if os.path.exists(p):
            lora_path = p
            break

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, dtype=dtype, device_map=None)
    base_model.to(device)

    if lora_path:
        print("Loading LoRA weights from:", lora_path)
        model = PeftModel.from_pretrained(base_model, lora_path)
        print("NanoSage (TinyLlama + LoRA) loaded!")
    else:
        model = base_model
        print("Base TinyLlama loaded!")

    model.eval()
    model_info = {"base_model": BASE_MODEL_NAME, "lora_path": lora_path or "not loaded", "device": str(device), "total_params": sum(p.numel() for p in model.parameters())}

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        prompt = format_prompt(req.message, req.history)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True, top_k=50, top_p=0.9, repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return ChatResponse(response=clean_response(tokenizer.decode(new_tokens, skip_special_tokens=True)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _stream_response(message, history):
    prompt = format_prompt(message, history)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    Thread(target=model.generate, kwargs=dict(**inputs, streamer=streamer, max_new_tokens=256, temperature=0.7, do_sample=True, top_k=50, top_p=0.9, repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id)).start()
    async def sse_generator():
        full = ""
        tokens_sent = []
        for token in streamer:
            if not token:
                continue
            full += token
            stop_hit = False
            for stop in STOP_STRINGS:
                if stop in full:
                    # Only send text before the stop string
                    clean = full[:full.index(stop)]
                    # Send only what hasnt been sent yet
                    already_sent = "".join(tokens_sent)
                    new_part = clean[len(already_sent):]
                    if new_part:
                        yield "data: " + json.dumps({"token": new_part}) + "\n\n"
                    stop_hit = True
                    break
            if stop_hit:
                break
            tokens_sent.append(token)
            yield "data: " + json.dumps({"token": token}) + "\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(sse_generator(), media_type="text/event-stream")

@app.get("/stream")
async def stream_get(message: str, history: Optional[str] = None):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    history_list = None
    if history:
        try:
            history_list = json.loads(history)
        except:
            pass
    return _stream_response(message, history_list)

@app.post("/stream")
async def stream_post(req: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return _stream_response(req.message, req.history)

@app.get("/health")
async def health():
    if model is None:
        return {"status": "unhealthy"}
    return {"status": "healthy", "model_name": "NanoSage (TinyLlama + LoRA)", **model_info}
