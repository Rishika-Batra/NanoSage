"""
api/server.py
NanoSage backend using Hugging Face Inference API
"""

import json
import os
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx

app = FastAPI(title="NanoSage LLM API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

SYSTEM_PROMPT = "You are NanoSage, a helpful and intelligent AI assistant built from scratch."
STOP_STRINGS = ["###", "<|endoftext|>", "NanoSage:"]


def clean_response(text: str) -> str:
    text = text.strip()
    if text.lower().startswith("nanosage:"):
        text = text[text.index(":")+1:].strip()
    for stop in STOP_STRINGS:
        if stop in text:
            text = text[:text.index(stop)].strip()
    return text


def format_prompt(message: str, history=None) -> str:
    prompt = f"### System:\n{SYSTEM_PROMPT}\n\n"
    if history:
        for turn in history:
            if "user" in turn and "assistant" in turn:
                prompt += f"### Instruction:\n{turn['user']}\n\n### Response:\n{turn['assistant']}\n\n"
            elif turn.get("role") == "user":
                prompt += f"### Instruction:\n{turn['content']}\n\n"
            elif turn.get("role") == "assistant":
                prompt += f"### Response:\n{turn['content']}\n\n"
    prompt += f"### Instruction:\n{message}\n\n### Response:\n"
    return prompt


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    prompt = format_prompt(req.message, req.history)
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "return_full_text": False,
        }
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(HF_API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"HF API error: {response.text}")
        data = response.json()
        text = data[0]["generated_text"] if isinstance(data, list) else data.get("generated_text", "")
        return ChatResponse(response=clean_response(text))


@app.get("/stream")
async def stream_get(message: str, history: Optional[str] = None):
    history_list = None
    if history:
        try:
            history_list = json.loads(history)
        except Exception:
            pass
    return await _stream_response(message, history_list)


@app.post("/stream")
async def stream_post(req: ChatRequest):
    return await _stream_response(req.message, req.history)


async def _stream_response(message: str, history) -> StreamingResponse:
    prompt = format_prompt(message, history)
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "return_full_text": False,
        },
        "stream": True,
    }

    async def sse_generator():
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", HF_API_URL, headers=headers, json=payload) as response:
                full = ""
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                        token = parsed.get("token", {}).get("text", "")
                        if not token:
                            continue
                        full += token
                        stop_hit = False
                        for stop in STOP_STRINGS:
                            if stop in full:
                                clean = full[:full.index(stop)].strip()
                                already = full[:full.index(stop)-len(token)]
                                new_part = clean[len(already):]
                                if new_part:
                                    yield f"data: {json.dumps({'token': new_part})}\n\n"
                                stop_hit = True
                                break
                        if stop_hit:
                            break
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    except Exception:
                        continue
                yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_name": "NanoSage (TinyLlama via HF API)",
        "model": HF_MODEL,
    }
