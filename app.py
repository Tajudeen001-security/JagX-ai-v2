import os
import json
import secrets
import threading

import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import uvicorn

# ---------- CONFIG ----------
# Free Hugging Face model served via HF's free Inference API.
# This token is YOURS (free HF account) - users never see it, they only get your jagx- keys.
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-1.5B-Instruct"

KEYS_FILE = "keys.json"
ADMIN_SECRET = os.environ.get("JAGX_ADMIN_SECRET", "change-this-admin-secret")

app = FastAPI(title="JagX AI 2.0")
lock = threading.Lock()


# ---------- KEY STORAGE ----------
def load_keys():
    if not os.path.exists(KEYS_FILE):
        with open(KEYS_FILE, "w") as f:
            json.dump({}, f)
    with open(KEYS_FILE, "r") as f:
        return json.load(f)


def save_keys(keys):
    with open(KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)


def is_valid_key(key: str) -> bool:
    keys = load_keys()
    return key in keys and keys[key].get("active", True)


# ---------- ROUTES ----------
class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 200


class CreateKeyRequest(BaseModel):
    owner_label: str
    admin_secret: str


@app.get("/")
def root():
    return {"status": "JagX AI 2.0 is running"}


@app.post("/create-key")
def create_key(req: CreateKeyRequest):
    if req.admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")

    with lock:
        keys = load_keys()
        new_key = "jagx-" + secrets.token_hex(16)
        keys[new_key] = {"owner": req.owner_label, "active": True}
        save_keys(keys)

    return {"api_key": new_key, "owner": req.owner_label}


@app.post("/chat")
def chat(req: ChatRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Server misconfigured: missing HF_TOKEN")

    payload = {
        "inputs": f"<|im_start|>system\nYou are JagX AI, a helpful assistant.<|im_end|>\n<|im_start|>user\n{req.message}<|im_end|>\n<|im_start|>assistant\n",
        "parameters": {"max_new_tokens": req.max_tokens, "return_full_text": False},
    }
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    r = requests.post(HF_MODEL_URL, headers=headers, json=payload, timeout=60)

    if r.status_code == 503:
        raise HTTPException(status_code=503, detail="Model is loading on HF, try again in ~20 seconds")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"HF API error: {r.text}")

    data = r.json()
    reply = data[0]["generated_text"] if isinstance(data, list) else str(data)
    return {"response": reply}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
        
