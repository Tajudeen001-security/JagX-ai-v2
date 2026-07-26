import os
import json
import secrets
import threading

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import uvicorn

# ---------- CONFIG ----------
# Small quantized model that can run on free CPU hosting.
# Swap REPO_ID / FILENAME for a different GGUF model if you want.
REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

KEYS_FILE = "keys.json"
ADMIN_SECRET = os.environ.get("JAGX_ADMIN_SECRET", "change-this-admin-secret")

# ---------- APP + MODEL ----------
app = FastAPI(title="JagX AI 2.0")
lock = threading.Lock()

print("Downloading model (first boot only, cached after)...")
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
llm = Llama(model_path=model_path, n_ctx=2048, n_threads=2)
print("Model loaded.")


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

    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are JagX AI, a helpful assistant."},
            {"role": "user", "content": req.message},
        ],
        max_tokens=req.max_tokens,
    )
    reply = output["choices"][0]["message"]["content"]
    return {"response": reply}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
