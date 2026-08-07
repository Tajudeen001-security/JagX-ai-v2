import os
import json
import secrets
import threading

import requests
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# ---------- CONFIG ----------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
# New OpenAI-compatible endpoint (works with Inference Providers)
HF_MODEL_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

KEYS_FILE = "keys.json"
ADMIN_SECRET = os.environ.get("JAGX_ADMIN_SECRET", "change-this-admin-secret")

# Permanent keys that survive redeploys - set this in Render's Environment tab.
# Format: comma-separated, e.g. "jagx-abc123,jagx-def456"
PERMANENT_KEYS = set(
    k.strip() for k in os.environ.get("JAGX_PERMANENT_KEYS", "").split(",") if k.strip()
)

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
    if key in PERMANENT_KEYS:
        return True
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

    # Modern OpenAI-compatible format
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are JagX AI, a helpful assistant."},
            {"role": "user", "content": req.message}
        ],
        "max_tokens": req.max_tokens,
        "temperature": 0.7,
    }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        r = requests.post(HF_MODEL_URL, headers=headers, json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Request to Hugging Face failed: {str(e)}")

    if r.status_code == 503:
        raise HTTPException(status_code=503, detail="Model is loading, try again in \~20 seconds")

    if r.status_code != 200:
        # Return the real error from Hugging Face so you can see it
        raise HTTPException(status_code=502, detail=f"HF API error: {r.text}")

    try:
        data = r.json()
        reply = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        raise HTTPException(status_code=502, detail=f"Unexpected response from HF: {r.text}")

    return {"response": reply}


# ---------- FRONTEND ----------
CHAT_UI_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>JagX AI</title>
  <style>
    body { font-family: -apple-system, sans-serif; background: #0d0d0f; color: #eee; margin: 0; padding: 16px; }
    h1 { font-size: 20px; color: #7c5cff; }
    input, textarea { width: 100%; box-sizing: border-box; padding: 10px; margin: 6px 0; border-radius: 8px; border: 1px solid #333; background: #1a1a1e; color: #eee; }
    button { width: 100%; padding: 12px; background: #7c5cff; color: white; border: none; border-radius: 8px; font-size: 16px; margin-top: 8px; }
    button:disabled { opacity: 0.5; }
    #chatBox { margin-top: 16px; }
    .msg { padding: 10px; border-radius: 8px; margin: 6px 0; }
    .user { background: #23232a; }
    .ai { background: #2a1f4d; }
    label { font-size: 13px; color: #aaa; }
  </style>
</head>
<body>
  <h1>JagX AI 2.0</h1>
  <label>Your JagX API key</label>
  <input id="apiKey" type="text" placeholder="jagx-xxxxxxxxxxxx">
  <label>Message</label>
  <textarea id="message" rows="3" placeholder="Ask JagX something..."></textarea>
  <button id="sendBtn" onclick="sendMessage()">Send</button>
  <div id="chatBox"></div>

  <script>
    async function sendMessage() {
      const key = document.getElementById('apiKey').value.trim();
      const message = document.getElementById('message').value.trim();
      const chatBox = document.getElementById('chatBox');
      const btn = document.getElementById('sendBtn');

      if (!key || !message) { alert('Enter both your API key and a message.'); return; }

      chatBox.innerHTML += `<div class="msg user"><b>You:</b> ${message}</div>`;
      document.getElementById('message').value = '';
      btn.disabled = true;
      btn.innerText = 'Thinking...';

      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'x-api-key': key },
          body: JSON.stringify({ message: message, max_tokens: 200 })
        });
        const data = await res.json();
        if (res.ok) {
          chatBox.innerHTML += `<div class="msg ai"><b>JagX:</b> ${data.response}</div>`;
        } else {
          chatBox.innerHTML += `<div class="msg ai"><b>Error:</b> ${data.detail}</div>`;
        }
      } catch (e) {
        chatBox.innerHTML += `<div class="msg ai"><b>Error:</b> ${e.message}</div>`;
      }

      btn.disabled = false;
      btn.innerText = 'Send';
      window.scrollTo(0, document.body.scrollHeight);
    }
  </script>
</body>
</html>
"""


@app.get("/chat-ui", response_class=HTMLResponse)
def chat_ui():
    return CHAT_UI_HTML


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
