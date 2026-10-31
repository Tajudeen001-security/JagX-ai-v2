function trimBase(b) { return (b || "").replace(/\/+$/, ""); }

export const api = {
  base: "",
  key: "",
  setKey(k) { this.key = k; },
  setBase(b) { this.base = trimBase(b); },
  _headers() {
    const h = { "Content-Type": "application/json" };
    if (this.key) h["x-api-key"] = this.key;
    return h;
  },
  async get(path) {
    const r = await fetch(this.base + path, { headers: this._headers() });
    if (!r.ok) throw new Error(`${r.status} ${await r.text().catch(() => "")}`.slice(0, 200));
    return r.json();
  },
  async post(path, body) {
    const r = await fetch(this.base + path, { method: "POST", headers: this._headers(), body: JSON.stringify(body) });
    if (!r.ok) throw new Error(`${r.status} ${await r.text().catch(() => "")}`.slice(0, 200));
    return r.json();
  },
  async stream(path, body, onChunk, ctrl) {
    const r = await fetch(this.base + path, {
      method: "POST",
      headers: this._headers(),
      body: JSON.stringify(body),
      signal: ctrl?.signal,
    });
    if (!r.ok || !r.body) throw new Error(`stream ${r.status}`);
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        const t = line.trim();
        if (!t.startsWith("data:")) continue;
        const data = t.slice(5).trim();
        if (!data || data === "[DONE]") continue;
        try {
          const j = JSON.parse(data);
          const piece = j.token ?? j.delta ?? j.text ?? j.content ?? j.chunk ?? (typeof j === "string" ? j : "");
          if (j.error) { onChunk("\n⚠️ " + j.error); break; }
          if (piece) onChunk(piece);
        } catch { onChunk(data); }
      }
    }
  },
};

// attach key from localStorage on load
api.setKey(localStorage.getItem("jagx_api_key") || "");
