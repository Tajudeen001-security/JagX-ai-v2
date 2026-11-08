/** Memory browser — the four layers, with consolidation. */

import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { Memory } from "../types";
import { IconBrain, IconDatabase, IconPlus, IconSpark, IconTrash } from "../icons";

const LAYERS = ["semantic", "episodic", "procedural"] as const;

const LAYER_DESC: Record<string, string> = {
  semantic: "Durable facts about you and your world.",
  episodic: "What happened: run summaries, outcomes, failures. Decays unless reused.",
  procedural: "How to do it: successful approaches and learned skills.",
};

export function MemoryView() {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [stats, setStats] = useState<{ total: number; by_layer: Record<string, number>; embedding_backend: string } | null>(null);
  const [layer, setLayer] = useState<string>("");
  const [content, setContent] = useState("");
  const [newLayer, setNewLayer] = useState("semantic");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");

  const load = useCallback(async () => {
    const r = await api.memories(layer);
    setMemories(r.memories);
    setStats(r.stats);
  }, [layer]);

  useEffect(() => { load(); }, [load]);

  const add = async () => {
    if (!content.trim()) return;
    setBusy(true);
    try { await api.addMemory(content.trim(), newLayer); setContent(""); await load(); }
    finally { setBusy(false); }
  };

  const consolidate = async () => {
    setBusy(true);
    setNote("");
    try {
      const r = await api.consolidate();
      setNote(`Scanned ${r.episodes_scanned} episodes · promoted ${r.promoted} to semantic · decayed ${r.decayed} · pruned ${r.pruned}`);
      await load();
    } finally { setBusy(false); }
  };

  const remove = async (id: string) => { await api.deleteMemory(id); await load(); };

  return (
    <div className="content-inner wide">
      <div className="card">
        <div className="card-head">
          <IconBrain size={20} />
          <div>
            <div className="card-title">Memory</div>
            <div className="card-sub">
              Four layers, because one layer is not memory. Retrieval is hybrid — local embeddings
              blended with keyword overlap — so it works with zero external services.
            </div>
          </div>
          <div className="spacer" />
          <button className="btn sm" onClick={consolidate} disabled={busy}>
            {busy ? <span className="spinner" /> : <IconSpark size={14} />} Consolidate
          </button>
        </div>

        {stats && (
          <div className="grid cols-4 mb-4">
            <div className="stat">
              <div className="stat-label">Total</div>
              <div className="stat-value">{stats.total}</div>
            </div>
            {LAYERS.map((l) => (
              <div key={l} className="stat">
                <div className="stat-label">{l}</div>
                <div className="stat-value">{stats.by_layer[l] || 0}</div>
              </div>
            ))}
          </div>
        )}

        {stats && (
          <div className="row small faint mb-4">
            <IconDatabase size={14} />
            <span>Embedding backend: <span className="mono">{stats.embedding_backend}</span>
              {stats.embedding_backend === "hashed" ? " (built-in, zero dependencies — install sentence-transformers for real vectors)" : " (local sentence-transformers)"}
            </span>
          </div>
        )}

        {note && <div className="alert ok">{note}</div>}

        <div className="row wrap" style={{ alignItems: "flex-end" }}>
          <div style={{ flex: "1 1 300px" }}>
            <label className="field-label">Add a memory</label>
            <input className="input" placeholder="A durable fact worth recalling later"
                   value={content} onChange={(e) => setContent(e.target.value)} />
          </div>
          <div style={{ flex: "0 0 150px" }}>
            <label className="field-label">Layer</label>
            <select className="select" value={newLayer} onChange={(e) => setNewLayer(e.target.value)}>
              {LAYERS.map((l) => <option key={l} value={l}>{l}</option>)}
            </select>
          </div>
          <button className="btn primary" onClick={add} disabled={busy || !content.trim()}>
            <IconPlus size={15} /> Save
          </button>
        </div>
      </div>

      <div className="row wrap mt-4">
        <button className={`btn sm${layer === "" ? " primary" : ""}`} onClick={() => setLayer("")}>All</button>
        {LAYERS.map((l) => (
          <button key={l} className={`btn sm${layer === l ? " primary" : ""}`} onClick={() => setLayer(l)}>{l}</button>
        ))}
      </div>

      <div className="mt-4">
        {memories.length === 0 && <div className="card center muted">No memories in this layer yet.</div>}
        {memories.map((m) => (
          <div key={m.id} className="list-item">
            <span className={`chip ${m.layer === "semantic" ? "accent" : m.layer === "procedural" ? "violet" : "info"}`}>{m.layer}</span>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="small">{m.content}</div>
              <div className="faint small mt-2">
                {m.tags ? `tags: ${m.tags} · ` : ""}strength {m.strength.toFixed(2)}
              </div>
            </div>
            <button className="btn ghost sm" onClick={() => remove(m.id)} aria-label="Delete memory">
              <IconTrash size={14} />
            </button>
          </div>
        ))}
      </div>

      <div className="card mt-4">
        <div className="card-title mb-4">How the layers work</div>
        {LAYERS.map((l) => (
          <div key={l} className="row" style={{ alignItems: "flex-start", padding: "6px 0" }}>
            <span className={`chip ${l === "semantic" ? "accent" : l === "procedural" ? "violet" : "info"}`}>{l}</span>
            <span className="small muted" style={{ flex: 1 }}>{LAYER_DESC[l]}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
