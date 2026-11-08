/** Goals board — the always-on goal runner. */

import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { Goal } from "../types";
import { IconCheck, IconClock, IconPlus, IconTarget, IconTrash } from "../icons";

const STATUS_TONE: Record<string, string> = {
  active: "accent", paused: "warn", blocked: "danger", done: "ok", exhausted: "warn",
};

export function GoalsView() {
  const [goals, setGoals] = useState<Goal[]>([]);
  const [title, setTitle] = useState("");
  const [criterion, setCriterion] = useState("");
  const [busy, setBusy] = useState(false);
  const [advancing, setAdvancing] = useState<string | null>(null);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      const r = await api.goals();
      setGoals(r.goals);
    } catch (e) { setError((e as Error).message); }
  }, []);

  useEffect(() => { load(); }, [load]);

  const create = async () => {
    if (!title.trim()) return;
    setBusy(true);
    try {
      await api.createGoal(title.trim(), criterion.trim());
      setTitle(""); setCriterion("");
      await load();
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };

  const advance = async (id: string) => {
    setAdvancing(id);
    setError("");
    try {
      const token = localStorage.getItem("jagx_token");
      const res = await fetch(`/api/goals/${id}/advance`, {
        method: "POST",
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!res.ok || !res.body) throw new Error(`Advance failed (${res.status})`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";
        for (const p of parts) {
          const line = p.split("\n").find((l) => l.startsWith("data:"));
          if (!line) continue;
          try {
            const ev = JSON.parse(line.slice(5).trim());
            if (ev.type === "error") setError(ev.message);
          } catch { /* skip */ }
        }
      }
      await load();
    } catch (e) { setError((e as Error).message); } finally { setAdvancing(null); }
  };

  const tickAll = async () => {
    setBusy(true);
    try { await api.tickGoals(); await load(); }
    catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };

  const remove = async (id: string) => {
    await api.deleteGoal(id);
    await load();
  };

  const counts = goals.reduce<Record<string, number>>((acc, g) => {
    acc[g.status] = (acc[g.status] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="content-inner wide">
      <div className="card">
        <div className="card-head">
          <IconTarget size={20} />
          <div>
            <div className="card-title">Goals</div>
            <div className="card-sub">
              The unit of work is a goal, not a conversation. Goals advance server-side and survive
              the session — the work continues when nobody is watching.
            </div>
          </div>
          <div className="spacer" />
          <button className="btn sm" onClick={tickAll} disabled={busy}>
            {busy ? <span className="spinner" /> : <IconClock size={14} />} Advance due goals
          </button>
        </div>

        <div className="grid cols-4 mb-4">
          {["active", "paused", "blocked", "done"].map((s) => (
            <div key={s} className="stat">
              <div className="stat-label">{s}</div>
              <div className="stat-value">{counts[s] || 0}</div>
            </div>
          ))}
        </div>

        <div className="row wrap" style={{ alignItems: "flex-end" }}>
          <div style={{ flex: "1 1 240px" }}>
            <label className="field-label">New goal</label>
            <input className="input" placeholder="e.g. Ship the JagX v12 release"
                   value={title} onChange={(e) => setTitle(e.target.value)} />
          </div>
          <div style={{ flex: "1 1 240px" }}>
            <label className="field-label">Success criterion</label>
            <input className="input" placeholder="e.g. Repo pushed and tests green"
                   value={criterion} onChange={(e) => setCriterion(e.target.value)} />
          </div>
          <button className="btn primary" onClick={create} disabled={busy || !title.trim()}>
            <IconPlus size={15} /> Add goal
          </button>
        </div>
      </div>

      {error && <div className="alert error mt-4">{error}</div>}

      <div className="mt-4">
        {goals.length === 0 && (
          <div className="card center muted">No goals yet. Add one above and JagX will start advancing it.</div>
        )}
        {goals.map((g) => (
          <div key={g.id} className="card" style={{ marginBottom: 12 }}>
            <div className="row wrap">
              <span className={`chip ${STATUS_TONE[g.status] || ""}`}>{g.status}</span>
              <div style={{ flex: 1, minWidth: 200 }}>
                <div className="card-title">{g.title}</div>
                {g.success_criterion && <div className="card-sub">Success: {g.success_criterion}</div>}
              </div>
              <span className="faint small mono">{g.steps_used}/{g.budget} steps</span>
              <button className="btn sm" onClick={() => advance(g.id)} disabled={advancing === g.id || g.status === "done"}>
                {advancing === g.id ? <span className="spinner" /> : <IconCheck size={14} />} Advance
              </button>
              <button className="btn sm danger" onClick={() => remove(g.id)} aria-label="Delete goal">
                <IconTrash size={14} />
              </button>
            </div>

            <div className="bar mt-4">
              <div
                className={`bar-fill ${g.status === "done" ? "ok" : g.status === "blocked" ? "danger" : ""}`}
                style={{ width: `${Math.min(100, (g.steps_used / Math.max(1, g.budget)) * 100)}%` }}
              />
            </div>

            {g.checkpoints && g.checkpoints.length > 0 && (
              <div className="mt-4">
                <div className="stat-label mb-4">Checkpoints</div>
                {g.checkpoints.slice(-6).map((c) => (
                  <div key={c.id} className="row" style={{ padding: "4px 0", alignItems: "flex-start" }}>
                    <span className={`tl-dot ${c.status === "ok" ? "ok" : c.status === "blocked" ? "err" : "warn"}`} style={{ marginTop: 7 }} />
                    <span className="small muted" style={{ flex: 1 }}>{c.note}</span>
                    <span className="faint small mono">#{c.idx}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
