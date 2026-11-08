/** Self-Improvement dashboard — the four JagX Forge loops. */

import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { AppConfig, ForgeStats, Lesson, Variant } from "../types";
import { IconBolt, IconChart, IconCheck, IconSpark, IconX } from "../icons";

export function ForgeView({ config }: { config: AppConfig | null }) {
  const [stats, setStats] = useState<ForgeStats | null>(null);
  const [lessons, setLessons] = useState<Lesson[]>([]);
  const [variants, setVariants] = useState<Variant[]>([]);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");
  const [error, setError] = useState("");
  const [dataset, setDataset] = useState<{ count: number; format: string } | null>(null);

  const load = useCallback(async () => {
    try {
      const [s, l, v] = await Promise.all([api.forgeStats(), api.forgeLessons(), api.forgeVariants()]);
      setStats(s); setLessons(l.lessons); setVariants(v.variants);
    } catch (e) { setError((e as Error).message); }
  }, []);

  useEffect(() => { load(); }, [load]);

  const optimize = async () => {
    setBusy(true); setNote(""); setError("");
    try {
      const r = await api.optimize();
      setNote(`Optimisation round complete · promoted ${r.promoted.length} · retired ${r.retired.length}`);
      await load();
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };

  const exportData = async () => {
    setBusy(true); setError("");
    try {
      const r = await api.dataset("messages");
      setDataset({ count: r.count, format: r.format });
      setNote(`Exported ${r.count} training rows. Run scripts/train_lora.py to fine-tune a LoRA locally.`);
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };

  const byClass = variants.reduce<Record<string, Variant[]>>((acc, v) => {
    (acc[v.task_class] ||= []).push(v);
    return acc;
  }, {});

  return (
    <div className="content-inner wide">
      <div className="card">
        <div className="card-head">
          <IconSpark size={20} />
          <div>
            <div className="card-title">Self-Improvement</div>
            <div className="card-sub">
              Four loops, in increasing order of risk and cost. Loops 1–3 are genuinely free and run
              in production. Loop 4 is free software that needs a GPU you own.
            </div>
          </div>
          <div className="spacer" />
          <button className="btn sm" onClick={optimize} disabled={busy}>
            {busy ? <span className="spinner" /> : <IconBolt size={14} />} Run optimisation
          </button>
        </div>

        {stats && (
          <div className="grid cols-4 mb-4">
            <div className="stat">
              <div className="stat-label">Runs scored</div>
              <div className="stat-value">{stats.runs}</div>
            </div>
            <div className="stat">
              <div className="stat-label">Lessons learned</div>
              <div className="stat-value">{stats.lessons}</div>
            </div>
            <div className="stat">
              <div className="stat-label">Mean score</div>
              <div className="stat-value">{stats.mean_score.toFixed(2)}</div>
              <div className="stat-hint">best {stats.best_score.toFixed(2)}</div>
            </div>
            <div className="stat">
              <div className="stat-label">Prompt variants</div>
              <div className="stat-value">{stats.active_variants}</div>
              <div className="stat-hint">{stats.variants} total</div>
            </div>
          </div>
        )}

        {note && <div className="alert ok">{note}</div>}
        {error && <div className="alert error">{error}</div>}

        <div className="grid cols-2">
          {(config?.forge?.loops || []).map((l) => (
            <div key={l.id} className="stat">
              <div className="row">
                <span className={`chip ${l.always_on ? "ok" : "warn"}`}>{l.always_on ? "always on" : "on demand"}</span>
                <span className="chip">{l.free ? "free" : "paid"}</span>
              </div>
              <div className="card-title mt-2" style={{ textTransform: "capitalize" }}>{l.id.replace(/_/g, " ")}</div>
              <div className="small muted mt-2">{l.what}</div>
              {l.requirement && <div className="faint small mt-2">Requires: {l.requirement}</div>}
            </div>
          ))}
        </div>

        {config?.forge?.honest_limits && (
          <div className="alert warn mt-4">{config.forge.honest_limits}</div>
        )}

        <div className="row wrap mt-4">
          <button className="btn" onClick={exportData} disabled={busy}>
            <IconChart size={14} /> Export SFT dataset
          </button>
          {dataset && <span className="small faint">{dataset.count} rows · {dataset.format} format</span>}
        </div>
      </div>

      <div className="card">
        <div className="card-title mb-4">Prompt variants (UCB1 bandit)</div>
        <div className="card-sub mb-4">
          Competing prompts per task class. The bandit explores untried arms, then exploits the
          winner. Every change is versioned and reversible.
        </div>
        {Object.entries(byClass).map(([cls, list]) => (
          <div key={cls} className="mb-4">
            <div className="stat-label mb-4">{cls}</div>
            {list.map((v) => {
              const n = v.wins + v.losses;
              const rate = n ? v.wins / n : 0;
              return (
                <div key={v.id} className="list-item">
                  <span className={`chip ${v.active ? "ok" : ""}`}>{v.active ? "active" : "retired"}</span>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="small" style={{ fontWeight: 600 }}>{v.name}</div>
                    <div className="faint small">{v.body}</div>
                    <div className="bar mt-2" style={{ maxWidth: 220 }}>
                      <div className={`bar-fill ${rate > 0.6 ? "ok" : rate > 0.3 ? "warn" : "danger"}`}
                           style={{ width: `${Math.round(rate * 100)}%` }} />
                    </div>
                  </div>
                  <span className="faint small mono">{v.wins}W / {v.losses}L</span>
                </div>
              );
            })}
          </div>
        ))}
      </div>

      <div className="card">
        <div className="card-title mb-4">Reflection log</div>
        <div className="card-sub mb-4">
          After every run, JagX scores its own trajectory and writes a lesson. Lessons are injected
          into future runs as guidance.
        </div>
        {lessons.length === 0 && <div className="muted small">No lessons yet — run a task in Chat.</div>}
        {lessons.slice(0, 25).map((l) => (
          <div key={l.id} className="list-item">
            <span className={`tl-dot ${l.score >= 0.8 ? "ok" : l.score >= 0.5 ? "warn" : "err"}`} style={{ marginTop: 7 }} />
            <div style={{ flex: 1 }}>
              <div className="small">{l.text}</div>
              <div className="faint small mt-2">score {l.score.toFixed(2)} · {l.kind}</div>
            </div>
            {l.score >= 0.8 ? <IconCheck size={15} /> : <IconX size={15} />}
          </div>
        ))}
      </div>
    </div>
  );
}
