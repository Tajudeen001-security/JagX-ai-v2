/** Settings — model, agent behaviour, guard rules, architecture reference. */

import { useEffect, useState } from "react";
import { api } from "../api";
import type { AppConfig, Settings } from "../types";
import { IconBolt, IconCode, IconGlobe, IconSearch, IconShield, IconSpark } from "../icons";

interface Props {
  settings: Settings | null;
  config: AppConfig | null;
  onSaved: (s: Settings) => void;
  theme: "dark" | "light";
  setTheme: (t: "dark" | "light") => void;
}

function Switch({ on, onChange, label, desc }: { on: boolean; onChange: (v: boolean) => void; label: string; desc?: string }) {
  return (
    <div className="switch" onClick={() => onChange(!on)} role="switch" aria-checked={on} tabIndex={0}
         onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onChange(!on); } }}>
      <div className={`switch-track${on ? " on" : ""}`}><div className="switch-knob" /></div>
      <div>
        <div className="switch-label">{label}</div>
        {desc && <div className="switch-desc">{desc}</div>}
      </div>
    </div>
  );
}

export function SettingsView({ settings, config, onSaved, theme, setTheme }: Props) {
  const [draft, setDraft] = useState<Settings | null>(settings);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");

  useEffect(() => { setDraft(settings); }, [settings]);

  if (!draft) return <div className="content-inner"><div className="card center muted">Loading…</div></div>;

  const save = async () => {
    setBusy(true); setNote("");
    try {
      const saved = await api.saveSettings(draft);
      onSaved(saved);
      setNote("Settings saved.");
    } finally { setBusy(false); }
  };

  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => setDraft({ ...draft, [k]: v });

  return (
    <div className="content-inner wide">
      <div className="card">
        <div className="card-head">
          <IconBolt size={20} />
          <div>
            <div className="card-title">Model & inference</div>
            <div className="card-sub">
              JagX is local-first. The provider chain puts Ollama, llama.cpp and vLLM ahead of every
              hosted endpoint, so a local runtime never needs a third-party API key.
            </div>
          </div>
        </div>

        <div className="grid cols-2">
          <div className="field">
            <label className="field-label">Model override</label>
            <input className="input" placeholder="leave blank to use the active provider"
                   value={draft.model} onChange={(e) => set("model", e.target.value)} />
            <div className="field-hint">Set a model name to pin a specific model on the active endpoint.</div>
          </div>
          <div className="field">
            <label className="field-label">Temperature · {draft.temperature.toFixed(2)}</label>
            <input type="range" min={0} max={1} step={0.05} style={{ width: "100%" }}
                   value={draft.temperature} onChange={(e) => set("temperature", Number(e.target.value))} />
            <div className="field-hint">Lower is more deterministic; higher is more exploratory.</div>
          </div>
        </div>

        <div className="field">
          <label className="field-label">Max agent steps · {draft.max_steps}</label>
          <input type="range" min={1} max={25} step={1} style={{ width: "100%" }}
                 value={draft.max_steps} onChange={(e) => set("max_steps", Number(e.target.value))} />
          <div className="field-hint">
            How many tool-calling turns the loop may take before it is forced to synthesise an answer.
          </div>
        </div>

        {config?.providers && config.providers.length > 0 && (
          <div className="mt-4">
            <div className="stat-label mb-4">Active provider chain</div>
            {config.providers.map((p, i) => (
              <div key={p.name} className="row" style={{ padding: "4px 0" }}>
                <span className={`chip ${i === 0 ? "accent" : ""}`}>{i === 0 ? "active" : `#${i + 1}`}</span>
                <span className="small mono">{p.name}</span>
                <span className="faint small">{p.model}</span>
                {p.local && <span className="chip ok">local · no key</span>}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-head">
          <IconSpark size={20} />
          <div>
            <div className="card-title">Agent behaviour</div>
            <div className="card-sub">How the loop plans, remembers, and checks its own work.</div>
          </div>
        </div>

        <Switch on={draft.auto_memory} onChange={(v) => set("auto_memory", v)}
                label="Automatic memory"
                desc="Recall relevant long-term memory before each run and store lessons after." />
        <Switch on={draft.critique} onChange={(v) => set("critique", v)}
                label="Self-critique (Verity)"
                desc="Have the critic review each answer against a rubric and request a revision if it fails." />
        <Switch on={draft.show_activity} onChange={(v) => set("show_activity", v)}
                label="Show activity timeline"
                desc="Stream the plan, tool calls, guard verdicts and critique into the UI." />

        <div className="field mt-4">
          <label className="field-label">Custom system prompt</label>
          <textarea className="textarea" rows={4}
                    placeholder="Leave blank to use the built-in JagX system prompt"
                    value={draft.system_prompt} onChange={(e) => set("system_prompt", e.target.value)} />
          <div className="field-hint">
            Appended to the built-in prompt. The self-improvement engine also injects a style
            variant here automatically.
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-head">
          <IconShield size={20} />
          <div>
            <div className="card-title">Guard rules</div>
            <div className="card-sub">
              Natural-language rules, one per line. Evaluated deterministically before the model, so
              the default path is predictable and auditable.
            </div>
          </div>
        </div>

        <div className="field">
          <label className="field-label">Rules</label>
          <textarea className="textarea" rows={5}
                    placeholder={"block send\nreview publish\nallow fetch_url"}
                    value={draft.guard_rules} onChange={(e) => set("guard_rules", e.target.value)} />
          <div className="field-hint">
            Syntax: <span className="mono">block &lt;text&gt;</span> · <span className="mono">review &lt;text&gt;</span> ·{" "}
            <span className="mono">allow &lt;text&gt;</span>. The four-verb hard boundary
            (send, pay, sign, delete) always requires review and cannot be overridden.
          </div>
        </div>

        {config?.guard && (
          <div className="row wrap">
            <span className="stat-label">Hard boundary:</span>
            {config.guard.hard_boundary.map((v) => <span key={v} className="chip warn">{v}</span>)}
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-head">
          <IconCode size={20} />
          <div>
            <div className="card-title">Tools & sandbox</div>
            <div className="card-sub">Every tool works with no third-party API key.</div>
          </div>
        </div>
        <div className="grid cols-2">
          {(config?.tools || []).map((t) => (
            <div key={t.name} className="stat">
              <div className="row">
                {t.category === "web" ? <IconGlobe size={14} /> : t.category === "compute" ? <IconCode size={14} /> : <IconSearch size={14} />}
                <span className="mono small">{t.name}</span>
                <span className="chip">{t.category}</span>
              </div>
              <div className="faint small mt-2">{t.description}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="card-head">
          <div>
            <div className="card-title">Appearance</div>
            <div className="card-sub">Dark and light themes, both fully styled.</div>
          </div>
        </div>
        <div className="row">
          <button className={`btn${theme === "dark" ? " primary" : ""}`} onClick={() => setTheme("dark")}>Dark</button>
          <button className={`btn${theme === "light" ? " primary" : ""}`} onClick={() => setTheme("light")}>Light</button>
        </div>
      </div>

      <div className="row mt-4">
        <button className="btn primary" onClick={save} disabled={busy}>
          {busy ? <span className="spinner" /> : null} Save settings
        </button>
        {note && <span className="small" style={{ color: "var(--ok)" }}>{note}</span>}
      </div>
    </div>
  );
}
