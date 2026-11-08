/** Chat: streaming answers, live activity timeline, council view, composer. */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, streamChat } from "../api";
import type { AgentEvent, AppConfig, Message, Settings, TimelineEntry } from "../types";
import { Markdown } from "../markdown";
import { Timeline } from "../components/Timeline";
import {
  IconBolt, IconCalc, IconCode, IconCopy, IconGlobe, IconLayers, IconSearch,
  IconSend, IconSpark, IconStop, IconUsers,
} from "../icons";

interface Props {
  conversationId: string | null;
  config: AppConfig | null;
  settings: Settings | null;
  onConversationCreated: (id: string) => void;
  onApprovalsChanged: () => void;
}

let seq = 0;
const nextId = () => `tl_${++seq}`;

function toolIcon(name: string) {
  if (name === "web_search") return <IconSearch size={14} />;
  if (name === "fetch_url") return <IconGlobe size={14} />;
  if (name === "run_code") return <IconCode size={14} />;
  if (name === "calculator") return <IconCalc size={14} />;
  return <IconBolt size={14} />;
}

const STARTERS = [
  {
    icon: <IconSearch size={17} />,
    title: "Research and compare",
    desc: "Research the latest open-weight agent frameworks and compare them in a table.",
  },
  {
    icon: <IconCode size={17} />,
    title: "Build and verify",
    desc: "Write a Python function that parses a CSV, then run it in the sandbox to prove it works.",
  },
  {
    icon: <IconUsers size={17} />,
    title: "Convene the council",
    desc: "Use the council to analyse the trade-offs of local inference versus hosted APIs.",
  },
  {
    icon: <IconLayers size={17} />,
    title: "Plan a long-running goal",
    desc: "Break down shipping a v1 product into a plan I can track as a goal.",
  },
];

export function ChatView({ conversationId, config, settings, onConversationCreated, onApprovalsChanged }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<"loop" | "council">("loop");
  const [running, setRunning] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [timeline, setTimeline] = useState<TimelineEntry[]>([]);
  const [collapsed, setCollapsed] = useState(false);
  const [error, setError] = useState("");
  const [meta, setMeta] = useState<{ task_class?: string; variant?: string; run_id?: string }>({});
  const [councilLanes, setCouncilLanes] = useState<{ lane: string; title: string; status: string; finding?: string }[]>([]);

  const abortRef = useRef<(() => void) | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // --- load conversation ---------------------------------------------------
  useEffect(() => {
    if (!conversationId) { setMessages([]); return; }
    let cancelled = false;
    (async () => {
      try {
        const r = await api.conversation(conversationId);
        if (!cancelled) setMessages(r.messages);
      } catch { /* ignore */ }
    })();
    return () => { cancelled = true; };
  }, [conversationId]);

  // --- autoscroll ----------------------------------------------------------
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, streamText, timeline]);

  // --- autosize textarea ---------------------------------------------------
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [input]);

  const push = useCallback((entry: Omit<TimelineEntry, "id" | "time">) => {
    setTimeline((prev) => [...prev, { ...entry, id: nextId(), time: Date.now() }]);
  }, []);

  const handleEvent = useCallback((e: AgentEvent) => {
    switch (e.type) {
      case "conversation":
        if (typeof e.conversation_id === "string") onConversationCreated(e.conversation_id);
        break;
      case "meta":
        setMeta({
          task_class: e.task_class as string,
          variant: e.variant as string,
          run_id: e.run_id as string,
        });
        break;
      case "run_start":
        push({ kind: "run", title: e.mode === "council" ? "Council convened" : "Run started", tone: "run" });
        break;
      case "plan": {
        const steps = (e.steps as string[]) || [];
        push({
          kind: "plan",
          title: `Plan · ${steps.length} step${steps.length === 1 ? "" : "s"}`,
          detail: steps.join(" → "),
          expandable: steps.map((s, i) => `${i + 1}. ${s}`).join("\n"),
          tone: "info",
        });
        break;
      }
      case "step_start":
        push({ kind: "step", title: `Step ${e.index}`, tone: "muted" });
        break;
      case "tool_call":
        push({
          kind: "tool",
          title: `Calling ${e.name}`,
          detail: JSON.stringify(e.arguments),
          expandable: JSON.stringify(e.arguments, null, 2),
          tone: "run",
        });
        break;
      case "guard": {
        const v = e.verdict as string;
        push({
          kind: "guard",
          title: v === "allow" ? "Guard: allowed" : v === "review" ? "Guard: held for your approval" : "Guard: refused",
          detail: e.reason as string,
          tone: v === "allow" ? "ok" : v === "review" ? "warn" : "err",
        });
        if (v !== "allow") onApprovalsChanged();
        break;
      }
      case "tool_result":
        push({
          kind: "result",
          title: `${e.name} ${e.ok ? "returned" : "failed"}${e.duration_ms ? ` · ${e.duration_ms}ms` : ""}`,
          detail: String(e.output || "").slice(0, 160),
          expandable: String(e.output || ""),
          tone: e.ok ? "ok" : "err",
        });
        break;
      case "council": {
        const phase = e.phase as string;
        if (phase === "decompose") {
          push({ kind: "council", title: "Atlas chose the lanes", detail: (e.lanes as string[])?.join(", "), tone: "violet" });
        } else if (phase === "fanout") {
          push({ kind: "council", title: "Specialists dispatched in parallel", detail: (e.lanes as string[])?.join(", "), tone: "violet" });
          setCouncilLanes(((e.lanes as string[]) || []).map((l) => ({ lane: l, title: l, status: "running" })));
        } else if (phase === "lane") {
          setCouncilLanes((prev) => {
            const rest = prev.filter((p) => p.lane !== e.lane);
            return [...rest, { lane: e.lane as string, title: (e.title as string) || (e.lane as string), status: e.status as string, finding: e.finding as string }];
          });
          push({
            kind: "council",
            title: `${e.title} finished`,
            detail: String(e.finding || "").slice(0, 160),
            expandable: String(e.finding || ""),
            tone: "violet",
          });
        } else if (phase === "synthesis") {
          push({ kind: "council", title: "Atlas synthesising", tone: "violet" });
        }
        break;
      }
      case "critique": {
        const v = e.verdict as string;
        const issues = (e.issues as string[]) || [];
        push({
          kind: "critique",
          title: v === "pass" ? `Verity: passed (${e.score})` : `Verity: revision requested (${e.score})`,
          detail: issues.join("; ") || "no issues",
          expandable: issues.length ? issues.map((i) => `• ${i}`).join("\n") : undefined,
          tone: v === "pass" ? "ok" : "warn",
        });
        break;
      }
      case "token":
        setStreamText((prev) => prev + (e.text as string));
        break;
      case "draft_reset":
        setStreamText("");
        break;
      case "answer":
        setStreamText(e.text as string);
        break;
      case "reflection":
        push({
          kind: "reflection",
          title: `Self-reflection · score ${e.score}`,
          detail: "Lesson written to memory for future runs",
          tone: "violet",
        });
        break;
      case "error":
        setError(String(e.message || "Unknown error"));
        push({ kind: "error", title: "Error", detail: String(e.message || ""), tone: "err" });
        break;
      case "end":
        break;
      default:
        break;
    }
  }, [push, onConversationCreated, onApprovalsChanged]);

  const send = useCallback((text?: string) => {
    const message = (text ?? input).trim();
    if (!message || running) return;

    setError("");
    setInput("");
    setStreamText("");
    setTimeline([]);
    setCouncilLanes([]);
    setMeta({});
    setRunning(true);

    const optimistic: Message = {
      id: `local_${Date.now()}`, role: "user", content: message, created_at: Date.now() / 1000,
    };
    setMessages((prev) => [...prev, optimistic]);

    abortRef.current = streamChat(
      { message, conversation_id: conversationId ?? undefined, mode },
      handleEvent,
      (msg) => setError(msg),
      () => {
        setRunning(false);
        abortRef.current = null;
        setStreamText((final) => {
          if (final.trim()) {
            setMessages((prev) => [
              ...prev,
              { id: `a_${Date.now()}`, role: "assistant", content: final, created_at: Date.now() / 1000 },
            ]);
          }
          return "";
        });
      },
    );
  }, [input, running, conversationId, mode, handleEvent]);

  const stop = useCallback(() => {
    abortRef.current?.();
    abortRef.current = null;
    setRunning(false);
  }, []);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  const providerLabel = useMemo(() => {
    const p = config?.providers?.[0];
    if (!p) return "no provider";
    return `${p.name} · ${p.model}`;
  }, [config]);

  const isEmpty = messages.length === 0 && !streamText && !running;

  return (
    <>
      <div className="content-inner" ref={scrollRef}>
        {isEmpty ? (
          <div className="empty">
            <div className="empty-mark">J</div>
            <h2>What should JagX work on?</h2>
            <p>
              JagX plans, calls tools, remembers across sessions, critiques its own answers, and
              learns from every run. Running on <strong>{providerLabel}</strong>
              {config?.providers?.[0]?.local ? " — a local open-weight model, no API key in use." : "."}
            </p>
            <div className="starters">
              {STARTERS.map((s) => (
                <button key={s.title} className="starter" onClick={() => send(s.desc)}>
                  <span className="starter-icon">{s.icon}</span>
                  <span>
                    <span className="starter-title">{s.title}</span>
                    <span className="starter-desc">{s.desc}</span>
                  </span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((m) => (
              <div key={m.id} className={`msg ${m.role}`}>
                <div className="msg-avatar">{m.role === "user" ? "You" : "J"}</div>
                <div className="msg-body">
                  <div className="msg-role">{m.role === "user" ? "You" : "JagX"}</div>
                  {m.role === "user" ? <div className="prose">{m.content}</div> : <Markdown text={m.content} />}
                </div>
              </div>
            ))}

            {(running || timeline.length > 0) && (
              <div className="msg assistant">
                <div className="msg-avatar">J</div>
                <div className="msg-body">
                  <div className="msg-role">
                    JagX
                    {meta.task_class ? <span className="chip accent" style={{ marginLeft: 8 }}>{meta.task_class}</span> : null}
                    {meta.variant ? <span className="chip" style={{ marginLeft: 6 }}>{meta.variant}</span> : null}
                    {mode === "council" ? <span className="chip violet" style={{ marginLeft: 6 }}>council</span> : null}
                  </div>

                  {councilLanes.length > 0 && (
                    <div className="grid cols-3 mb-4">
                      {councilLanes.map((l) => (
                        <div key={l.lane} className="stat" style={{ padding: 12 }}>
                          <div className="row">
                            <span className={`tl-dot ${l.status === "done" ? "ok" : l.status === "error" ? "err" : "run"}`} style={{ marginTop: 0 }} />
                            <span className="stat-label" style={{ margin: 0 }}>{l.title}</span>
                          </div>
                          <div className="small muted mt-2" style={{ maxHeight: 60, overflow: "hidden" }}>
                            {l.finding ? l.finding.slice(0, 120) : "working…"}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  <Timeline
                    entries={timeline}
                    running={running}
                    collapsed={collapsed}
                    onToggle={() => setCollapsed((v) => !v)}
                  />

                  {streamText && (
                    <div className="prose">
                      <span dangerouslySetInnerHTML={{ __html: "" }} />
                      <Markdown text={streamText} />
                      {running && <span className="cursor" />}
                    </div>
                  )}

                  {!streamText && running && (
                    <div className="row muted small">
                      <span className="spinner" /> <span>Thinking…</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {error && <div className="alert error">{error}</div>}
          </>
        )}
      </div>

      <div className="composer-wrap">
        <div className="composer">
          <div className="composer-box">
            <textarea
              ref={textareaRef}
              className="composer-input"
              placeholder="Ask JagX to research, build, analyse, or run something…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              rows={1}
              disabled={running}
            />
            <div className="composer-bar">
              <div className="mode-toggle" role="group" aria-label="Agent mode">
                <button
                  className={`mode-btn${mode === "loop" ? " active" : ""}`}
                  onClick={() => setMode("loop")}
                  title="Single agent loop with tools"
                >
                  Agent
                </button>
                <button
                  className={`mode-btn${mode === "council" ? " active" : ""}`}
                  onClick={() => setMode("council")}
                  title="Coordinator + parallel specialists + critic"
                >
                  Council
                </button>
              </div>
              <div className="composer-spacer" />
              {running ? (
                <button className="send-btn" onClick={stop} title="Stop" aria-label="Stop">
                  <IconStop size={15} />
                </button>
              ) : (
                <button className="send-btn" onClick={() => send()} disabled={!input.trim()} title="Send" aria-label="Send">
                  <IconSend size={15} />
                </button>
              )}
            </div>
          </div>
          <div className="composer-hint">
            Enter to send · Shift+Enter for a new line · {settings?.critique ? "self-critique on" : "self-critique off"}
          </div>
        </div>
      </div>
    </>
  );
}
