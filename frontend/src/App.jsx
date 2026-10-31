import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import "./styles.css";

const MODES = [
  { id: "fast", label: "Fast", hint: "Single quick answer" },
  { id: "agents", label: "Agents", hint: "Multi-agent specialist team" },
  { id: "reason", label: "Reason", hint: "Deep step-by-step reasoning" },
];

const SIDEBAR_TABS = ["chats", "jobs", "billing"];

function useLocalState(key, initial) {
  const [v, setV] = useState(() => localStorage.getItem(key) ?? initial);
  useEffect(() => { localStorage.setItem(key, v); }, [key, v]);
  return [v, setV];
}

export default function App() {
  const [apiKey, setApiKey] = useLocalState("jagx_api_key", "");
  const [apiBase, setApiBase] = useLocalState("jagx_api_base", "");
  const [tab, setTab] = useState("chats");
  const [mode, setMode] = useState("agents");
  const [reflect, setReflect] = useState(false);
  const [chats, setChats] = useLocalState("jagx_chats", JSON.stringify([]));
  const [activeChat, setActiveChat] = useState(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [catalog, setCatalog] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const scrollRef = useRef(null);
  const abortRef = useRef(null);

  const chatList = useMemo(() => { try { return JSON.parse(chats) || []; } catch { return []; } }, [chats]);
  const current = chatList.find((c) => c.id === activeChat) || null;

  useEffect(() => {
    api.setKey(apiKey);
    if (apiBase) api.setBase(apiBase);
  }, [apiKey, apiBase]);

  useEffect(() => {
    let alive = true;
    api.get("/health").then((h) => { if (alive) setHealth(h); }).catch(() => setHealth(null));
    return () => { alive = false; };
  }, [apiBase]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [current?.messages?.length, busy]);

  useEffect(() => {
    if (tab === "jobs") refreshJobs();
    if (tab === "billing") api.get("/billing/catalog").then(setCatalog).catch(() => setCatalog(null));
  }, [tab]);

  const persist = useCallback((list) => setChats(JSON.stringify(list)), [setChats]);

  const newChat = () => {
    const c = { id: "c" + Date.now(), title: "New chat", messages: [] };
    const list = [c, ...chatList];
    persist(list);
    setActiveChat(c.id);
  };

  const updateChat = (id, patch) => {
    const list = chatList.map((c) => (c.id === id ? { ...c, ...patch } : c));
    persist(list);
    return list;
  };

  const pushMessage = (id, msg) => {
    const list = chatList.map((c) => (c.id === id ? { ...c, messages: [...(c.messages || []), msg] } : c));
    persist(list);
  };

  const patchLast = (id, role, patch) => {
    const list = chatList.map((c) => {
      if (c.id !== id) return c;
      const msgs = [...c.messages];
      for (let i = msgs.length - 1; i >= 0; i--) {
        if (msgs[i].role === role) { msgs[i] = { ...msgs[i], ...patch }; break; }
      }
      return { ...c, messages: msgs };
    });
    persist(list);
  };

  const refreshJobs = async () => {
    try {
      const [st, list] = await Promise.all([
        api.get("/jobs/worker/status").catch(() => null),
        api.get("/jobs?limit=20").catch(() => null),
      ]);
      setJobs({ status: st, items: list?.jobs || [] });
    } catch { setJobs({ status: null, items: [] }); }
  };

  const send = async (asJob = false) => {
    const text = input.trim();
    if (!text || busy) return;
    let chatId = activeChat;
    if (!chatId) {
      const c = { id: "c" + Date.now(), title: text.slice(0, 40), messages: [] };
      persist([c, ...chatList]);
      chatId = c.id;
      setActiveChat(c.id);
    }
    setInput("");
    pushMessage(chatId, { role: "user", text });
    pushMessage(chatId, { role: "assistant", text: "", streaming: true });
    setBusy(true);
    if (asJob) pushMessage(chatId, { role: "system", text: "Submitted as durable job…" });

    try {
      if (asJob) {
        const j = await api.post("/jobs", { message: text, mode });
        patchLast(chatId, "assistant", { streaming: false, text: `✅ Job queued (id: ${j.job?.id ?? "?"}). Track it in the Jobs panel.` });
        refreshJobs();
      } else {
        const ctrl = new AbortController();
        abortRef.current = ctrl;
        let acc = "";
        await api.stream("/chat", { message: text, mode, reflect, stream: true }, (chunk) => {
          acc += chunk;
          patchLast(chatId, "assistant", { text: acc, streaming: true });
        }, ctrl);
        patchLast(chatId, "assistant", { streaming: false, text: acc || "(no response)" });
      }
      const list = chatListRef.current;
      const c = list.find((x) => x.id === chatId);
      if (c && (c.title === "New chat" || !c.title)) updateChat(chatId, { title: text.slice(0, 40) });
    } catch (e) {
      patchLast(chatId, "assistant", { streaming: false, text: e.name === "AbortError" ? "_Stopped._" : `⚠️ ${e.message}` });
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  };

  const chatListRef = useRef([]);
  useEffect(() => { chatListRef.current = chatList; }, [chatList]);

  const stop = () => abortRef.current?.abort();

  const deleteChat = (id) => {
    const list = chatList.filter((c) => c.id !== id);
    persist(list);
    if (activeChat === id) setActiveChat(null);
  };

  if (!apiKey) return <Setup onDone={setApiKey} />;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="logo">J</span>
          <div>
            <div className="brand-name">JagX AI</div>
            <div className="brand-sub">{health ? `v${health.version} · online` : "offline"}</div>
          </div>
          <button className="icon-btn" onClick={() => setSettingsOpen(!settingsOpen)}>⚙</button>
        </div>

        {settingsOpen && (
          <div className="settings">
            <label>API key</label>
            <input value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
            <label>API base URL (blank = same origin)</label>
            <input value={apiBase} onChange={(e) => setApiBase(e.target.value)} placeholder="https://jagx-ai-v2.onrender.com" />
            <button onClick={() => { setApiKey(""); }}>Sign out</button>
          </div>
        )}

        <button className="new-chat" onClick={newChat}>＋ New chat</button>

        <div className="tabs">
          {SIDEBAR_TABS.map((t) => (
            <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>{t}</button>
          ))}
        </div>

        {tab === "chats" && (
          <div className="chat-list">
            {chatList.length === 0 && <div className="empty">No chats yet</div>}
            {chatList.map((c) => (
              <div key={c.id} className={`chat-item ${c.id === activeChat ? "active" : ""}`} onClick={() => setActiveChat(c.id)}>
                <span className="chat-title">{c.title || "New chat"}</span>
                <button className="icon-btn" onClick={(e) => { e.stopPropagation(); deleteChat(c.id); }}>🗑</button>
              </div>
            ))}
          </div>
        )}

        {tab === "jobs" && (
          <div className="panel">
            <button className="new-chat small" onClick={refreshJobs}>↻ Refresh</button>
            {jobs?.status && <div className="stat">Queue: {jobs.status.queue_depth ?? "?"} · active: {jobs.status.active ?? "?"}</div>}
            {(jobs?.items || []).map((j) => (
              <div key={j.id} className="chat-item">
                <span className={`pill ${j.status}`}>{j.status}</span>
                <span className="chat-title">{(j.payload?.message || j.type || j.id || "").toString().slice(0, 26)}</span>
              </div>
            ))}
            {(!jobs || jobs.items.length === 0) && <div className="empty">No jobs yet — send one with "Run as job".</div>}
          </div>
        )}

        {tab === "billing" && (
          <div className="panel">
            {!catalog && <div className="empty">Billing unavailable</div>}
            {catalog?.tiers && Object.entries(catalog.tiers).map(([k, v]) => (
              <div key={k} className="chat-item"><span className="chat-title">{k}</span><span>₦{v.price ?? v.amount ?? ""}</span></div>
            ))}
            {catalog?.payment && <div className="stat">OPay {catalog.payment.account}</div>}
          </div>
        )}
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="modes">
            {MODES.map((m) => (
              <button key={m.id} className={mode === m.id ? "active" : ""} title={m.hint} onClick={() => setMode(m.id)}>{m.label}</button>
            ))}
          </div>
          <label className="reflect">
            <input type="checkbox" checked={reflect} onChange={(e) => setReflect(e.target.checked)} /> Reflect
          </label>
          <button className="new-chat small" onClick={() => send(true)} disabled={busy}>Run as job</button>
        </header>

        <div className="messages" ref={scrollRef}>
          {!current || (current.messages || []).length === 0 ? (
            <div className="welcome">
              <div className="welcome-logo">J</div>
              <h1>JagX AI</h1>
              <p>Multi-agent assistant — Engineer, Researcher, Architect, Creator, Africa & Media.</p>
              <div className="hints">
                <div className="hint" onClick={() => setInput("Design a Flutter auth flow with JWT")}>Design a Flutter auth flow with JWT</div>
                <div className="hint" onClick={() => setInput("Research the history of mobile money in Kenya")}>Research mobile money in Kenya</div>
                <div className="hint" onClick={() => setInput("How do I accept Naira payments in Lagos with low fees?")}>Accept Naira payments in Lagos</div>
              </div>
            </div>
          ) : (
            (current.messages || []).map((m, i) => (
              <div key={i} className={`msg ${m.role}`}>
                <div className="avatar">{m.role === "user" ? "🧑" : "J"}</div>
                <div className="bubble">{renderContent(m.text)}{m.streaming && <span className="cursor" />}</div>
              </div>
            ))
          )}
          {busy && <div className="typing">JagX is thinking…</div>}
        </div>

        <div className="composer">
          <textarea
            rows={1}
            value={input}
            placeholder="Message JagX…"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
          />
          {busy ? <button className="send stop" onClick={stop}>Stop</button> : <button className="send" onClick={() => send()} disabled={!input.trim()}>➤</button>}
        </div>
      </main>
    </div>
  );
}

function renderContent(text) {
  if (!text) return null;
  const parts = String(text).split(/```(\w*)\n?/);
  return parts.map((p, i) => (i % 2 === 1 ? null : <span key={i} className="text">{p}</span>));
}

function Setup({ onDone }) {
  const [key, setKey] = useState("");
  const [base, setBase] = useState("");
  return (
    <div className="setup">
      <div className="setup-card">
        <div className="welcome-logo">J</div>
        <h1>JagX AI</h1>
        <p>Enter your JagX API key to start.</p>
        <input placeholder="jagx-xxxxxxxx" value={key} onChange={(e) => setKey(e.target.value)} />
        <input placeholder="Backend URL (blank = same origin)" value={base} onChange={(e) => setBase(e.target.value)} />
        <button onClick={() => { if (base) localStorage.setItem("jagx_api_base", base); onDone(key); }}>Start chatting</button>
      </div>
    </div>
  );
}
