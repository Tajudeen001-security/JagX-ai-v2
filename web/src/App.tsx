/** JagX v13 — app shell: auth gate, sidebar, view routing, theme. */

import { useCallback, useEffect, useState } from "react";
import { api, getToken, setToken } from "./api";
import type { AppConfig, Conversation, Settings, User } from "./types";
import { AuthView } from "./components/AuthView";
import { Sidebar } from "./components/Sidebar";
import { ChatView } from "./views/ChatView";
import { GoalsView } from "./views/GoalsView";
import { MemoryView } from "./views/MemoryView";
import { MediaView } from "./views/MediaView";
import { ForgeView } from "./views/ForgeView";
import { ApprovalsView } from "./views/ApprovalsView";
import { SettingsView } from "./views/SettingsView";
import { IconMenu, IconMoon, IconSun } from "./icons";

export type View = "chat" | "media" | "goals" | "memory" | "forge" | "approvals" | "settings";

export function App() {
  const [user, setUser] = useState<User | null>(null);
  const [booting, setBooting] = useState(true);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [view, setView] = useState<View>("chat");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [theme, setTheme] = useState<"dark" | "light">(
    () => (localStorage.getItem("jagx_theme") as "dark" | "light") || "dark",
  );
  const [pendingApprovals, setPendingApprovals] = useState(0);

  // --- theme ---------------------------------------------------------------
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("jagx_theme", theme);
  }, [theme]);

  // --- boot ----------------------------------------------------------------
  useEffect(() => {
    (async () => {
      if (!getToken()) {
        setBooting(false);
        return;
      }
      try {
        const [me, cfg, st, convs] = await Promise.all([
          api.me(), api.config(), api.settings(), api.conversations(),
        ]);
        setUser(me);
        setConfig(cfg);
        setSettings(st);
        setConversations(convs.conversations);
        if (convs.conversations.length) setActiveId(convs.conversations[0].id);
      } catch {
        setToken(null);
      } finally {
        setBooting(false);
      }
    })();
  }, []);

  const refreshConversations = useCallback(async () => {
    try {
      const r = await api.conversations();
      setConversations(r.conversations);
      return r.conversations;
    } catch {
      return [];
    }
  }, []);

  const refreshApprovals = useCallback(async () => {
    try {
      const r = await api.approvals("pending");
      setPendingApprovals(r.approvals.length);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    if (user) refreshApprovals();
  }, [user, refreshApprovals, view]);

  const onAuthed = useCallback(async (u: User) => {
    setUser(u);
    const [cfg, st, convs] = await Promise.all([api.config(), api.settings(), api.conversations()]);
    setConfig(cfg);
    setSettings(st);
    setConversations(convs.conversations);
    setActiveId(convs.conversations[0]?.id ?? null);
  }, []);

  const logout = useCallback(() => {
    setToken(null);
    setUser(null);
    setConversations([]);
    setActiveId(null);
  }, []);

  if (booting) {
    return (
      <div className="auth-wrap">
        <div className="row" style={{ gap: 12 }}>
          <span className="spinner" />
          <span className="muted">Starting JagX…</span>
        </div>
      </div>
    );
  }

  if (!user) return <AuthView onAuthed={onAuthed} />;

  const activeConv = conversations.find((c) => c.id === activeId) ?? null;

  return (
    <div className="app">
      <Sidebar
        open={sidebarOpen}
        view={view}
        setView={setView}
        conversations={conversations}
        activeId={activeId}
        onSelect={(id) => { setActiveId(id); setView("chat"); }}
        onNew={async () => {
          const c = await api.createConversation();
          setConversations((prev) => [c, ...prev]);
          setActiveId(c.id);
          setView("chat");
        }}
        onDelete={async (id) => {
          await api.deleteConversation(id);
          const rest = conversations.filter((c) => c.id !== id);
          setConversations(rest);
          if (activeId === id) setActiveId(rest[0]?.id ?? null);
        }}
        user={user}
        onLogout={logout}
        pendingApprovals={pendingApprovals}
        version={config?.version ?? ""}
      />

      <div className="main">
        <div className="topbar">
          <button className="icon-btn" onClick={() => setSidebarOpen((v) => !v)} aria-label="Toggle sidebar">
            <IconMenu />
          </button>
          <div>
            <div className="topbar-title">
              {view === "chat" ? (activeConv?.title || "New task")
                : view === "media" ? "Media Studio"
                : view === "goals" ? "Goals"
                : view === "memory" ? "Memory"
                : view === "forge" ? "Self-Improvement"
                : view === "approvals" ? "Approvals"
                : "Settings"}
            </div>
            <div className="topbar-sub">
              {config?.providers?.length
                ? `${config.providers[0].name} · ${config.providers[0].model}${config.providers[0].local ? " · local" : ""}`
                : "no provider reachable"}
            </div>
          </div>
          <div className="topbar-spacer" />
          <button
            className="icon-btn"
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
            aria-label="Toggle theme"
            title="Toggle theme"
          >
            {theme === "dark" ? <IconSun /> : <IconMoon />}
          </button>
        </div>

        <div className="content">
          {view === "chat" && (
            <ChatView
              conversationId={activeId}
              config={config}
              settings={settings}
              onConversationCreated={async (id) => {
                setActiveId(id);
                await refreshConversations();
              }}
              onApprovalsChanged={refreshApprovals}
            />
          )}
          {view === "media" && <MediaView />}
          {view === "goals" && <GoalsView />}
          {view === "memory" && <MemoryView />}
          {view === "forge" && <ForgeView config={config} />}
          {view === "approvals" && <ApprovalsView onChanged={refreshApprovals} />}
          {view === "settings" && (
            <SettingsView
              settings={settings}
              config={config}
              onSaved={setSettings}
              theme={theme}
              setTheme={setTheme}
            />
          )}
        </div>
      </div>
    </div>
  );
}
