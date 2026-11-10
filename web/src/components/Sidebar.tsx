/** Left navigation: views, conversation list, account. */

import type { Conversation, User } from "../types";
import type { View } from "../App";
import {
  IconBrain, IconChat, IconImage, IconPlus, IconSettings, IconShield, IconSpark, IconTarget, IconTrash,
} from "../icons";

interface Props {
  open: boolean;
  view: View;
  setView: (v: View) => void;
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  user: User;
  onLogout: () => void;
  pendingApprovals: number;
  version: string;
}

export function Sidebar({
  open, view, setView, conversations, activeId, onSelect, onNew, onDelete,
  user, onLogout, pendingApprovals, version,
}: Props) {
  const nav: { key: View; label: string; icon: JSX.Element; badge?: number }[] = [
    { key: "chat", label: "Chat", icon: <IconChat className="icon" /> },
    { key: "media", label: "Media Studio", icon: <IconImage className="icon" /> },
    { key: "goals", label: "Goals", icon: <IconTarget className="icon" /> },
    { key: "memory", label: "Memory", icon: <IconBrain className="icon" /> },
    { key: "forge", label: "Self-Improvement", icon: <IconSpark className="icon" /> },
    { key: "approvals", label: "Approvals", icon: <IconShield className="icon" />, badge: pendingApprovals },
    { key: "settings", label: "Settings", icon: <IconSettings className="icon" /> },
  ];

  return (
    <aside className={`sidebar${open ? "" : " collapsed"}`}>
      <div className="brand">
        <div className="brand-mark">J</div>
        <div className="brand-text">
          <span className="brand-name">JagX AI</span>
          <span className="brand-sub">v{version || "13"} · local-first</span>
        </div>
      </div>

      <div className="sidebar-section">
        <button className="btn primary block" onClick={onNew}>
          <IconPlus size={16} /> New task
        </button>
      </div>

      <div className="sidebar-section" style={{ paddingTop: 0 }}>
        {nav.map((n) => (
          <button
            key={n.key}
            className={`nav-item${view === n.key ? " active" : ""}`}
            onClick={() => setView(n.key)}
          >
            {n.icon}
            <span>{n.label}</span>
            {n.badge ? <span className="nav-badge">{n.badge}</span> : null}
          </button>
        ))}
      </div>

      <div className="sidebar-label" style={{ paddingLeft: 20 }}>Recent</div>
      <div className="conv-list">
        {conversations.length === 0 && (
          <div className="faint small" style={{ padding: "8px 12px" }}>No tasks yet.</div>
        )}
        {conversations.map((c) => (
          <div
            key={c.id}
            className={`conv-item${activeId === c.id && view === "chat" ? " active" : ""}`}
            onClick={() => onSelect(c.id)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => { if (e.key === "Enter") onSelect(c.id); }}
          >
            <span className="conv-title">{c.title}</span>
            <button
              className="conv-del"
              aria-label="Delete task"
              onClick={(e) => { e.stopPropagation(); onDelete(c.id); }}
            >
              <IconTrash size={14} />
            </button>
          </div>
        ))}
      </div>

      <div className="sidebar-foot">
        <div className="user-chip">
          <div className="user-avatar">{user.email.slice(0, 1).toUpperCase()}</div>
          <div style={{ minWidth: 0, flex: 1 }}>
            <div className="user-email">{user.email}</div>
          </div>
          <button className="btn ghost sm" onClick={onLogout}>Sign out</button>
        </div>
      </div>
    </aside>
  );
}
