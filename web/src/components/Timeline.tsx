/** The activity timeline — renders the agent's live trajectory. */

import { useState } from "react";
import type { TimelineEntry } from "../types";
import { IconChevron } from "../icons";

function toneClass(tone: TimelineEntry["tone"]): string {
  switch (tone) {
    case "ok": return "tl-dot ok";
    case "warn": return "tl-dot warn";
    case "err": return "tl-dot err";
    case "run": return "tl-dot run";
    case "info": return "tl-dot info";
    case "violet": return "tl-dot violet";
    default: return "tl-dot";
  }
}

function Row({ entry }: { entry: TimelineEntry }) {
  const [open, setOpen] = useState(false);
  const time = new Date(entry.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

  return (
    <div className="tl-item">
      <span className={toneClass(entry.tone)} />
      <div className="tl-main">
        <div className="tl-title">{entry.title}</div>
        {entry.detail && (
          <div
            className={`tl-detail${entry.expandable ? " expandable" : ""}`}
            onClick={() => entry.expandable && setOpen((v) => !v)}
            title={entry.expandable ? "Click to expand" : undefined}
          >
            {entry.detail}
          </div>
        )}
        {open && entry.expandable && <pre className="tl-pre">{entry.expandable}</pre>}
      </div>
      <span className="tl-time">{time}</span>
    </div>
  );
}

export function Timeline({
  entries, running, collapsed, onToggle,
}: {
  entries: TimelineEntry[];
  running: boolean;
  collapsed: boolean;
  onToggle: () => void;
}) {
  if (!entries.length) return null;

  return (
    <div className="timeline">
      <div className="timeline-head">
        <span>{running ? "Working" : "Activity"}</span>
        {running && <span className="spinner" style={{ width: 11, height: 11 }} />}
        <span className="spacer" />
        <span className="faint">{entries.length} step{entries.length === 1 ? "" : "s"}</span>
        <button className="icon-btn" style={{ width: 24, height: 24 }} onClick={onToggle} aria-label="Toggle activity">
          <IconChevron size={14} className={collapsed ? "" : "rot90"} />
        </button>
      </div>
      {!collapsed && (
        <div className="timeline-body">
          {entries.map((e) => <Row key={e.id} entry={e} />)}
        </div>
      )}
    </div>
  );
}
