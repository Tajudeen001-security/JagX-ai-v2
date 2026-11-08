/** Approvals — the review queue from JagX Guard. */

import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { Approval } from "../types";
import { IconCheck, IconShield, IconX } from "../icons";

export function ApprovalsView({ onChanged }: { onChanged: () => void }) {
  const [approvals, setApprovals] = useState<Approval[]>([]);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      const r = await api.approvals();
      setApprovals(r.approvals);
    } catch (e) { setError((e as Error).message); }
  }, []);

  useEffect(() => { load(); }, [load]);

  const decide = async (id: string, status: "approved" | "rejected") => {
    await api.decideApproval(id, status);
    await load();
    onChanged();
  };

  const pending = approvals.filter((a) => a.status === "pending");
  const decided = approvals.filter((a) => a.status !== "pending");

  return (
    <div className="content-inner wide">
      <div className="card">
        <div className="card-head">
          <IconShield size={20} />
          <div>
            <div className="card-title">Approvals</div>
            <div className="card-sub">
              Every action passes through JagX Guard before it executes. Anything that leaves the
              machine or spends something waits here for you. The hard boundary is four verbs:
              <strong> send, pay, sign, delete</strong>.
            </div>
          </div>
        </div>

        {error && <div className="alert error">{error}</div>}

        {pending.length === 0 ? (
          <div className="center muted" style={{ padding: 24 }}>
            Nothing waiting for you. Actions that need approval will appear here.
          </div>
        ) : (
          pending.map((a) => (
            <div key={a.id} className="list-item">
              <span className="chip warn">pending</span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="small" style={{ fontWeight: 600 }}>{a.action}</div>
                <div className="faint small mono" style={{ wordBreak: "break-word" }}>{a.detail}</div>
              </div>
              <button className="btn sm primary" onClick={() => decide(a.id, "approved")}>
                <IconCheck size={14} /> Approve
              </button>
              <button className="btn sm danger" onClick={() => decide(a.id, "rejected")}>
                <IconX size={14} /> Reject
              </button>
            </div>
          ))
        )}
      </div>

      {decided.length > 0 && (
        <div className="card">
          <div className="card-title mb-4">History</div>
          {decided.slice(0, 30).map((a) => (
            <div key={a.id} className="list-item">
              <span className={`chip ${a.status === "approved" ? "ok" : "danger"}`}>{a.status}</span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="small">{a.action}</div>
                <div className="faint small mono" style={{ wordBreak: "break-word" }}>{a.detail}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
