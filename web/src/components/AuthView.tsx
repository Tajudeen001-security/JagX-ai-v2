/** Sign in / create account. */

import { useState } from "react";
import { api, setToken } from "../api";
import type { User } from "../types";

export function AuthView({ onAuthed }: { onAuthed: (u: User) => void }) {
  const [mode, setMode] = useState<"login" | "signup">("signup");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      const res = mode === "signup" ? await api.signup(email, password) : await api.login(email, password);
      setToken(res.token);
      onAuthed(res.user);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="auth-wrap">
      <form className="auth-card" onSubmit={submit}>
        <div className="auth-mark">J</div>
        <h1 className="auth-title">JagX AI</h1>
        <p className="auth-sub">
          An autonomous agent that plans, uses tools, remembers, and improves itself.
          Local-first — no third-party API key required.
        </p>

        {error && <div className="alert error">{error}</div>}

        <div className="field">
          <label className="field-label" htmlFor="email">Email</label>
          <input
            id="email" className="input" type="email" autoComplete="email" required
            value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com"
          />
        </div>

        <div className="field">
          <label className="field-label" htmlFor="password">Password</label>
          <input
            id="password" className="input" type="password" required minLength={6}
            autoComplete={mode === "signup" ? "new-password" : "current-password"}
            value={password} onChange={(e) => setPassword(e.target.value)} placeholder="At least 6 characters"
          />
        </div>

        <button className="btn primary block" type="submit" disabled={busy}>
          {busy ? <span className="spinner" /> : null}
          {mode === "signup" ? "Create account" : "Sign in"}
        </button>

        <div className="auth-switch">
          {mode === "signup" ? "Already have an account? " : "New to JagX? "}
          <button type="button" onClick={() => { setMode(mode === "signup" ? "login" : "signup"); setError(""); }}>
            {mode === "signup" ? "Sign in" : "Create one"}
          </button>
        </div>
      </form>
    </div>
  );
}
