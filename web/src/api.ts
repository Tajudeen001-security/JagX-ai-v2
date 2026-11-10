/** JagX v13 — API client. Token in localStorage, SSE via fetch + ReadableStream. */

import type {
  AgentEvent, AppConfig, Approval, Conversation, DocumentResult, EvalSuite, ForgeStats,
  Goal, Lesson, MediaAsset, MediaCapabilities, MediaStats, Memory, Message, Settings,
  UnderstandResult, User, Variant,
} from "./types";

const TOKEN_KEY = "jagx_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}
export function setToken(t: string | null) {
  if (t) localStorage.setItem(TOKEN_KEY, t);
  else localStorage.removeItem(TOKEN_KEY);
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...((init.headers as Record<string, string>) || {}),
  };
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch(`/api${path}`, { ...init, headers });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* keep the default */ }
    throw new ApiError(detail, res.status);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  // --- auth ---------------------------------------------------------------
  signup: (email: string, password: string) =>
    request<{ token: string; user: User }>("/auth/signup", {
      method: "POST", body: JSON.stringify({ email, password }),
    }),
  login: (email: string, password: string) =>
    request<{ token: string; user: User }>("/auth/login", {
      method: "POST", body: JSON.stringify({ email, password }),
    }),
  me: () => request<User>("/auth/me"),

  // --- meta ---------------------------------------------------------------
  config: () => request<AppConfig>("/config"),
  health: () => request<Record<string, unknown>>("/health"),

  // --- conversations ------------------------------------------------------
  conversations: () => request<{ conversations: Conversation[] }>("/conversations"),
  createConversation: () => request<Conversation>("/conversations", { method: "POST" }),
  conversation: (id: string) =>
    request<{ conversation: Conversation; messages: Message[] }>(`/conversations/${id}`),
  renameConversation: (id: string, title: string) =>
    request<{ ok: boolean }>(`/conversations/${id}`, { method: "PATCH", body: JSON.stringify({ title }) }),
  deleteConversation: (id: string) =>
    request<{ ok: boolean }>(`/conversations/${id}`, { method: "DELETE" }),

  // --- memory -------------------------------------------------------------
  memories: (layer = "") =>
    request<{ memories: Memory[]; stats: { total: number; by_layer: Record<string, number>; embedding_backend: string } }>(
      `/memories${layer ? `?layer=${layer}` : ""}`,
    ),
  addMemory: (content: string, layer = "semantic", tags = "") =>
    request<Memory>("/memories", { method: "POST", body: JSON.stringify({ content, layer, tags }) }),
  deleteMemory: (id: string) => request<{ ok: boolean }>(`/memories/${id}`, { method: "DELETE" }),
  consolidate: () =>
    request<{ episodes_scanned: number; promoted: number; decayed: number; pruned: number }>(
      "/memories/consolidate", { method: "POST" },
    ),

  // --- goals --------------------------------------------------------------
  goals: () => request<{ goals: Goal[] }>("/goals"),
  createGoal: (title: string, success_criterion = "", priority = 3, budget = 20) =>
    request<Goal>("/goals", { method: "POST", body: JSON.stringify({ title, success_criterion, priority, budget }) }),
  patchGoal: (id: string, patch: Partial<Goal>) =>
    request<Goal>(`/goals/${id}`, { method: "PATCH", body: JSON.stringify(patch) }),
  deleteGoal: (id: string) => request<{ ok: boolean }>(`/goals/${id}`, { method: "DELETE" }),
  tickGoals: () => request<{ advanced: unknown[] }>("/goals/tick", { method: "POST" }),

  // --- approvals ----------------------------------------------------------
  approvals: (status = "") => request<{ approvals: Approval[] }>(`/approvals${status ? `?status=${status}` : ""}`),
  decideApproval: (id: string, status: "approved" | "rejected") =>
    request<{ ok: boolean }>(`/approvals/${id}`, { method: "POST", body: JSON.stringify({ status }) }),

  // --- forge --------------------------------------------------------------
  forgeStats: () => request<ForgeStats>("/forge/stats"),
  forgeLessons: () => request<{ lessons: Lesson[] }>("/forge/lessons"),
  forgeVariants: (taskClass = "") =>
    request<{ variants: Variant[] }>(`/forge/variants${taskClass ? `?task_class=${taskClass}` : ""}`),
  createVariant: (task_class: string, name: string, body: string) =>
    request<Variant>("/forge/variants", { method: "POST", body: JSON.stringify({ task_class, name, body }) }),
  optimize: (evalId = "") =>
    request<{ promoted: unknown[]; retired: unknown[]; task_classes: unknown[] }>(
      `/forge/optimize${evalId ? `?eval_id=${evalId}` : ""}`, { method: "POST" },
    ),
  dataset: (fmt = "messages") =>
    request<{ format: string; count: number; rows: unknown[] }>(`/forge/dataset?fmt=${fmt}`),

  // --- evals --------------------------------------------------------------
  evals: () => request<{ evals: EvalSuite[] }>("/evals"),
  runEval: (id: string, variantId = "") =>
    request<{ score: number; passed: number; total: number; cases: { input: string; ok: boolean; reason: string }[] }>(
      `/evals/${id}/run${variantId ? `?variant_id=${variantId}` : ""}`, { method: "POST" },
    ),
  compareEval: (id: string) =>
    request<{ ranking: { name: string; task_class: string; score: number }[]; winner: { name: string } | null }>(
      `/evals/${id}/compare`, { method: "POST" },
    ),

  // --- tools & settings ---------------------------------------------------
  tools: () => request<{ tools: { name: string; description: string; category: string }[]; sandbox: Record<string, unknown> }>("/tools"),
  settings: () => request<Settings>("/settings"),
  saveSettings: (values: Partial<Settings>) =>
    request<Settings>("/settings", { method: "PUT", body: JSON.stringify(values) }),

  // --- multimodal ---------------------------------------------------------
  mediaCapabilities: () => request<MediaCapabilities>("/media/capabilities"),
  media: (kind = "") =>
    request<{ media: MediaAsset[]; stats: MediaStats }>(`/media${kind ? `?kind=${kind}` : ""}`),
  mediaDetail: (id: string) => request<MediaAsset>(`/media/${id}`),
  generateImage: (body: {
    prompt: string; width?: number; height?: number; steps?: number;
    seed?: number | null; style?: string; negative?: string;
  }) => request<MediaAsset>("/media/generate", { method: "POST", body: JSON.stringify(body) }),
  editImage: (id: string, body: {
    operation: string; prompt?: string; box?: number[] | null; direction?: string;
    amount?: number; factor?: number; style?: string; tolerance?: number;
    degrees?: number; width?: number; height?: number;
  }) => request<MediaAsset>(`/media/${id}/edit`, { method: "POST", body: JSON.stringify(body) }),
  understandImage: (id: string, question = "") =>
    request<UnderstandResult>(`/media/${id}/understand`, { method: "POST", body: JSON.stringify({ question }) }),
  readDocument: (id: string) => request<DocumentResult>(`/media/${id}/read`, { method: "POST" }),
  deleteMedia: (id: string) => request<{ ok: boolean }>(`/media/${id}`, { method: "DELETE" }),

  /** Multipart upload — bypasses the JSON request helper. */
  uploadMedia: async (file: File): Promise<MediaAsset> => {
    const token = getToken();
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/media/upload", {
      method: "POST",
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      body: form,
    });
    if (!res.ok) {
      let detail = `Upload failed (${res.status})`;
      try {
        const body = await res.json();
        if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      } catch { /* keep default */ }
      throw new ApiError(detail, res.status);
    }
    return (await res.json()) as MediaAsset;
  },
};

/** Absolute URL for an asset's raw bytes (used in <img src>). */
export function mediaUrl(id: string): string {
  return `/api/media/${id}/raw`;
}

/** Absolute URL for an asset's thumbnail. */
export function mediaThumbUrl(id: string): string {
  return `/api/media/${id}/thumb`;
}

/**
 * Stream an agent run. Calls `onEvent` for every SSE event.
 * Returns an abort function.
 */
export function streamChat(
  body: { message: string; conversation_id?: string; mode: string; model?: string },
  onEvent: (e: AgentEvent) => void,
  onError: (msg: string) => void,
  onDone: () => void,
): () => void {
  const controller = new AbortController();
  const token = getToken();

  (async () => {
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        let detail = `Stream failed (${res.status})`;
        try {
          const j = await res.json();
          if (j?.detail) detail = j.detail;
        } catch { /* keep default */ }
        onError(detail);
        onDone();
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";
        for (const part of parts) {
          const line = part.split("\n").find((l) => l.startsWith("data:"));
          if (!line) continue;
          const raw = line.slice(5).trim();
          if (!raw) continue;
          try {
            onEvent(JSON.parse(raw) as AgentEvent);
          } catch { /* skip malformed frame */ }
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        onError((err as Error).message || "Stream error");
      }
    } finally {
      onDone();
    }
  })();

  return () => controller.abort();
}
