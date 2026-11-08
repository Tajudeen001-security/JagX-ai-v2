export type Role = "user" | "assistant";

export interface Message {
  id: string;
  role: Role;
  content: string;
  created_at: number;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
}

export interface User {
  id: string;
  email: string;
}

export interface Provider {
  name: string;
  model: string;
  local: boolean;
}

export interface ToolSpec {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  category: string;
}

export interface AppConfig {
  app: string;
  version: string;
  providers: Provider[];
  local_first: boolean;
  modes: string[];
  tools: ToolSpec[];
  council: {
    coordinator: { name: string; role: string };
    lanes: { name: string; key: string; role: string; tools: string[] }[];
    critic: { name: string; role: string };
  };
  guard: {
    verdicts: string[];
    hard_boundary: string[];
    always_blocked: string[];
    safe_tools: string[];
    rule_syntax: string;
  };
  forge: {
    loops: { id: string; free: boolean; always_on: boolean; what: string; requirement?: string }[];
    honest_limits: string;
  };
  evals: { checkers: string[]; default_suites: string[] };
  goals: { unit: string; statuses: string[] };
  limits: { max_steps: number; max_tokens: number; critique_passes: number };
}

export interface Memory {
  id: string;
  layer: string;
  content: string;
  tags: string;
  strength: number;
  created_at: number;
}

export interface Goal {
  id: string;
  title: string;
  success_criterion: string;
  status: string;
  priority: number;
  steps_used: number;
  budget: number;
  created_at: number;
  checkpoints?: Checkpoint[];
}

export interface Checkpoint {
  id: string;
  idx: number;
  note: string;
  status: string;
  created_at: number;
}

export interface Lesson {
  id: string;
  kind: string;
  text: string;
  score: number;
  created_at: number;
}

export interface Variant {
  id: string;
  task_class: string;
  name: string;
  body: string;
  wins: number;
  losses: number;
  score: number;
  active: number;
}

export interface ForgeStats {
  runs: number;
  lessons: number;
  variants: number;
  active_variants: number;
  mean_score: number;
  best_score: number;
  loops: Record<string, string>;
}

export interface Approval {
  id: string;
  action: string;
  detail: string;
  status: string;
  created_at: number;
}

export interface EvalSuite {
  id: string;
  name: string;
  cases: { input: string }[];
}

export interface Settings {
  model: string;
  temperature: number;
  max_steps: number;
  system_prompt: string;
  auto_memory: boolean;
  show_activity: boolean;
  critique: boolean;
  council_threshold: number;
  guard_rules: string;
  theme: string;
}

/** A single event from the SSE stream. */
export interface AgentEvent {
  type: string;
  [key: string]: unknown;
}

export interface TimelineEntry {
  id: string;
  kind: string;
  title: string;
  detail?: string;
  expandable?: string;
  tone: "ok" | "warn" | "err" | "run" | "info" | "violet" | "muted";
  time: number;
}
