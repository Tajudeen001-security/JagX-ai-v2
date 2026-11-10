"""JagX v12 — JagX Guard: the control plane.

Every action the agent proposes passes through Guard *before* it executes.
Guard returns one of three verdicts:

    allow   read-only or reversible — search, fetch, read memory, run code in the sandbox
    review  needs a human — anything that leaves the machine or spends something
    block   refused outright

The hard boundary is stated as four verbs, in the spirit of the systems we
studied: **send, pay, sign, delete** always require review, and a small set
(credential changes, irreversible destruction) is always blocked.

Rules are natural language, stored per user. Evaluation is deterministic first
(so the default path is predictable and auditable) and model-assisted second.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ALLOW = "allow"
REVIEW = "review"
BLOCK = "block"

# Tools that are read-only or run inside the sandbox: always safe.
# ``goal_step`` is a goal runner's *description* of a step, not a tool call —
# it is allowed unless it matches one of the boundary patterns below.
SAFE_TOOLS = {
    "web_search",
    "fetch_url",
    "calculator",
    "recall",
    "list_memories",
    "run_code",
    "remember",
    "goal_step",
    # multimodal: everything runs locally on this machine and writes only to
    # the user's own gallery, so it is read-only with respect to the outside
    # world. Nothing here sends, pays, signs or deletes.
    "generate_image",
    "edit_image",
    "understand_image",
    "read_document",
    "list_media",
}

# The four verbs that always wait for a human.
HARD_BOUNDARY = ("send", "pay", "sign", "delete")

# Never allowed, regardless of user rules.
ALWAYS_BLOCKED = (
    "change password",
    "change the password",
    "rotate credential",
    "exfiltrate",
    "rm -rf /",
    "drop database",
    "wipe disk",
    "format disk",
    "disable firewall",
    "install keylogger",
)

# Patterns that indicate an outbound or spending action.
_REVIEW_PATTERNS = [
    (r"\b(send|email|message|post|publish|tweet|dm|reply to)\b", "send"),
    (r"\b(pay|purchase|buy|checkout|transfer|invoice|charge|subscribe)\b", "pay"),
    (r"\b(sign|signature|contract|agree to terms|accept terms)\b", "sign"),
    (r"\b(delete|remove|drop|destroy|unlink|revoke|unsubscribe)\b", "delete"),
    (r"\b(api[_ ]?key|token|secret|password|credential)\b", "credential"),
]


@dataclass
class Verdict:
    verdict: str
    reason: str
    matched: str = ""
    rule: str = ""

    @property
    def allowed(self) -> bool:
        return self.verdict == ALLOW

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "matched": self.matched,
            "rule": self.rule,
        }


def _user_rules(raw: str) -> list[str]:
    if not raw:
        return []
    return [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def check(action: str, detail: str = "", *, tool: str = "", user_rules: str = "") -> Verdict:
    """Evaluate one proposed action.

    ``action`` is a short verb phrase ("send email to X"), ``detail`` the full
    payload, ``tool`` the tool name when the action came from a tool call.
    """
    text = f"{action} {detail}".lower()

    # 1) Hard blocks first — these can never be overridden.
    for bad in ALWAYS_BLOCKED:
        if bad in text:
            return Verdict(BLOCK, f"Refused: '{bad}' is never permitted.", matched=bad)

    # 2) User rules: "block <x>" / "review <x>" / "allow <x>".
    for rule in _user_rules(user_rules):
        low = rule.lower()
        m = re.match(r"^(block|review|allow)\s*[:\-]?\s*(.+)$", low)
        if not m:
            continue
        kind, pattern = m.group(1), m.group(2).strip()
        if pattern and pattern in text:
            if kind == "block":
                return Verdict(BLOCK, f"Blocked by your rule: {rule}", matched=pattern, rule=rule)
            if kind == "review":
                return Verdict(REVIEW, f"Needs your approval (your rule: {rule})", matched=pattern, rule=rule)
            return Verdict(ALLOW, f"Allowed by your rule: {rule}", matched=pattern, rule=rule)

    # 3) Safe tools with no outbound verb are allowed.
    if tool in SAFE_TOOLS:
        for pattern, verb in _REVIEW_PATTERNS:
            if re.search(pattern, text):
                if verb == "credential":
                    return Verdict(BLOCK, "Refused: credential handling is out of scope.", matched=verb)
                return Verdict(
                    REVIEW,
                    f"Needs your approval: this looks like a '{verb}' action.",
                    matched=verb,
                )
        return Verdict(ALLOW, "Read-only or sandboxed action.", rule="safe-tool")

    # 4) Unknown tool / action: default to review, never silently allow.
    for pattern, verb in _REVIEW_PATTERNS:
        if re.search(pattern, text):
            if verb == "credential":
                return Verdict(BLOCK, "Refused: credential handling is out of scope.", matched=verb)
            return Verdict(REVIEW, f"Needs your approval: this looks like a '{verb}' action.", matched=verb)

    return Verdict(REVIEW, "Unrecognised action — defaulting to review.", rule="default-review")


def describe() -> dict:
    return {
        "verdicts": [ALLOW, REVIEW, BLOCK],
        "hard_boundary": list(HARD_BOUNDARY),
        "always_blocked": list(ALWAYS_BLOCKED),
        "safe_tools": sorted(SAFE_TOOLS),
        "rule_syntax": "one per line: 'block <text>' | 'review <text>' | 'allow <text>'",
    }
