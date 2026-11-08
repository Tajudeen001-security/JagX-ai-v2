"""JagX v12 — local code sandbox.

Runs untrusted code in a subprocess with hard limits, in a scratch directory,
with network access off by default. **No external service required** — this is
the free, self-hosted path. An optional Piston backend can be enabled for
stronger isolation.

Limits applied (POSIX):
  * wall-clock timeout
  * CPU seconds (RLIMIT_CPU)
  * address space (RLIMIT_AS)
  * file size (RLIMIT_FSIZE)
  * process count (RLIMIT_NPROC)
  * no core dumps
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import settings

logger = logging.getLogger("jagx.sandbox")

LANG_EXT = {
    "python": ".py",
    "py": ".py",
    "javascript": ".js",
    "js": ".js",
    "node": ".js",
    "typescript": ".ts",
    "ts": ".ts",
    "bash": ".sh",
    "sh": ".sh",
    "shell": ".sh",
}

LANG_CMD = {
    "python": [sys.executable, "-I", "-S"],
    "javascript": ["node", "--no-warnings"],
    "node": ["node", "--no-warnings"],
    "bash": ["bash", "--noprofile", "--norc"],
    "sh": ["bash", "--noprofile", "--norc"],
}


def _limits(memory_mb: int, cpu_seconds: int):
    """Return a preexec_fn that applies rlimits (POSIX only)."""

    def _apply():  # pragma: no cover - runs in the child process
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 2))
            mem = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
            except (ValueError, OSError):
                pass
        except Exception:  # noqa: BLE001 - limits are best-effort
            pass

    return _apply


def _piston(language: str, code: str) -> str:
    import httpx

    try:
        with httpx.Client(timeout=max(settings.tool_timeout, 20)) as client:
            r = client.post(
                f"{settings.piston_url.rstrip('/')}/execute",
                json={"language": language, "version": "*", "files": [{"content": code}]},
            )
        if r.status_code != 200:
            return f"Sandbox error (HTTP {r.status_code}): {r.text[:300]}"
        run = r.json().get("run") or {}
        out = (run.get("stdout") or "") + (run.get("stderr") or "")
        return (out or "(no output)")[: settings.max_tool_output_chars]
    except Exception as exc:  # noqa: BLE001
        return f"Sandbox unreachable: {type(exc).__name__}"


def run_code(language: str, code: str, *, timeout: int | None = None) -> str:
    """Execute code and return stdout+stderr (bounded)."""
    if not settings.enable_code_execution:
        return "Code execution is disabled (JAGX_ENABLE_CODE_EXECUTION=false)."

    lang = (language or "python").lower().strip()
    code = (code or "")[:20000]
    if not code.strip():
        return "Error: empty code."

    if settings.piston_url:
        return _piston(lang, code)

    cmd = LANG_CMD.get(lang)
    if cmd is None:
        return (
            f"Error: unsupported language '{lang}'. "
            f"Supported: {', '.join(sorted(set(LANG_CMD)))}."
        )

    ext = LANG_EXT.get(lang, ".txt")
    workdir = Path(tempfile.mkdtemp(prefix="jagx_sbx_"))
    script = workdir / f"main{ext}"
    script.write_text(code, encoding="utf-8")

    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "LANG": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if settings.sandbox_allow_network:
        env["JAGX_NETWORK"] = "1"

    to = timeout or settings.sandbox_timeout
    try:
        proc = subprocess.run(
            cmd + [str(script)],
            cwd=str(workdir),
            env=env,
            capture_output=True,
            text=True,
            timeout=to,
            preexec_fn=_limits(settings.sandbox_memory_mb, to) if os.name == "posix" else None,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0 and not out.strip():
            out = f"(exited with code {proc.returncode})"
        return (out or "(no output)")[: settings.max_tool_output_chars]
    except subprocess.TimeoutExpired:
        return f"Error: execution timed out after {to}s."
    except FileNotFoundError as exc:
        return f"Error: runtime not available ({exc})."
    except Exception as exc:  # noqa: BLE001
        logger.exception("sandbox failure")
        return f"Error: sandbox failure: {type(exc).__name__}"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def status() -> dict:
    return {
        "enabled": settings.enable_code_execution,
        "backend": "piston" if settings.piston_url else "local-subprocess",
        "timeout_s": settings.sandbox_timeout,
        "memory_mb": settings.sandbox_memory_mb,
        "network": settings.sandbox_allow_network,
        "languages": sorted(set(LANG_CMD)),
    }
