"""Resumable coding-task runner with a strict test-command allowlist.

This worker deliberately does not run model-suggested shell commands or edit an
arbitrary repository. Configure a trusted, pre-provisioned workspace before
enabling repository mutation in a future version.
"""
from __future__ import annotations
import os, subprocess, time
from pathlib import Path
from jagx_api import job_store
from jagx_api.chat import generate_response
from jagx_api.config import settings

def run(job:dict)->dict:
    jid=job["id"]; payload=job["payload"]; task=payload["message"]
    checkpoint=job.get("checkpoint") or {}
    completed=checkpoint.get("completed_steps",[])
    workspace=os.environ.get("JAGX_CODING_WORKSPACE","").strip()
    if "plan" not in completed:
        if job_store.cancellation_requested(jid): return {"cancelled":True}
        job_store.checkpoint(jid,{"step":"planning","completed_steps":completed})
        result=generate_response("Create a concise, step-by-step software implementation plan for this task. Identify files, tests, risks, and acceptance criteria. Do not claim changes have been made.\n\nTask:\n"+task, payload.get("history"),memory_context=payload.get("memory_context"))
        plan=result.get("response","") if isinstance(result,dict) else str(result)
        completed=completed+["plan"]
        job_store.checkpoint(jid,{"step":"plan_ready","completed_steps":completed,"plan":plan[:16000]})
    else:
        plan=checkpoint.get("plan","")
    if job_store.cancellation_requested(jid): return {"cancelled":True}
    if not workspace:
        return {"status":"needs_workspace","plan":plan,
                "message":"Plan saved. To run tests or change repository files, configure a trusted workspace and explicit repository credentials. No repository files were modified."}
    root=Path(workspace).resolve()
    if not root.is_dir() or not (root/".git").exists():
        raise RuntimeError("JAGX_CODING_WORKSPACE must point to a trusted Git repository checkout")
    if "tests" not in completed:
        job_store.checkpoint(jid,{"step":"testing","completed_steps":completed+["plan"],"plan":plan[:16000]})
        # Fixed command; no task/model-supplied shell text is ever executed.
        proc=subprocess.run(["python","-m","pytest","-q"],cwd=str(root),capture_output=True,text=True,timeout=600,
                            env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"})
        tests={"exit_code":proc.returncode,"stdout":proc.stdout[-12000:],"stderr":proc.stderr[-12000:]}
        job_store.checkpoint(jid,{"step":"tests_finished","completed_steps":completed+["plan","tests"],"plan":plan[:16000],"tests":tests})
        return {"status":"tests_finished","plan":plan,"tests":tests,
                "message":"This worker ran the fixed pytest command. It did not edit files or commit changes."}
    return {"status":"checkpoint_restored","plan":plan,"message":"Saved plan checkpoint restored."}
