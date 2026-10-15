"""Background worker for durable AI jobs."""
from __future__ import annotations
import logging, threading, time
from jagx_api import job_store, memory_store
from jagx_api.chat import generate_response
from jagx_api.config import settings

logger=logging.getLogger("jagx-ai")
_stop=threading.Event()
_thread=None

def _run_one(job):
    if job["kind"] != "chat":
        job_store.fail(job["id"],"Unsupported job kind")
        return
    payload=job["payload"]
    try:
        # Keep user memory scoped to the API-key hash passed by the authenticated route.
        result=generate_response(payload["message"],payload.get("history"),payload.get("memory_context"),
                                 training_consent=bool(payload.get("training_consent",False)))
        if not isinstance(result,dict):
            raise RuntimeError("AI engine returned an invalid result")
        job_store.finish(job["id"],result)
    except Exception as exc:
        logger.exception("Background job %s failed",job["id"])
        job_store.fail(job["id"],f"{type(exc).__name__}: {str(exc)[:1000]}",retry=True)

def _loop():
    while not _stop.is_set():
        try:
            job=job_store.claim_next()
            if job:
                _run_one(job)
                continue
        except Exception:
            logger.exception("Background job worker iteration failed")
        _stop.wait(settings.job_poll_seconds)

def start():
    global _thread
    if _thread and _thread.is_alive(): return
    _stop.clear()
    job_store.init()
    _thread=threading.Thread(target=_loop,name="jagx-job-worker",daemon=True)
    _thread.start()
    logger.info("Durable job worker started; database=%s",settings.jobs_db)

def stop():
    _stop.set()
    if _thread and _thread.is_alive():
        _thread.join(timeout=3)
