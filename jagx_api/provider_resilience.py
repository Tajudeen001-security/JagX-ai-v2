"""Provider health, exponential backoff, and per-provider circuit breakers."""
from __future__ import annotations
import threading, time
from collections import defaultdict

_lock=threading.RLock()
_state=defaultdict(lambda: {"failures":0,"opened_until":0.0,"last_error":None,"last_success":None})

def allow(name:str)->bool:
    with _lock:
        return time.monotonic() >= _state[name]["opened_until"]

def success(name:str)->None:
    with _lock:
        _state[name].update(failures=0,opened_until=0.0,last_error=None,last_success=time.time())

def failure(name:str, *, threshold:int=3, cooldown:float=20.0, error:str="provider failure")->None:
    with _lock:
        s=_state[name]; s["failures"]+=1; s["last_error"]=error[:240]
        if s["failures"]>=threshold:
            # Exponential cooldown, capped to five minutes.
            exponent=min(4,s["failures"]-threshold)
            s["opened_until"]=time.monotonic()+min(300.0,cooldown*(2**exponent))

def snapshot()->dict:
    now=time.monotonic()
    with _lock:
        return {name:{"failures":s["failures"],"circuit":"open" if s["opened_until"]>now else "closed",
                      "retry_after_seconds":max(0,round(s["opened_until"]-now,1)),
                      "last_error":s["last_error"],"last_success_unix":s["last_success"]}
                for name,s in _state.items()}

def reset()->None:
    with _lock: _state.clear()
