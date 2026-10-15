from __future__ import annotations
import hashlib
from typing import Optional
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from jagx_api import auth, conversation_store, job_store, memory_store
from jagx_api.config import settings

router=APIRouter()
class JobRequest(BaseModel):
    message: str=Field(...,min_length=1,max_length=32000)
    conversation_id: Optional[str]=None
    training_consent: bool=False

def _owner(key:str)->str:
    return hashlib.sha256(("jagx-job-v1:"+key).encode()).hexdigest()

def _key(value:Optional[str])->str:
    key=(value or "").strip()
    if not auth.is_valid_key(key):
        raise HTTPException(status_code=401,detail="Invalid or missing API key")
    ok,info=auth.check_rate_limit(key)
    if not ok: raise HTTPException(status_code=429,detail=info)
    return key

@router.post("/jobs",status_code=202)
def create_job(req:JobRequest,x_api_key:Optional[str]=Header(None,alias="x-api-key")):
    key=_key(x_api_key); history=[]
    if req.conversation_id:
        saved=conversation_store.get(key,req.conversation_id,limit=settings.max_history)
        if saved is None: raise HTTPException(status_code=404,detail="Conversation not found")
        history=[{"role":m["role"],"content":m["content"]} for m in saved["messages"]]
    hits=memory_store.recall(hashlib.sha256(("jagx-memory-v1:"+key).encode()).hexdigest(),req.message)
    memory_context="\\n".join(str(hit.get("text","")) for hit in hits) if hits else None
    payload={"message":req.message,"history":history,"memory_context":memory_context,"training_consent":req.training_consent,"conversation_id":req.conversation_id}
    job=job_store.enqueue(_owner(key),"chat",payload)
    return {"ok":True,"job":job,"poll_url":f"/jobs/{job['id']}"}

@router.get("/jobs")
def list_jobs(limit:int=20,x_api_key:Optional[str]=Header(None,alias="x-api-key")):
    return {"ok":True,"jobs":job_store.list_for(_owner(_key(x_api_key)),limit)}

@router.get("/jobs/{job_id}")
def get_job(job_id:str,x_api_key:Optional[str]=Header(None,alias="x-api-key")):
    job=job_store.get(_owner(_key(x_api_key)),job_id)
    if job is None: raise HTTPException(status_code=404,detail="Job not found")
    return {"ok":True,"job":job}

@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id:str,x_api_key:Optional[str]=Header(None,alias="x-api-key")):
    job=job_store.request_cancel(_owner(_key(x_api_key)),job_id)
    if job is None: raise HTTPException(status_code=404,detail="Job not found")
    return {"ok":True,"job":job}
