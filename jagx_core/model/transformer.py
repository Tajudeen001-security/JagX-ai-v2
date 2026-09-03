from __future__ import annotations
import math
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
from .config import ModelConfig

def _rotate_half(x):
    x1,x2=x[...,:x.shape[-1]//2],x[...,x.shape[-1]//2:]
    return torch.cat((-x2,x1),dim=-1)

def apply_rotary_pos_emb(q,k,cos,sin):
    return (q*cos)+(_rotate_half(q)*sin),(k*cos)+(_rotate_half(k)*sin)

class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-6): super().__init__(); self.eps=eps; self.weight=nn.Parameter(torch.ones(dim))
    def forward(self,x): return self.weight*(x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps))

class RotaryEmbedding(nn.Module):
    def __init__(self,dim,max_seq_len=2048,theta=10000.0):
        super().__init__(); self.register_buffer('inv_freq',1.0/(theta**(torch.arange(0,dim,2).float()/dim)),persistent=False); self._build_cache(max_seq_len)
    def _build_cache(self,n):
        t=torch.arange(n,device=self.inv_freq.device,dtype=self.inv_freq.dtype); f=torch.outer(t,self.inv_freq); e=torch.cat((f,f),-1)
        self.register_buffer('cos_cached',e.cos()[None,None],persistent=False); self.register_buffer('sin_cached',e.sin()[None,None],persistent=False)
    def forward(self,n):
        if n>self.cos_cached.shape[2]: self._build_cache(n)
        return self.cos_cached[:,:,:n],self.sin_cached[:,:,:n]

class CausalSelfAttention(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.n_heads=cfg.n_heads; self.n_kv_heads=cfg.n_kv_heads; self.head_dim=cfg.d_model//cfg.n_heads; self.n_rep=self.n_heads//self.n_kv_heads
        self.q_proj=nn.Linear(cfg.d_model,cfg.n_heads*self.head_dim,bias=False); self.k_proj=nn.Linear(cfg.d_model,cfg.n_kv_heads*self.head_dim,bias=False); self.v_proj=nn.Linear(cfg.d_model,cfg.n_kv_heads*self.head_dim,bias=False); self.o_proj=nn.Linear(cfg.d_model,cfg.d_model,bias=False); self.dropout=nn.Dropout(cfg.dropout)
    def _repeat_kv(self,x):
        if self.n_rep==1:return x
        b,n,t,d=x.shape; return x[:,:,None,:,:].expand(b,n,self.n_rep,t,d).reshape(b,n*self.n_rep,t,d)
    def forward(self,x,cos,sin,past_kv=None,use_cache=False):
        b,t,_=x.shape; q=self.q_proj(x).view(b,t,self.n_heads,self.head_dim).transpose(1,2); k=self.k_proj(x).view(b,t,self.n_kv_heads,self.head_dim).transpose(1,2); v=self.v_proj(x).view(b,t,self.n_kv_heads,self.head_dim).transpose(1,2); q,k=apply_rotary_pos_emb(q,k,cos,sin)
        if past_kv is not None: k=torch.cat([past_kv[0],k],2); v=torch.cat([past_kv[1],v],2)
        present=(k,v) if use_cache else None; k=self._repeat_kv(k); v=self._repeat_kv(v); scores=q@k.transpose(-2,-1)/math.sqrt(self.head_dim); total=scores.size(-1); mask=torch.tril(torch.ones(t,total,device=x.device,dtype=torch.bool),diagonal=total-t); scores=scores.masked_fill(~mask[None,None],torch.finfo(scores.dtype).min); a=self.dropout(F.softmax(scores,-1)); return self.o_proj((a@v).transpose(1,2).contiguous().view(b,t,-1)),present

class SwiGLUMLP(nn.Module):
    def __init__(self,cfg): super().__init__(); self.w1=nn.Linear(cfg.d_model,cfg.d_ff,bias=False); self.w3=nn.Linear(cfg.d_model,cfg.d_ff,bias=False); self.w2=nn.Linear(cfg.d_ff,cfg.d_model,bias=False); self.dropout=nn.Dropout(cfg.dropout)
    def forward(self,x): return self.dropout(self.w2(F.silu(self.w1(x))*self.w3(x)))

class Block(nn.Module):
    def __init__(self,cfg): super().__init__(); self.norm1=RMSNorm(cfg.d_model,cfg.rms_norm_eps) if cfg.use_rms_norm else nn.LayerNorm(cfg.d_model); self.norm2=RMSNorm(cfg.d_model,cfg.rms_norm_eps) if cfg.use_rms_norm else nn.LayerNorm(cfg.d_model); self.attn=CausalSelfAttention(cfg); self.mlp=SwiGLUMLP(cfg)
    def forward(self,x,cos,sin,past_kv=None,use_cache=False):
        h,p=self.attn(self.norm1(x),cos,sin,past_kv,use_cache); return x+h+self.mlp(self.norm2(x+h)),p

class JagXTransformer(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg.validate(); self.token_embedding=nn.Embedding(cfg.vocab_size,cfg.d_model); self.rope=RotaryEmbedding(cfg.d_model//cfg.n_heads,cfg.max_seq_len,cfg.rope_theta); self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)]); self.norm=RMSNorm(cfg.d_model,cfg.rms_norm_eps) if cfg.use_rms_norm else nn.LayerNorm(cfg.d_model); self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False); self.lm_head.weight=self.token_embedding.weight if cfg.tie_embeddings else self.lm_head.weight; self.apply(self._init)
    def _init(self,m):
        if isinstance(m,nn.Linear): nn.init.normal_(m.weight,0,self.cfg.initializer_range); m.bias is not None and nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,self.cfg.initializer_range)
    def forward(self,input_ids,labels=None,past_key_values=None,use_cache=False):
        b,t=input_ids.shape; past_len=past_key_values[0][0].shape[2] if past_key_values else 0; x=self.token_embedding(input_ids); cos,sin=self.rope(past_len+t); cos,sin=cos[:,:,past_len:past_len+t],sin[:,:,past_len:past_len+t]; presents=[] if use_cache else None
        for i,blk in enumerate(self.blocks): x,p=blk(x,cos,sin,past_key_values[i] if past_key_values else None,use_cache); use_cache and presents.append(p)
        logits=self.lm_head(self.norm(x)); loss=None
        if labels is not None: loss=F.cross_entropy(logits[...,:-1,:].contiguous().view(-1,logits.size(-1)),labels[...,1:].contiguous().view(-1),ignore_index=-100)
        return (logits,loss,presents) if use_cache else (logits,loss)
