"""Billing / manual tier upgrade API + admin HTML panel."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from jagx_api import auth, billing
from jagx_api.config import settings

router = APIRouter(prefix="/billing", tags=["billing"])


class UpgradeRequest(BaseModel):
    target_tier: str = Field(..., description="premium | premium_plus")
    payment_ref: str = Field(..., min_length=3, max_length=200)
    amount_ngn: Optional[float] = None
    note: str = ""
    payer_email: str = ""
    payer_name: str = ""


class AdminConfirmRequest(BaseModel):
    request_id: str
    admin_secret: str
    admin_email: str
    action: str = Field(default="confirm", pattern="^(confirm|reject)$")


def _require_key(x_api_key: Optional[str]) -> str:
    key = (x_api_key or "").strip()
    if not auth.is_valid_key(key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    ok, info = auth.check_rate_limit(key)
    if not ok:
        raise HTTPException(status_code=429, detail=info)
    return key


@router.get("/catalog")
def billing_catalog():
    return {"ok": True, **billing.catalog()}


@router.get("/payment-note")
def payment_note(tier: str = "premium"):
    note = billing.payment_note_template(tier)
    if not note.get("ok"):
        raise HTTPException(status_code=400, detail=note.get("error", "bad tier"))
    return note


@router.post("/upgrade-request")
def upgrade_request(
    req: UpgradeRequest,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    key = _require_key(x_api_key)
    info = auth.key_info(key) or {}
    result = billing.create_upgrade_request(
        api_key=key,
        owner=str(info.get("owner") or ""),
        target_tier=req.target_tier,
        payment_ref=req.payment_ref,
        amount_ngn=req.amount_ngn,
        note=req.note,
        payer_email=req.payer_email,
        payer_name=req.payer_name,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "bad request"))
    return result


@router.get("/my-requests")
def my_requests(
    limit: int = 20,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    key = _require_key(x_api_key)
    info = auth.key_info(key) or {}
    owner = str(info.get("owner") or "")
    all_rows = billing.list_requests(limit=200)
    mine = [r for r in all_rows if r.get("owner") == owner][: max(1, min(limit, 50))]
    return {"ok": True, "requests": mine, "current_tier": info.get("tier")}


@router.get("/admin/pending")
def admin_pending(
    admin_secret: str,
    admin_email: str,
    limit: int = 50,
):
    if not settings.admin_secret or admin_secret != settings.admin_secret:
        raise HTTPException(status_code=403, detail="invalid admin_secret")
    if not billing.is_admin_email(admin_email):
        raise HTTPException(status_code=403, detail="admin_email not authorized")
    return {"ok": True, "requests": billing.list_requests(status="pending", limit=limit)}


@router.post("/admin/confirm")
def admin_confirm(req: AdminConfirmRequest):
    result = billing.confirm_request(
        req.request_id,
        admin_secret=req.admin_secret,
        admin_email=req.admin_email,
        action=req.action,
    )
    if not result.get("ok"):
        err = str(result.get("error", "failed")).lower()
        code = 403 if "admin" in err or "secret" in err else 400
        raise HTTPException(status_code=code, detail=result.get("error", "failed"))
    return result


@router.get("/admin/panel", response_class=HTMLResponse)
def admin_panel():
    emails = ", ".join(sorted(billing.ADMIN_EMAILS))
    acct = billing.SETTLEMENT_ACCOUNT
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>JagX Billing Admin</title>
  <style>
    :root {{ font-family: system-ui, sans-serif; color: #111; }}
    body {{ max-width: 920px; margin: 24px auto; padding: 0 16px; background: #f6f7f9; }}
    h1 {{ font-size: 1.35rem; margin-bottom: 4px; }}
    .sub {{ color: #555; margin-bottom: 20px; font-size: 0.95rem; }}
    .card {{ background: #fff; border: 1px solid #e2e4e8; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
    label {{ display: block; font-size: 0.85rem; color: #444; margin-top: 10px; }}
    input, select {{ width: 100%; padding: 8px 10px; margin-top: 4px; border: 1px solid #ccc; border-radius: 6px; box-sizing: border-box; }}
    button {{ margin-top: 12px; margin-right: 8px; padding: 8px 14px; border: 0; border-radius: 6px; cursor: pointer; font-weight: 600; }}
    .btn-load {{ background: #1a73e8; color: #fff; }}
    .btn-ok {{ background: #0d904f; color: #fff; }}
    .btn-no {{ background: #c5221f; color: #fff; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ text-align: left; padding: 8px 6px; border-bottom: 1px solid #eee; vertical-align: top; }}
    th {{ color: #555; font-weight: 600; }}
    .msg {{ margin-top: 12px; padding: 10px; border-radius: 6px; display: none; }}
    .msg.ok {{ display: block; background: #e6f4ea; color: #137333; }}
    .msg.err {{ display: block; background: #fce8e6; color: #c5221f; }}
    .hint {{ font-size: 0.8rem; color: #666; margin-top: 8px; }}
    code {{ background: #eef; padding: 1px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>JagX &amp; JRILICENSE — Billing Admin</h1>
  <p class="sub">Manual OPay settlements · Authorized: {emails}<br/>
  Pay-in: <strong>{acct['bank']}</strong> <code>{acct['account_number']}</code> — {acct['account_name']}</p>

  <div class="card">
    <label>Admin email
      <select id="admin_email">
        <option value="gbadamositajudeenwan@gmail.com">gbadamositajudeenwan@gmail.com</option>
        <option value="gbadamositajudeeneh@gmail.com">gbadamositajudeeneh@gmail.com</option>
      </select>
    </label>
    <label>Admin secret
      <input id="admin_secret" type="password" placeholder="JAGX_ADMIN_SECRET" autocomplete="off" />
    </label>
    <button class="btn-load" type="button" onclick="loadPending()">Load pending</button>
    <div id="msg" class="msg"></div>
    <p class="hint">Secret stays in this browser tab only.</p>
  </div>

  <div class="card">
    <h2 style="font-size:1.05rem;margin:0 0 12px">Pending upgrades</h2>
    <div id="list"><p class="hint">Load pending to see requests.</p></div>
  </div>

<script>
async function loadPending() {{
  const email = document.getElementById('admin_email').value;
  const secret = document.getElementById('admin_secret').value;
  const msg = document.getElementById('msg');
  msg.className = 'msg';
  try {{
    const url = '/billing/admin/pending?admin_secret=' + encodeURIComponent(secret)
      + '&admin_email=' + encodeURIComponent(email);
    const r = await fetch(url);
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'failed');
    const rows = data.requests || [];
    if (!rows.length) {{
      document.getElementById('list').innerHTML = '<p class="hint">No pending requests.</p>';
      return;
    }}
    let html = '<table><thead><tr><th>When</th><th>Owner / payer</th><th>Tier</th><th>Amount</th><th>Ref</th><th>Action</th></tr></thead><tbody>';
    for (const row of rows) {{
      const when = row.created_at ? new Date(row.created_at * 1000).toLocaleString() : '';
      const who = (row.payer_name || row.owner || '') + (row.payer_email ? '<br/><small>' + row.payer_email + '</small>' : '');
      html += '<tr>'
        + '<td>' + when + '</td>'
        + '<td>' + who + '<br/><small>' + (row.api_key_hint || '') + '</small></td>'
        + '<td>' + (row.target_tier || '') + '</td>'
        + '<td>₦' + (row.amount_ngn || row.expected_ngn || '') + '</td>'
        + '<td><code>' + (row.payment_ref || '') + '</code>'
        + (row.note ? '<br/><small>' + row.note + '</small>' : '') + '</td>'
        + '<td>'
        + '<button class="btn-ok" type="button" onclick="act(\'' + row.id + '\',\'confirm\')">Confirm</button>'
        + '<button class="btn-no" type="button" onclick="act(\'' + row.id + '\',\'reject\')">Reject</button>'
        + '</td></tr>';
    }}
    html += '</tbody></table>';
    document.getElementById('list').innerHTML = html;
  }} catch (e) {{
    msg.textContent = e.message || String(e);
    msg.className = 'msg err';
  }}
}}

async function act(requestId, action) {{
  const email = document.getElementById('admin_email').value;
  const secret = document.getElementById('admin_secret').value;
  const msg = document.getElementById('msg');
  msg.className = 'msg';
  try {{
    const r = await fetch('/billing/admin/confirm', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{
        request_id: requestId,
        admin_secret: secret,
        admin_email: email,
        action: action
      }})
    }});
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'failed');
    msg.textContent = (data.message || action + ' ok') + ' · ' + (data.tier || '');
    msg.className = 'msg ok';
    loadPending();
  }} catch (e) {{
    msg.textContent = e.message || String(e);
    msg.className = 'msg err';
  }}
}}
</script>
</body>
</html>"""
    return HTMLResponse(html)
