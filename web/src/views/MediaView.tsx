/**
 * JagX v13 — Media Studio.
 *
 * One place for the whole multimodal surface: generate, upload, edit, read and
 * inspect. Everything here talks to the local engine — no external image API,
 * no API key. The capability banner tells the truth about which tier is live.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, getToken, mediaUrl } from "../api";
import type { DocumentResult, MediaAsset, MediaCapabilities, MediaStats, UnderstandResult } from "../types";
import {
  IconCrop, IconDownload, IconExpand, IconEye, IconFile, IconImage, IconPalette,
  IconSpark, IconTrash, IconUpload, IconWand, IconX,
} from "../icons";

type Tab = "generate" | "edit" | "inspect" | "gallery";

const STYLE_PRESETS = [
  "grayscale", "sepia", "invert", "poster", "sketch", "warm",
  "cool", "vivid", "soft", "noir", "blur", "sharpen",
];

const EDIT_OPS: { key: string; label: string; hint: string }[] = [
  { key: "inpaint", label: "Inpaint", hint: "Fill a region with new content" },
  { key: "outpaint", label: "Outpaint", hint: "Extend the canvas outward" },
  { key: "remove-background", label: "Remove background", hint: "Knock out the backdrop" },
  { key: "upscale", label: "Upscale", hint: "Enlarge with sharpening" },
  { key: "style", label: "Style", hint: "Apply a named look" },
  { key: "transform", label: "Transform", hint: "Rotate, flip, resize, crop" },
];

function fmtBytes(n: number): string {
  if (!n) return "0 B";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

/**
 * Images live behind the auth header, so a plain <img src> would 401.
 * Fetch the bytes with the token and hand back an object URL.
 */
function useAuthedImage(id: string | null, thumb = false) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!id) { setUrl(null); return; }
    let revoked = false;
    let objectUrl: string | null = null;
    (async () => {
      try {
        const token = getToken();
        const res = await fetch(thumb ? `/api/media/${id}/thumb` : mediaUrl(id), {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        });
        if (!res.ok) return;
        const blob = await res.blob();
        objectUrl = URL.createObjectURL(blob);
        if (revoked) { URL.revokeObjectURL(objectUrl); return; }
        setUrl(objectUrl);
      } catch { /* leave null */ }
    })();
    return () => {
      revoked = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [id, thumb]);
  return url;
}

function AssetThumb({ asset, onOpen }: { asset: MediaAsset; onOpen: () => void }) {
  const url = useAuthedImage(asset.id, true);
  return (
    <button className="media-tile" onClick={onOpen} title={asset.prompt || asset.filename}>
      <div className="media-tile-frame">
        {asset.kind === "image" && url
          ? <img src={url} alt={asset.prompt || asset.filename} loading="lazy" />
          : <div className="media-tile-doc"><IconFile size={26} /><span>{asset.ext.replace(".", "").toUpperCase()}</span></div>}
      </div>
      <div className="media-tile-meta">
        <span className={`chip ${asset.source === "generated" ? "accent" : asset.source === "edited" ? "violet" : "info"}`}>
          {asset.source}
        </span>
        {asset.width > 0 && <span className="faint small">{asset.width}×{asset.height}</span>}
      </div>
    </button>
  );
}

export function MediaView() {
  const [tab, setTab] = useState<Tab>("generate");
  const [caps, setCaps] = useState<MediaCapabilities | null>(null);
  const [assets, setAssets] = useState<MediaAsset[]>([]);
  const [stats, setStats] = useState<MediaStats | null>(null);
  const [selected, setSelected] = useState<MediaAsset | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [note, setNote] = useState("");

  // generate form
  const [prompt, setPrompt] = useState("");
  const [width, setWidth] = useState(768);
  const [height, setHeight] = useState(768);
  const [styleHint, setStyleHint] = useState("");
  const [seed, setSeed] = useState<string>("");

  // edit form
  const [op, setOp] = useState("style");
  const [editStyle, setEditStyle] = useState("sepia");
  const [editPrompt, setEditPrompt] = useState("");
  const [box, setBox] = useState("40,40,200,200");
  const [direction, setDirection] = useState("right");
  const [amount, setAmount] = useState(128);
  const [factor, setFactor] = useState(2);
  const [degrees, setDegrees] = useState(90);

  // inspect
  const [question, setQuestion] = useState("");
  const [understanding, setUnderstanding] = useState<UnderstandResult | null>(null);
  const [docResult, setDocResult] = useState<DocumentResult | null>(null);

  const fileRef = useRef<HTMLInputElement>(null);
  const selectedUrl = useAuthedImage(selected?.kind === "image" ? selected.id : null);

  const load = useCallback(async () => {
    try {
      const r = await api.media();
      setAssets(r.media);
      setStats(r.stats);
    } catch (e) { setError((e as Error).message); }
  }, []);

  useEffect(() => {
    (async () => {
      try { setCaps(await api.mediaCapabilities()); } catch { /* banner is optional */ }
      await load();
    })();
  }, [load]);

  const run = async (fn: () => Promise<void>) => {
    setBusy(true); setError(""); setNote("");
    try { await fn(); } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };

  const doGenerate = () => run(async () => {
    if (!prompt.trim()) return;
    const asset = await api.generateImage({
      prompt: prompt.trim(),
      width, height,
      style: styleHint.trim(),
      seed: seed.trim() === "" ? null : Number(seed),
    });
    setSelected(asset);
    setNote(`Generated ${asset.width}×${asset.height} with the ${asset.backend} backend.`);
    await load();
  });

  const doUpload = (file: File) => run(async () => {
    const asset = await api.uploadMedia(file);
    setSelected(asset);
    setNote(`Uploaded ${asset.filename} (${asset.kind}).`);
    await load();
  });

  const doEdit = () => run(async () => {
    if (!selected) return;
    const body: Parameters<typeof api.editImage>[1] = { operation: op };
    if (op === "inpaint") {
      body.prompt = editPrompt;
      body.box = box.split(",").map((v) => Number(v.trim())).filter((v) => !Number.isNaN(v));
    }
    if (op === "outpaint") { body.direction = direction; body.amount = amount; }
    if (op === "upscale") body.factor = factor;
    if (op === "style") body.style = editStyle;
    if (op === "transform") body.degrees = degrees;

    const asset = await api.editImage(selected.id, body);
    setSelected(asset);
    setNote(`${op} applied via ${asset.backend}.`);
    await load();
  });

  const doUnderstand = () => run(async () => {
    if (!selected) return;
    setDocResult(null);
    setUnderstanding(await api.understandImage(selected.id, question));
  });

  const doRead = () => run(async () => {
    if (!selected) return;
    setUnderstanding(null);
    setDocResult(await api.readDocument(selected.id));
  });

  const doDelete = (id: string) => run(async () => {
    await api.deleteMedia(id);
    if (selected?.id === id) { setSelected(null); setUnderstanding(null); setDocResult(null); }
    await load();
  });

  const download = async (asset: MediaAsset) => {
    const token = getToken();
    const res = await fetch(mediaUrl(asset.id), {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (!res.ok) return;
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = asset.filename || `${asset.id}${asset.ext}`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const tier = useMemo(() => {
    if (!caps) return null;
    const c = caps.capabilities;
    const real = c.image_generation !== "fallback";
    return {
      real,
      label: real ? `Real diffusion (${c.image_generation})` : "Built-in renderer",
      vision: c.vision === "vlm" ? "Local vision model" : "OCR + analysis",
    };
  }, [caps]);

  return (
    <div className="content-inner wide">
      {/* ---- capability banner ------------------------------------------ */}
      {caps && (
        <div className={`card media-banner${tier?.real ? " ok" : ""}`}>
          <div className="row" style={{ alignItems: "flex-start", gap: 12 }}>
            <IconSpark size={20} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="card-title">
                Multimodal engine — {tier?.label} · {tier?.vision}
              </div>
              <div className="card-sub">
                Runs entirely on this machine. <strong>No API key, no external image service.</strong>{" "}
                {caps.capabilities.ocr ? "OCR available. " : "OCR unavailable. "}
                {caps.capabilities.documents.length} document formats supported.
              </div>
              {!tier?.real && (
                <div className="small faint mt-2">
                  Install a real model to upgrade the same tools:{" "}
                  <span className="mono">pip install torch diffusers accelerate</span> — or point{" "}
                  <span className="mono">JAGX_COMFYUI_URL</span> at a local ComfyUI.
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ---- tabs -------------------------------------------------------- */}
      <div className="media-tabs">
        {([
          ["generate", "Generate", <IconWand size={15} key="g" />],
          ["edit", "Edit", <IconPalette size={15} key="e" />],
          ["inspect", "Read & Inspect", <IconEye size={15} key="i" />],
          ["gallery", "Gallery", <IconImage size={15} key="l" />],
        ] as [Tab, string, JSX.Element][]).map(([key, label, icon]) => (
          <button key={key} className={`media-tab${tab === key ? " active" : ""}`} onClick={() => setTab(key)}>
            {icon}<span>{label}</span>
          </button>
        ))}
        <div className="spacer" />
        <button className="btn sm" onClick={() => fileRef.current?.click()} disabled={busy}>
          <IconUpload size={14} /> Upload
        </button>
        <input
          ref={fileRef}
          type="file"
          hidden
          accept="image/*,.pdf,.docx,.xlsx,.csv,.tsv,.txt,.md,.json,.log"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) doUpload(f);
            e.target.value = "";
          }}
        />
      </div>

      {error && <div className="alert error mt-4">{error}</div>}
      {note && <div className="alert ok mt-4">{note}</div>}

      <div className="media-layout">
        {/* ---- left: the active panel ---------------------------------- */}
        <div className="media-panel">
          {tab === "generate" && (
            <div className="card">
              <div className="card-head">
                <IconWand size={18} />
                <div>
                  <div className="card-title">Text to image</div>
                  <div className="card-sub">Describe what you want. The prompt seeds the render deterministically.</div>
                </div>
              </div>
              <label className="field-label">Prompt</label>
              <textarea
                className="textarea"
                rows={3}
                placeholder="A lighthouse at dusk over a calm sea, long exposure"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />
              <div className="row wrap mt-4" style={{ alignItems: "flex-end" }}>
                <div style={{ flex: "1 1 120px" }}>
                  <label className="field-label">Width</label>
                  <input className="input" type="number" min={64} max={1024} value={width}
                         onChange={(e) => setWidth(Number(e.target.value))} />
                </div>
                <div style={{ flex: "1 1 120px" }}>
                  <label className="field-label">Height</label>
                  <input className="input" type="number" min={64} max={1024} value={height}
                         onChange={(e) => setHeight(Number(e.target.value))} />
                </div>
                <div style={{ flex: "1 1 160px" }}>
                  <label className="field-label">Style hint</label>
                  <input className="input" placeholder="flat vector, isometric…" value={styleHint}
                         onChange={(e) => setStyleHint(e.target.value)} />
                </div>
                <div style={{ flex: "0 0 110px" }}>
                  <label className="field-label">Seed</label>
                  <input className="input" placeholder="auto" value={seed}
                         onChange={(e) => setSeed(e.target.value)} />
                </div>
              </div>
              <div className="row mt-4">
                <button className="btn primary" onClick={doGenerate} disabled={busy || !prompt.trim()}>
                  {busy ? <span className="spinner" /> : <IconSpark size={15} />} Generate
                </button>
                <div className="spacer" />
                <div className="row wrap" style={{ gap: 6 }}>
                  {[["Square", 768, 768], ["Wide", 1024, 576], ["Tall", 576, 1024]].map(([l, w, h]) => (
                    <button key={l as string} className="btn ghost sm"
                            onClick={() => { setWidth(w as number); setHeight(h as number); }}>
                      {l as string}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {tab === "edit" && (
            <div className="card">
              <div className="card-head">
                <IconPalette size={18} />
                <div>
                  <div className="card-title">Edit an image</div>
                  <div className="card-sub">
                    {selected ? `Editing ${selected.filename || selected.id}` : "Pick an image from the gallery first."}
                  </div>
                </div>
              </div>

              <div className="row wrap" style={{ gap: 6 }}>
                {EDIT_OPS.map((o) => (
                  <button key={o.key} className={`btn sm${op === o.key ? " primary" : ""}`}
                          onClick={() => setOp(o.key)} title={o.hint}>
                    {o.label}
                  </button>
                ))}
              </div>

              <div className="mt-4">
                {op === "inpaint" && (
                  <>
                    <label className="field-label">Region to fill (x0,y0,x1,y1 — pixels or 0–1 fractions)</label>
                    <input className="input" value={box} onChange={(e) => setBox(e.target.value)} />
                    <label className="field-label mt-4">What should appear there</label>
                    <input className="input" placeholder="a small wooden boat" value={editPrompt}
                           onChange={(e) => setEditPrompt(e.target.value)} />
                  </>
                )}
                {op === "outpaint" && (
                  <div className="row wrap" style={{ alignItems: "flex-end" }}>
                    <div style={{ flex: "1 1 160px" }}>
                      <label className="field-label">Direction</label>
                      <select className="select" value={direction} onChange={(e) => setDirection(e.target.value)}>
                        {["right", "left", "up", "down", "all"].map((d) => <option key={d} value={d}>{d}</option>)}
                      </select>
                    </div>
                    <div style={{ flex: "1 1 160px" }}>
                      <label className="field-label">Pixels to add</label>
                      <input className="input" type="number" min={16} max={1024} value={amount}
                             onChange={(e) => setAmount(Number(e.target.value))} />
                    </div>
                  </div>
                )}
                {op === "upscale" && (
                  <>
                    <label className="field-label">Factor ({factor}×)</label>
                    <input type="range" min={1.1} max={4} step={0.1} value={factor}
                           onChange={(e) => setFactor(Number(e.target.value))} style={{ width: "100%" }} />
                  </>
                )}
                {op === "style" && (
                  <>
                    <label className="field-label">Look</label>
                    <div className="row wrap" style={{ gap: 6 }}>
                      {STYLE_PRESETS.map((s) => (
                        <button key={s} className={`btn sm${editStyle === s ? " primary" : ""}`}
                                onClick={() => setEditStyle(s)}>{s}</button>
                      ))}
                    </div>
                  </>
                )}
                {op === "transform" && (
                  <>
                    <label className="field-label">Rotation degrees ({degrees}°)</label>
                    <input type="range" min={-180} max={180} step={5} value={degrees}
                           onChange={(e) => setDegrees(Number(e.target.value))} style={{ width: "100%" }} />
                  </>
                )}
                {op === "remove-background" && (
                  <div className="small faint">
                    Flood-fills from the border to knock out a near-uniform backdrop and writes an alpha channel.
                    Install <span className="mono">rembg</span> for complex backgrounds.
                  </div>
                )}
              </div>

              <div className="row mt-4">
                <button className="btn primary" onClick={doEdit} disabled={busy || !selected}>
                  {busy ? <span className="spinner" /> : <IconCrop size={15} />} Apply {op}
                </button>
              </div>
            </div>
          )}

          {tab === "inspect" && (
            <div className="card">
              <div className="card-head">
                <IconEye size={18} />
                <div>
                  <div className="card-title">Read &amp; inspect</div>
                  <div className="card-sub">
                    {selected ? `Inspecting ${selected.filename || selected.id}` : "Pick an image or document first."}
                  </div>
                </div>
              </div>

              <label className="field-label">Ask a question about the image (optional)</label>
              <input className="input" placeholder="What is in this image?" value={question}
                     onChange={(e) => setQuestion(e.target.value)} />

              <div className="row mt-4">
                <button className="btn primary" onClick={doUnderstand} disabled={busy || !selected}>
                  {busy ? <span className="spinner" /> : <IconEye size={15} />} Describe image
                </button>
                <button className="btn" onClick={doRead} disabled={busy || !selected}>
                  <IconFile size={15} /> Read as document
                </button>
              </div>

              {understanding && (
                <div className="mt-4">
                  <div className="row small faint mb-2">
                    <span className="chip info">{understanding.backend}</span>
                    <span>{understanding.stats.width}×{understanding.stats.height} · {understanding.stats.orientation}</span>
                  </div>
                  <div className="media-readout">{understanding.description}</div>
                  {understanding.ocr_text && (
                    <>
                      <div className="field-label mt-4">Text found in the image</div>
                      <pre className="media-pre">{understanding.ocr_text}</pre>
                    </>
                  )}
                  <div className="field-label mt-4">Dominant colours</div>
                  <div className="row" style={{ gap: 8 }}>
                    {understanding.stats.dominant_colors.map((c) => (
                      <div key={c.hex} className="swatch" title={`${c.hex} · ${Math.round(c.share * 100)}%`}>
                        <span style={{ background: c.hex }} />
                        <em>{c.hex}</em>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {docResult && (
                <div className="mt-4">
                  <div className="row small faint mb-2">
                    <span className="chip info">{String(docResult.meta.engine ?? docResult.kind)}</span>
                    <span>{docResult.description}</span>
                  </div>
                  <pre className="media-pre">{docResult.text || "(no text extracted)"}</pre>
                </div>
              )}
            </div>
          )}

          {tab === "gallery" && (
            <div className="card">
              <div className="card-head">
                <IconImage size={18} />
                <div>
                  <div className="card-title">Gallery</div>
                  <div className="card-sub">
                    {stats ? `${stats.total_count} assets · ${fmtBytes(stats.total_bytes)}` : "Loading…"}
                  </div>
                </div>
              </div>
              {assets.length === 0
                ? <div className="muted small">Nothing here yet. Generate an image or upload a file.</div>
                : (
                  <div className="media-grid">
                    {assets.map((a) => (
                      <AssetThumb key={a.id} asset={a} onOpen={() => { setSelected(a); setUnderstanding(null); setDocResult(null); }} />
                    ))}
                  </div>
                )}
            </div>
          )}
        </div>

        {/* ---- right: preview ------------------------------------------ */}
        <div className="media-preview">
          <div className="card">
            <div className="card-head">
              <IconExpand size={18} />
              <div>
                <div className="card-title">Preview</div>
                <div className="card-sub">{selected ? (selected.prompt || selected.filename || selected.id) : "Nothing selected"}</div>
              </div>
            </div>

            {!selected && <div className="media-empty"><IconImage size={30} /><span>Select or create an asset</span></div>}

            {selected && (
              <>
                <div className="media-stage">
                  {selected.kind === "image" && selectedUrl
                    ? <img src={selectedUrl} alt={selected.prompt || selected.filename} />
                    : <div className="media-empty"><IconFile size={30} /><span>{selected.filename || selected.id}</span></div>}
                </div>

                <div className="row wrap small faint mt-4" style={{ gap: 8 }}>
                  <span className={`chip ${selected.source === "generated" ? "accent" : selected.source === "edited" ? "violet" : "info"}`}>
                    {selected.source}
                  </span>
                  {selected.backend && <span className="chip">{selected.backend}</span>}
                  {selected.width > 0 && <span>{selected.width}×{selected.height}</span>}
                  <span>{fmtBytes(selected.bytes)}</span>
                  {selected.operation && <span>op: {selected.operation}</span>}
                </div>

                <div className="row mt-4" style={{ gap: 8 }}>
                  <button className="btn sm" onClick={() => download(selected)}>
                    <IconDownload size={14} /> Download
                  </button>
                  <button className="btn sm" onClick={() => { setTab("edit"); }}>
                    <IconPalette size={14} /> Edit
                  </button>
                  <button className="btn sm" onClick={() => { setTab("inspect"); }}>
                    <IconEye size={14} /> Inspect
                  </button>
                  <div className="spacer" />
                  <button className="btn ghost sm" onClick={() => doDelete(selected.id)} disabled={busy} aria-label="Delete asset">
                    <IconTrash size={14} />
                  </button>
                </div>
              </>
            )}
          </div>

          {caps && (
            <div className="card">
              <div className="card-title mb-2">Honest limits</div>
              <div className="small faint">{caps.honest_limits}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
