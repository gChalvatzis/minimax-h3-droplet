# =============================================================================
# api_server.py  --  single-endpoint REST API in front of a local ComfyUI
#                    running MiniMax H3 (text/image -> video + audio) + Turbo LoRA
#
# One public endpoint, always multipart/form-data:
#
#   POST /v1/videos
#     prompt          (str,   required)
#     duration        (float, seconds; default 6; hard cap = MAX_SECONDS)
#     audio_enabled   (bool,  default true)
#     reference_image (file,  optional -> if present the request becomes image-to-video)
#     negative        (str,   optional)
#     width,height    (int,   optional; defaults from env, rounded to /16)
#     seed            (int,   optional; random if omitted)
#     steps           (int,   optional; default STEPS env - tuned for the Turbo LoRA)
#     lora            (str,   optional; filename in models/loras; default = Turbo LoRA)
#     lora_strength   (float, optional; default 1.0)
#   -> {"id": "...", "status": "queued", "resolved": {...}}
#
#   GET  /v1/videos/{id}   -> {status, progress, error, output_url?}
#   GET  /files/{name}     -> the finished .mp4
#   GET  /v1/loras         -> installed LoRA files
#   GET  /healthz          -> liveness + ComfyUI reachability
#
# All endpoints except /healthz require header:  X-API-Key: <API_KEY>
#
# Env: COMFY_URL API_KEY WF_DIR OUT_DIR LORA_DIR
#      FPS(=24) MAX_SECONDS(=15) STEPS(=6) DEFAULT_WIDTH(=848) DEFAULT_HEIGHT(=480)
#      DEFAULT_LORA(=minimax_h3_turbo_v4_step600_ema.safetensors)
#
# The server writes params into an API-format ComfyUI workflow using a mapping
# file. For each of t2v / i2v it looks for, in order:
#     <kind>_audio.json  (when audio_enabled)   then   <kind>.json
# plus the sibling  <kind>.map.json . See WF_DIR/README.txt (written by setup).
# =============================================================================
from __future__ import annotations

import copy
import json
import os
import time
import uuid
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Optional

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
WF_DIR = Path(os.environ.get("WF_DIR", "./workflows"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "./outputs"))
LORA_DIR = Path(os.environ.get("LORA_DIR", "./loras"))

FPS = int(os.environ.get("FPS", "24"))
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "15"))
DEFAULT_STEPS = int(os.environ.get("STEPS", "6"))
DEFAULT_WIDTH = int(os.environ.get("DEFAULT_WIDTH", "848"))
DEFAULT_HEIGHT = int(os.environ.get("DEFAULT_HEIGHT", "480"))
DEFAULT_LORA = os.environ.get("DEFAULT_LORA", "minimax_h3_turbo_v4_step600_ema.safetensors")

OUT_DIR.mkdir(parents=True, exist_ok=True)
CLIENT_ID = str(uuid.uuid4())
app = FastAPI(title="MiniMax H3 Video API", version="2.0")

# ------------------------------------------------------------------ jobs ----
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = Lock()


def _new_job(resolved: dict) -> str:
    jid = uuid.uuid4().hex[:16]
    with JOBS_LOCK:
        JOBS[jid] = {"status": "queued", "created": time.time(), "error": None,
                     "output_file": None, "comfy_prompt_id": None,
                     "progress": None, "resolved": resolved}
    return jid


def _set(jid: str, **kw) -> None:
    with JOBS_LOCK:
        JOBS[jid].update(kw)


# ------------------------------------------------------------------ auth ----
def _auth(x_api_key: Optional[str]) -> None:
    if not API_KEY:
        raise HTTPException(500, "server misconfigured: API_KEY not set")
    if x_api_key != API_KEY:
        raise HTTPException(401, "missing or invalid X-API-Key")


# ------------------------------------------------------- helpers -----------
def _round16(n: int) -> int:
    return max(16, int(round(n / 16.0)) * 16)


def _frames_for(duration_s: float) -> int:
    """duration -> frame count, snapped to 4n+1 which H3/most video latents expect."""
    raw = max(1, int(round(duration_s * FPS)))
    return ((raw - 1) // 4) * 4 + 1


def _pick_workflow(kind: str, audio: bool) -> tuple[dict, dict, str]:
    candidates = []
    if audio:
        candidates.append(WF_DIR / f"{kind}_audio.json")
    candidates.append(WF_DIR / f"{kind}.json")
    wf_path = next((p for p in candidates if p.exists()), None)
    if wf_path is None:
        raise HTTPException(
            503,
            f"no workflow for kind='{kind}' (looked for {[p.name for p in candidates]}). "
            f"Export an API-format workflow from ComfyUI once - see {WF_DIR}/README.txt",
        )
    map_path = WF_DIR / f"{kind}.map.json"
    if not map_path.exists():
        raise HTTPException(503, f"{map_path} not found - copy it from {kind}.map.json.example and edit")
    return json.loads(wf_path.read_text()), json.loads(map_path.read_text()), wf_path.name


def _apply(graph: dict, mapping: dict, params: dict[str, Any]) -> dict:
    g = copy.deepcopy(graph)
    for name, value in params.items():
        if value is None or name not in mapping:
            continue
        tgt = mapping[name]
        nid, key = str(tgt["node"]), tgt["input"]
        if nid not in g:
            raise HTTPException(500, f"{name}: mapping node {nid} is not in the workflow graph")
        g[nid].setdefault("inputs", {})[key] = value
    return g


# --------------------------------------------------------- comfy plumbing ---
def _comfy_upload_image(raw: bytes, filename: str) -> str:
    r = httpx.post(f"{COMFY_URL}/upload/image",
                   files={"image": (filename or "input.png", raw, "application/octet-stream")},
                   data={"overwrite": "true"}, timeout=120)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def _comfy_submit(graph: dict) -> str:
    r = httpx.post(f"{COMFY_URL}/prompt",
                   json={"prompt": graph, "client_id": CLIENT_ID}, timeout=60)
    if r.status_code != 200:
        raise HTTPException(502, f"ComfyUI rejected the workflow: {r.text[:800]}")
    return r.json()["prompt_id"]


def _first_video(outputs: dict) -> Optional[dict]:
    for node_out in outputs.values():
        for key in ("gifs", "videos", "images"):
            for item in node_out.get(key, []) or []:
                if item.get("filename", "").lower().endswith((".mp4", ".webm", ".mov", ".mkv")):
                    return item
    return None


def _download(fileinfo: dict, jid: str) -> Path:
    r = httpx.get(f"{COMFY_URL}/view",
                  params={"filename": fileinfo["filename"],
                          "subfolder": fileinfo.get("subfolder", ""),
                          "type": fileinfo.get("type", "output")},
                  timeout=600)
    r.raise_for_status()
    ext = os.path.splitext(fileinfo["filename"])[1] or ".mp4"
    dest = OUT_DIR / f"{jid}{ext}"
    dest.write_bytes(r.content)
    return dest


def _comfy_wait(prompt_id: str, jid: str, timeout_s: int = 7200) -> Path:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        h = httpx.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30)
        if h.status_code == 200 and prompt_id in h.json():
            entry = h.json()[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI error: {status.get('messages')}")
            fi = _first_video(entry.get("outputs", {}))
            if fi:
                return _download(fi, jid)
            if status.get("completed"):
                raise RuntimeError("workflow finished but produced no video "
                                   "(is a VideoCombine/Save node present?)")
        try:
            q = httpx.get(f"{COMFY_URL}/queue", timeout=10).json()
            if any(prompt_id == it[1] for it in q.get("queue_running", [])):
                _set(jid, progress="running")
            elif any(prompt_id == it[1] for it in q.get("queue_pending", [])):
                _set(jid, progress="pending")
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("timed out waiting for ComfyUI")


def _run_job(jid: str, kind: str, audio: bool, params: dict,
             image_bytes: Optional[bytes], image_name: Optional[str]) -> None:
    try:
        _set(jid, status="running")
        graph, mapping, used = _pick_workflow(kind, audio)
        if image_bytes is not None:
            params["image"] = _comfy_upload_image(image_bytes, image_name or "input.png")
        graph = _apply(graph, mapping, params)
        pid = _comfy_submit(graph)
        _set(jid, comfy_prompt_id=pid, progress=f"submitted ({used})")
        out = _comfy_wait(pid, jid)
        _set(jid, status="succeeded", output_file=str(out), progress="done")
    except HTTPException as e:
        _set(jid, status="failed", error=f"{e.status_code}: {e.detail}")
    except Exception as e:  # noqa: BLE001
        _set(jid, status="failed", error=str(e))


# --------------------------------------------------------------- routes ----
@app.get("/healthz")
def healthz():
    ok = False
    try:
        ok = httpx.get(f"{COMFY_URL}/system_stats", timeout=5).status_code == 200
    except Exception:
        pass
    return {"ok": True, "comfy_reachable": ok, "fps": FPS, "max_seconds": MAX_SECONDS,
            "workflows": sorted(p.name for p in WF_DIR.glob("*.json"))}


@app.get("/v1/loras")
def list_loras(x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    loras = sorted(p.name for p in LORA_DIR.iterdir()
                   if p.suffix in (".safetensors", ".pt")) if LORA_DIR.exists() else []
    return {"loras": loras, "default": DEFAULT_LORA}


@app.post("/v1/videos")
async def create_video(
    prompt: str = Form(...),
    duration: float = Form(6.0),
    audio_enabled: bool = Form(True),
    negative: Optional[str] = Form(None),
    width: int = Form(DEFAULT_WIDTH),
    height: int = Form(DEFAULT_HEIGHT),
    seed: Optional[int] = Form(None),
    steps: int = Form(DEFAULT_STEPS),
    lora: Optional[str] = Form(None),
    lora_strength: float = Form(1.0),
    reference_image: Optional[UploadFile] = File(None),
    x_api_key: Optional[str] = Header(None),
):
    _auth(x_api_key)
    if not prompt.strip():
        raise HTTPException(422, "prompt is empty")
    if duration <= 0 or duration > MAX_SECONDS:
        raise HTTPException(422, f"duration must be in (0, {MAX_SECONDS}] seconds "
                                 f"(H3 open weights cap short clips; raise MAX_SECONDS to override)")

    img_bytes = await reference_image.read() if reference_image is not None else None
    if reference_image is not None and not img_bytes:
        raise HTTPException(422, "reference_image was sent but is empty")
    kind = "i2v" if img_bytes else "t2v"

    frames = _frames_for(duration)
    resolved = {
        "kind": kind, "audio_enabled": audio_enabled,
        "duration_s": duration, "fps": FPS, "frames": frames,
        "width": _round16(width), "height": _round16(height),
        "steps": steps, "lora": lora or DEFAULT_LORA, "lora_strength": lora_strength,
    }
    params = {
        "prompt": prompt,
        "negative": negative,
        "frames": frames,
        "fps": FPS,
        "width": resolved["width"],
        "height": resolved["height"],
        "steps": steps,
        "seed": seed if seed is not None else int(time.time() * 1e6) % (2**31),
        "lora": lora or DEFAULT_LORA,
        "lora_strength": lora_strength,
        "audio_enabled": audio_enabled,
    }
    jid = _new_job(resolved)
    Thread(target=_run_job, args=(jid, kind, audio_enabled, params, img_bytes,
                                  reference_image.filename if reference_image else None),
           daemon=True).start()
    return {"id": jid, "status": "queued", "resolved": resolved}


@app.get("/v1/videos/{job_id}")
def job_status(job_id: str, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job id")
    out = {"id": job_id, "status": job["status"], "progress": job["progress"],
           "error": job["error"], "resolved": job["resolved"]}
    if job["status"] == "succeeded" and job["output_file"]:
        out["output_url"] = f"/files/{Path(job['output_file']).name}"
    return out


@app.get("/files/{name}")
def get_file(name: str, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    path = (OUT_DIR / name).resolve()
    if not path.is_file() or path.parent != OUT_DIR.resolve():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="video/mp4", filename=name)
