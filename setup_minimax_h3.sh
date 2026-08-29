#!/usr/bin/env bash
# =============================================================================
# setup_minimax_h3.sh
#
# One-shot provisioning of MiniMax H3 (open-weights text-to-video / image-to-video)
# + LoRA + a public REST API, on a fresh DigitalOcean GPU Droplet.
#
# Target OS   : Ubuntu 22.04 / 24.04 with the NVIDIA driver already installed
#               (DigitalOcean "AI/ML Ready" GPU Droplet image).
# Target GPU  : NVIDIA RTX 4000 Ada (20 GB) or RTX 6000 Ada (48 GB). The script
#               auto-detects VRAM and picks a model variant accordingly.
#
# DESIGN FOR YOUR WORKFLOW (create droplet -> work -> destroy droplet):
#   Everything heavy (Python venv, ComfyUI, ~40 GB of model weights, your LoRAs,
#   workflows, outputs) is installed under $PERSIST_ROOT. If you attach the SAME
#   DigitalOcean Volume (block storage) to each new droplet, re-running this
#   script is near-instant: it only re-checks packages and restarts services.
#   Without a Volume it still works, it just re-downloads the weights each time.
#
# USAGE:
#   export HF_TOKEN=hf_xxxxxxxx        # HF account that has accepted the H3 licence
#   export API_KEY='long-random-secret'  # protects your public API
#   # optional: export PERSIST_ROOT=/mnt/volume_fra1_01
#   sudo -E bash setup_minimax_h3.sh                 # full setup + start
#   sudo -E bash setup_minimax_h3.sh --setup-only    # provision, don't start services
#   sudo -E bash setup_minimax_h3.sh --restart       # just restart services
#   sudo -E bash setup_minimax_h3.sh --stop          # stop services
#   sudo -E bash setup_minimax_h3.sh --list-models   # print the remote file list of $HF_REPO
#
# NOTE ON LICENCE / REGION: the MiniMax H3 Community Licence currently excludes
# open-weight deployment in the US, EU, UK and South Korea unless MiniMax grants
# you individual authorisation. Greece is in the EU. Confirm your authorisation
# and pick your droplet region accordingly before running this in production.
# =============================================================================

set -euo pipefail

# ----------------------------- configuration ---------------------------------
COMFY_PORT="${COMFY_PORT:-8188}"          # ComfyUI backend, bound to 127.0.0.1 only
API_PORT="${API_PORT:-8000}"             # public FastAPI, bound to 0.0.0.0
TORCH_CUDA="${TORCH_CUDA:-cu124}"        # cu124 wheels match recent DO GPU images
PYTHON_BIN="${PYTHON_BIN:-python3}"

# HuggingFace repo that hosts ComfyUI-ready H3 files. VERIFY the filenames below
# against `sudo -E bash setup_minimax_h3.sh --list-models` on first run and
# override via env if MiniMax has re-tagged them.
HF_REPO="${HF_REPO:-Comfy-Org/MiniMax-H3}"

# Model files to fetch:  "<path-in-repo>::<destination-subdir-under-ComfyUI/models>"
# Two diffusion variants; the script keeps only the one that fits your VRAM.
DIFF_INT8="${DIFF_INT8:-minimax_h3_fl2va_pruned_int8.safetensors}"          # ~21 GB, for 20-24 GB cards w/ offload or 48 GB comfortably
DIFF_BF16="${DIFF_BF16:-minimax_h3_fl2va_bf16.safetensors}"                 # full precision, needs a big card
TEXT_ENCODER="${TEXT_ENCODER:-qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors}"
VIDEO_VAE="${VIDEO_VAE:-minimax_h3_video_vae_fp16.safetensors}"
AUDIO_VAE="${AUDIO_VAE:-minimax_h3_audio_vae_fp32.safetensors}"

# LoRA to auto-download. Default = the community "Turbo" distill LoRA: it renders
# video + synced audio in ~4-8 sampling steps instead of ~20, which is the single
# biggest speed win on a 20 GB card that is already offloading to system RAM.
# Drop any extra .safetensors into $PERSIST_ROOT/loras and they are exposed too.
LORA_HF_REPO="${LORA_HF_REPO:-larryvrh/MiniMax-H3-Turbo-Lora}"
LORA_HF_FILE="${LORA_HF_FILE:-minimax_h3_turbo_v4_step600_ema.safetensors}"

# API runtime tuning (passed through to api_server.py)
API_FPS="${API_FPS:-24}"                  # frames-per-second used to turn duration(s) into frame count
API_MAX_SECONDS="${API_MAX_SECONDS:-15}"  # hard cap on requested clip length
API_STEPS="${API_STEPS:-6}"               # sampler steps; 6 suits the Turbo LoRA
API_DEF_W="${API_DEF_W:-848}"             # default width  (20 GB card -> keep modest)
API_DEF_H="${API_DEF_H:-480}"             # default height

# ----------------------------- helpers --------------------------------------
log()  { printf '\033[1;36m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

MODE="full"
case "${1:-}" in
  --setup-only) MODE="setup-only" ;;
  --restart)    MODE="restart" ;;
  --stop)       MODE="stop" ;;
  --list-models) MODE="list-models" ;;
  "" )          MODE="full" ;;
  *) die "unknown argument: $1" ;;
esac

# ----------------------------- locate persistent storage -------------------
detect_persist_root() {
  if [[ -n "${PERSIST_ROOT:-}" ]]; then echo "$PERSIST_ROOT"; return; fi
  # DigitalOcean Volumes mount under /mnt/volume_*
  local v
  v="$(findmnt -rno TARGET --list 2>/dev/null | grep -E '^/mnt/volume' | head -n1 || true)"
  if [[ -n "$v" ]]; then echo "$v/minimax-h3"; return; fi
  # any other extra mount under /mnt
  v="$(findmnt -rno TARGET --list 2>/dev/null | grep -E '^/mnt/' | head -n1 || true)"
  if [[ -n "$v" ]]; then echo "$v/minimax-h3"; return; fi
  echo "/opt/minimax-h3"
}

PERSIST_ROOT="$(detect_persist_root)"
VENV="$PERSIST_ROOT/venv"
COMFY_DIR="$PERSIST_ROOT/ComfyUI"
MODELS_DIR="$COMFY_DIR/models"
LORA_STORE="$PERSIST_ROOT/loras"
WF_DIR="$PERSIST_ROOT/workflows"
OUT_DIR="$PERSIST_ROOT/outputs"
RUN_DIR="$PERSIST_ROOT/run"
LOG_DIR="$PERSIST_ROOT/logs"
APP_DIR="$PERSIST_ROOT/app"           # holds api_server.py

py() { "$VENV/bin/python" "$@"; }
pip() { "$VENV/bin/python" -m pip "$@"; }

# ----------------------------- service control ----------------------------
stop_services() {
  for name in api comfyui; do
    local pidf="$RUN_DIR/$name.pid"
    if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
      log "stopping $name (pid $(cat "$pidf"))"
      kill "$(cat "$pidf")" 2>/dev/null || true
      sleep 2; kill -9 "$(cat "$pidf")" 2>/dev/null || true
    fi
    rm -f "$pidf"
  done
}

start_comfyui() {
  mkdir -p "$RUN_DIR" "$LOG_DIR" "$OUT_DIR"
  # picked up on --restart too (written by the provisioning run)
  [[ -z "${COMFY_EXTRA_ARGS:-}" && -f "$RUN_DIR/comfy_extra_args" ]] && \
    COMFY_EXTRA_ARGS="$(cat "$RUN_DIR/comfy_extra_args")"
  log "starting ComfyUI on 127.0.0.1:$COMFY_PORT  (extra: '${COMFY_EXTRA_ARGS:-none}')"
  # shellcheck disable=SC2086  (COMFY_EXTRA_ARGS is intentionally word-split: e.g. "--lowvram")
  ( cd "$COMFY_DIR" && nohup "$VENV/bin/python" main.py \
        --listen 127.0.0.1 --port "$COMFY_PORT" \
        --output-directory "$OUT_DIR" ${COMFY_EXTRA_ARGS:-} \
        > "$LOG_DIR/comfyui.log" 2>&1 & echo $! > "$RUN_DIR/comfyui.pid" )
  log "waiting for ComfyUI to answer ..."
  for _ in $(seq 1 90); do
    if curl -fsS "http://127.0.0.1:$COMFY_PORT/system_stats" >/dev/null 2>&1; then
      log "ComfyUI is up"; return 0
    fi
    sleep 2
  done
  die "ComfyUI did not start — see $LOG_DIR/comfyui.log"
}

start_api() {
  mkdir -p "$RUN_DIR" "$LOG_DIR"
  [[ -n "${API_KEY:-}" ]] || die "API_KEY env var is required to start the public API"
  log "starting FastAPI on 0.0.0.0:$API_PORT"
  ( cd "$APP_DIR" && \
    COMFY_URL="http://127.0.0.1:$COMFY_PORT" \
    API_KEY="$API_KEY" \
    WF_DIR="$WF_DIR" \
    OUT_DIR="$OUT_DIR" \
    LORA_DIR="$MODELS_DIR/loras" \
    FPS="$API_FPS" \
    MAX_SECONDS="$API_MAX_SECONDS" \
    STEPS="$API_STEPS" \
    DEFAULT_WIDTH="$API_DEF_W" \
    DEFAULT_HEIGHT="$API_DEF_H" \
    DEFAULT_LORA="$(basename "$LORA_HF_FILE")" \
    nohup "$VENV/bin/python" -m uvicorn api_server:app \
        --host 0.0.0.0 --port "$API_PORT" \
        > "$LOG_DIR/api.log" 2>&1 & echo $! > "$RUN_DIR/api.pid" )
  sleep 3
  curl -fsS "http://127.0.0.1:$API_PORT/healthz" >/dev/null 2>&1 \
    && log "API is up" || warn "API health check failed — see $LOG_DIR/api.log"
}

# ----------------------------- early exits --------------------------------
if [[ "$MODE" == "stop" ]]; then stop_services; log "stopped."; exit 0; fi
if [[ "$MODE" == "restart" ]]; then stop_services; start_comfyui; start_api; exit 0; fi
if [[ "$MODE" == "list-models" ]]; then
  [[ -x "$VENV/bin/python" ]] || die "run a normal setup first"
  py - <<PY
from huggingface_hub import list_repo_files
for f in sorted(list_repo_files("$HF_REPO")):
    print(f)
PY
  exit 0
fi

# ============================ PROVISIONING ================================
[[ $EUID -eq 0 ]] || die "run with sudo -E (need apt + the -E keeps HF_TOKEN/API_KEY)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — this is not a GPU droplet or the driver is missing"

log "persistent root : $PERSIST_ROOT"
mkdir -p "$PERSIST_ROOT" "$LORA_STORE" "$WF_DIR" "$OUT_DIR" "$RUN_DIR" "$LOG_DIR" "$APP_DIR" \
         "$MODELS_DIR/diffusion_models" "$MODELS_DIR/text_encoders" "$MODELS_DIR/vae" "$MODELS_DIR/loras"

# --- 1. VRAM detection --------------------------------------------------
VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
log "detected GPU VRAM: ${VRAM_MIB} MiB"
if [[ "$VRAM_MIB" -ge 44000 ]]; then
  DIFF_FILE="$DIFF_INT8"; COMFY_VRAM_FLAG=""            # 48 GB card: INT8 fits with headroom for audio+VAE
  log "profile: 48 GB -> INT8 diffusion model, no offload"
elif [[ "$VRAM_MIB" -ge 18000 ]]; then
  DIFF_FILE="$DIFF_INT8"; COMFY_VRAM_FLAG="--lowvram"    # 20 GB card: INT8 + aggressive offload to your 64 GB RAM
  warn "profile: 20 GB -> INT8 diffusion model WITH --lowvram offload. Generations will be slower."
else
  DIFF_FILE="$DIFF_INT8"; COMFY_VRAM_FLAG="--novram"
  warn "profile: <18 GB -> --novram. Expect very slow generation; consider a bigger droplet."
fi

# --- 2. system packages ----------------------------------------------
log "installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git git-lfs curl ffmpeg aria2 build-essential \
                      python3-venv python3-dev pkg-config >/dev/null
git lfs install --skip-repo >/dev/null 2>&1 || true

# --- 3. python venv -------------------------------------------------
if [[ ! -x "$VENV/bin/python" ]]; then
  log "creating venv at $VENV"
  "$PYTHON_BIN" -m venv "$VENV"
fi
pip install -q --upgrade pip wheel setuptools

# --- 4. PyTorch ---------------------------------------------------
if ! py -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  log "installing PyTorch ($TORCH_CUDA)"
  pip install -q torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
else
  log "PyTorch with CUDA already present"
fi

# --- 5. ComfyUI ---------------------------------------------------
if [[ ! -d "$COMFY_DIR/.git" ]]; then
  log "cloning ComfyUI"
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
else
  log "updating ComfyUI"
  git -C "$COMFY_DIR" pull --ff-only || warn "could not fast-forward ComfyUI"
fi
pip install -q -r "$COMFY_DIR/requirements.txt"

# --- 6. custom nodes (video muxing + manager) ----------------------
declare -A NODES=(
  [ComfyUI-VideoHelperSuite]=https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
  [ComfyUI-Manager]=https://github.com/ltdrdata/ComfyUI-Manager.git
)
for name in "${!NODES[@]}"; do
  dir="$COMFY_DIR/custom_nodes/$name"
  if [[ ! -d "$dir/.git" ]]; then
    log "installing custom node: $name"
    git clone --depth 1 "${NODES[$name]}" "$dir"
  else
    git -C "$dir" pull --ff-only || true
  fi
  [[ -f "$dir/requirements.txt" ]] && pip install -q -r "$dir/requirements.txt" || true
done

# --- 7. API server dependencies ----------------------------------
log "installing API server dependencies"
pip install -q fastapi "uvicorn[standard]" httpx python-multipart "huggingface_hub[cli]" pillow

# --- 8. model weights ------------------------------------------
hf_get() {  # <repo> <path-in-repo> <dest-dir>
  local repo="$1" path="$2" dest="$3" fname
  fname="$(basename "$path")"
  if [[ -f "$dest/$fname" ]]; then
    log "  present: $fname"; return 0
  fi
  log "  downloading: $fname"
  py -m huggingface_hub.commands.huggingface_cli download "$repo" "$path" \
      --local-dir "$dest" --local-dir-use-symlinks False \
    || "$VENV/bin/hf" download "$repo" "$path" --local-dir "$dest"
  # flatten if HF created sub-dirs
  if [[ ! -f "$dest/$fname" ]]; then
    found="$(find "$dest" -name "$fname" -type f | head -n1 || true)"
    [[ -n "$found" ]] && mv "$found" "$dest/$fname"
  fi
}

if [[ -z "${HF_TOKEN:-}" ]]; then
  warn "HF_TOKEN is not set — download of gated H3 weights will fail. Export it and re-run."
else
  "$VENV/bin/hf" auth login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || \
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

log "fetching model weights from $HF_REPO (this is the slow part on first run)"
hf_get "$HF_REPO" "$DIFF_FILE"      "$MODELS_DIR/diffusion_models"
hf_get "$HF_REPO" "$TEXT_ENCODER"   "$MODELS_DIR/text_encoders"
hf_get "$HF_REPO" "$VIDEO_VAE"      "$MODELS_DIR/vae"
hf_get "$HF_REPO" "$AUDIO_VAE"      "$MODELS_DIR/vae"

# --- 9. LoRAs -------------------------------------------------
if [[ -n "$LORA_HF_REPO" && -n "$LORA_HF_FILE" ]]; then
  hf_get "$LORA_HF_REPO" "$LORA_HF_FILE" "$LORA_STORE"
fi
# expose every LoRA kept on the persistent volume to ComfyUI
shopt -s nullglob
for f in "$LORA_STORE"/*.safetensors "$LORA_STORE"/*.pt; do
  ln -sf "$f" "$MODELS_DIR/loras/$(basename "$f")"
done
shopt -u nullglob
log "LoRAs available: $(ls -1 "$MODELS_DIR/loras" 2>/dev/null | tr '\n' ' ' || echo '(none)')"

# --- 10. drop the API server file --------------------------------
if [[ -f "$(dirname "$0")/api_server.py" ]]; then
  cp "$(dirname "$0")/api_server.py" "$APP_DIR/api_server.py"
  log "installed api_server.py into $APP_DIR"
else
  warn "api_server.py not found next to this script — copy it into $APP_DIR manually"
fi

# --- 11. workflow templates -------------------------------------
# The API server drives ComfyUI with an API-format workflow JSON plus a small
# mapping file. These are created ONCE and then live on the persistent volume.
if [[ ! -f "$WF_DIR/t2v.json" || ! -f "$WF_DIR/i2v.json" ]]; then
  cat > "$WF_DIR/README.txt" <<'TXT'
ONE-TIME STEP - create the workflow files the API server drives. This folder is
on your persistent Volume, so you only ever do this once.

Per kind (t2v = text only, i2v = has a reference image) the server looks for:
    <kind>_audio.json   used when the request has audio_enabled=true   (optional)
    <kind>.json         fallback / used when audio_enabled=false        (required)
    <kind>.map.json     REQUIRED - maps API params -> node inputs (see .example)
If a single <kind>.json can toggle audio internally, add an "audio_enabled"
entry to <kind>.map.json pointing at that boolean and skip the *_audio.json file.

STEPS
1. SSH tunnel to the UI:  ssh -L 8188:127.0.0.1:8188 root@<droplet-ip>
   then open http://127.0.0.1:8188
2. Workflow > Browse Templates > Video > "MiniMax H3".
     text-to-video : the T2V template.
     image-to-video: the first-frame (FL2VA) template - it has a Load Image node.
3. Turbo LoRA wiring (this script installs minimax_h3_turbo_v4_step600_ema.safetensors):
     - insert a LoraLoader between the model loader and the sampler
     - drive SamplerCustomAdvanced from the "MiniMax-H3 Turbo Sampler" node
     - scheduler = simple, steps >= 4  (API default is 6)
4. Queue it once to confirm it renders video (and audio, in the audio graph).
5. Settings (gear) > enable "Dev mode" (adds the "Save (API Format)" button).
6. "Save (API Format)" into THIS folder as t2v.json / t2v_audio.json /
   i2v.json / i2v_audio.json as appropriate.
7. Copy t2v.map.json.example -> t2v.map.json (and the i2v pair) and edit the node
   ids / input names to match YOUR saved graph. List only params you want the API
   to control. Node ids are the object keys in the API-format JSON you just saved.
TXT
  cat > "$WF_DIR/t2v.map.json.example" <<'JSON'
{
  "_comment": "Replace node ids/inputs with the ones from YOUR Save(API Format) json. Delete lines you do not want the API to drive.",
  "prompt":        {"node": "6",  "input": "text"},
  "negative":      {"node": "7",  "input": "text"},
  "seed":          {"node": "3",  "input": "noise_seed"},
  "steps":         {"node": "3",  "input": "steps"},
  "width":         {"node": "5",  "input": "width"},
  "height":        {"node": "5",  "input": "height"},
  "frames":        {"node": "5",  "input": "length"},
  "fps":           {"node": "40", "input": "frame_rate"},
  "lora":          {"node": "10", "input": "lora_name"},
  "lora_strength": {"node": "10", "input": "strength_model"},
  "audio_enabled": {"node": "44", "input": "enabled"},
  "image":         {"node": "12", "input": "image"}
}
JSON
  cp "$WF_DIR/t2v.map.json.example" "$WF_DIR/i2v.map.json.example"
  warn "No workflow files yet. Read $WF_DIR/README.txt and create them once."
fi

# --- 12. firewall reminder ------------------------------------
log "provisioning complete."
cat <<EOF

  Persistent root : $PERSIST_ROOT
  Diffusion model : $DIFF_FILE   (VRAM ${VRAM_MIB} MiB, comfy flag: '${COMFY_VRAM_FLAG:-none}')
  ComfyUI (local) : http://127.0.0.1:$COMFY_PORT
  Public API      : http://0.0.0.0:$API_PORT     (protect with the DigitalOcean Cloud Firewall!)

  Open ONLY port $API_PORT to your own IP in the DigitalOcean firewall. Do NOT expose $COMFY_PORT.
EOF

mkdir -p "$RUN_DIR"
printf '%s' "$COMFY_VRAM_FLAG" > "$RUN_DIR/comfy_extra_args"

if [[ "$MODE" == "setup-only" ]]; then exit 0; fi

# --- 13. launch --------------------------------------------
export COMFY_EXTRA_ARGS="$COMFY_VRAM_FLAG"
stop_services
start_comfyui
start_api

cat <<EOF

Ready. One public endpoint, always multipart/form-data.

  Text-to-video (no image):
    curl -s -X POST http://<droplet-ip>:$API_PORT/v1/videos \\
      -H "X-API-Key: \$API_KEY" \\
      -F prompt="a red fox trotting through fresh snow, cinematic, shallow depth of field" \\
      -F duration=8 -F audio_enabled=true

  Image-to-video (reference image as form field):
    curl -s -X POST http://<droplet-ip>:$API_PORT/v1/videos \\
      -H "X-API-Key: \$API_KEY" \\
      -F prompt="the camera slowly pushes in, leaves drift past" \\
      -F duration=6 -F audio_enabled=false \\
      -F reference_image=@/path/to/first_frame.png

  Poll until status=succeeded, then GET output_url:
    curl -s http://<droplet-ip>:$API_PORT/v1/videos/<id> -H "X-API-Key: \$API_KEY"
    curl -s -OJ http://<droplet-ip>:$API_PORT/files/<id>.mp4 -H "X-API-Key: \$API_KEY"

  duration is in SECONDS, capped at $API_MAX_SECONDS (raise API_MAX_SECONDS to override).
  Logs: $LOG_DIR/comfyui.log , $LOG_DIR/api.log
EOF
