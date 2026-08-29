# minimax-h3-droplet

Provisioning for MiniMax H3 (text/image -> video + audio) + Turbo LoRA + a REST API,
built to run on an ephemeral GPU pod that is rebuilt every session (no persistent volume).

## Files
- `setup_minimax_h3.sh` - one command: installs deps, ComfyUI, downloads weights, starts the API.
- `api_server.py` - FastAPI wrapper. One endpoint `POST /v1/videos` (multipart): `prompt`,
  `duration` (seconds), `audio_enabled`, optional `reference_image`.
- `workflows/` - after the one-time ComfyUI "Save (API Format)" export, commit
  `t2v.json` / `i2v.json` (+ `_audio` variants) and their `*.map.json` here.
  `setup_minimax_h3.sh` re-seeds them into the pod on every run.

## Per session (RunPod A40)
```bash
cd /workspace && git clone <this-repo-url> src && cd src
bash setup_minimax_h3.sh
```
Needs env vars `HF_TOKEN` and `API_KEY` (set them in the pod template).
