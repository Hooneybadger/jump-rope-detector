"""Fetch pinned browser assets; run once before deploying (no runtime CDN)."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / "static" / "vendor" / "mediapipe"
VERSION = "0.10.21"
BASE = f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{VERSION}"
FILES = {"LICENSE": "https://raw.githubusercontent.com/google-ai-edge/mediapipe/v0.10.21/LICENSE",
         "vision_bundle.mjs": f"{BASE}/vision_bundle.mjs",
         **{f"wasm/{name}": f"{BASE}/wasm/{name}" for name in (
             "vision_wasm_internal.js", "vision_wasm_internal.wasm",
             "vision_wasm_nosimd_internal.js", "vision_wasm_nosimd_internal.wasm")},
         "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"}

def main():
    manifest = {"tasks_vision_version": VERSION, "files": {}}
    for name, url in FILES.items():
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        path.write_bytes(data)
        manifest["files"][name] = {"url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        print(name, len(data), flush=True)
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
