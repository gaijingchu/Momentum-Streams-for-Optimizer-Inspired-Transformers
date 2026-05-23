"""Upload the vanilla-AdamW-OWT 30k-step checkpoint to HF Hub.

src:  ${CACHE}/checkpoints_vanilla_adamw_owt/best.pt
dst:  gaijingchu/checkpoints_vanilla_adamw_owt   (public model repo)

HF_TOKEN must be in the environment; no token is written to any file.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
from pathlib import Path

from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN", "")
assert token.startswith("hf_"), "HF_TOKEN env var not set"

CACHE = Path(os.environ.get("CACHE", "./checkpoints_cache"))
api = HfApi(token=token)

src = CACHE / "checkpoints_vanilla_adamw_owt" / "best.pt"
if not src.exists():
    raise SystemExit(f"✗ checkpoint missing: {src}")

repo_id = "gaijingchu/checkpoints_vanilla_adamw_owt"
sz = src.stat().st_size / 1e9
print(f"Uploading {src} ({sz:.2f} GB) → {repo_id} ...", flush=True)
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
api.upload_file(
    path_or_fileobj=str(src),
    path_in_repo="best.pt",
    repo_id=repo_id,
    repo_type="model",
)
print(f"✓ done: https://huggingface.co/{repo_id}", flush=True)
