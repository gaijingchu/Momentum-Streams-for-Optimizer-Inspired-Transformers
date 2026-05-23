"""Upload the new factorial-ablation checkpoints to HF Hub under gaijingchu/.

OWT variants → one repo per variant (gaijingchu/checkpoints_{name}).
TS variants → subdirs under gaijingchu/checkpoints_ts_archive/{name}/best.pt.

Reads HF_TOKEN from env (no token persisted on disk).
"""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN", "")
assert token.startswith("hf_"), "HF_TOKEN env var not set"

CACHE = Path(os.environ.get("CACHE", "./checkpoints_cache"))
api = HfApi(token=token)

# OWT variants → one repo each
owt_uploads = [
    ("hb_owt",      "checkpoints_hb_owt"),
    ("rmsprop_owt", "checkpoints_rmsprop_owt"),
]
print(f"\n=== OWT uploads ===", flush=True)
for name, dirname in owt_uploads:
    src = CACHE / dirname / "best.pt"
    if not src.exists():
        print(f"  ✗ SKIP {name}: {src} does not exist", flush=True)
        continue
    repo_id = f"gaijingchu/checkpoints_{name}"
    sz = src.stat().st_size / 1e9
    print(f"  Uploading {name} ({sz:.2f} GB) → {repo_id} ...", flush=True)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo="best.pt",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  ✓ {name} done", flush=True)

# TS variants → ts_archive repo, one subdir per variant
ts_uploads = [
    ("hb_ts",      "checkpoints_hb_ts"),
    ("rmsprop_ts", "checkpoints_rmsprop_ts"),
    ("ortho_ts",   "checkpoints_ortho_ts"),
]
archive_repo = "gaijingchu/checkpoints_ts_archive"
print(f"\n=== TS uploads → {archive_repo} ===", flush=True)
api.create_repo(repo_id=archive_repo, repo_type="model", exist_ok=True, private=False)
for name, dirname in ts_uploads:
    src = CACHE / dirname / "best.pt"
    if not src.exists():
        print(f"  ✗ SKIP {name}: {src} does not exist", flush=True)
        continue
    sz = src.stat().st_size / 1e9
    print(f"  Uploading {name} ({sz:.2f} GB) → {archive_repo}/{name}/best.pt ...", flush=True)
    api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo=f"{name}/best.pt",
        repo_id=archive_repo,
        repo_type="model",
    )
    print(f"  ✓ {name} done", flush=True)

print("\n=== All uploads complete ===", flush=True)
