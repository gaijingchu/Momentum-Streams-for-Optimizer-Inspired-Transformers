"""Download the 4 base YuriiFormer/TMMFormer checkpoints from HF into $CACHE.

OWT checkpoints live in their own repos; TS checkpoints are bundled in
`checkpoints_ts_archive/<variant>/best.pt` — handle both cases.
"""

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

CACHE = Path(os.environ["CACHE"])
HF_USER = "gaijingchu"

# (local dirname, hf repo, path within repo)
JOBS = [
    ("checkpoints_yurii_owt", f"{HF_USER}/checkpoints_yurii_owt",   "best.pt"),
    ("checkpoints_tmm_owt",   f"{HF_USER}/checkpoints_tmm_owt",     "best.pt"),
    ("checkpoints_yurii_ts",  f"{HF_USER}/checkpoints_ts_archive",  "yurii_ts/best.pt"),
    ("checkpoints_tmm_ts",    f"{HF_USER}/checkpoints_ts_archive",  "tmm_ts/best.pt"),
]

for local_dir, repo, path_in_repo in JOBS:
    target_dir = CACHE / local_dir
    target_file = target_dir / "best.pt"
    if target_file.exists():
        print(f"[skip] {target_file} already present ({target_file.stat().st_size / 1e9:.2f} GB)")
        continue
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dl]   {repo} :: {path_in_repo}  ->  {target_file}")
    got = hf_hub_download(repo_id=repo, filename=path_in_repo, local_dir=str(target_dir))
    if path_in_repo != "best.pt":
        # hf_hub_download writes to <local_dir>/<path_in_repo>; move to best.pt
        os.replace(got, target_file)
        # clean up empty sub-directories left behind
        parent = Path(got).parent
        if parent != target_dir and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    print(f"       done, size={target_file.stat().st_size / 1e9:.2f} GB")
