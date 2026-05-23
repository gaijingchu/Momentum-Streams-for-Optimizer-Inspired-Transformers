"""Download missing checkpoints from HF to $CACHE on the compute node."""
import os
from huggingface_hub import hf_hub_download
from pathlib import Path

assert os.environ.get("HF_TOKEN", "").startswith("hf_"), "HF_TOKEN not set"
CACHE = Path(os.environ["CACHE"])

needed_owt = [
    "checkpoints_vanilla_owt",
    "checkpoints_vanilla_adamw_owt",
    "checkpoints_adam_owt",
    "checkpoints_adamw_owt",
    "checkpoints_tmm_owt",
    "checkpoints_yurii_owt",
    "checkpoints_tmm_sam_owt",
    "checkpoints_tmm_wsd_owt",
    "checkpoints_tmm_sawd_owt",
    "checkpoints_yurii_sam_owt",
    "checkpoints_yurii_wsd_owt",
    "checkpoints_yurii_sawd_owt",
    # TS-pretrained TMM-SAM/WSD/SAWD have their own repos (not in ts_archive)
    "checkpoints_tmm_sam_ts",
    "checkpoints_tmm_wsd_ts",
    "checkpoints_tmm_sawd_ts",
]

for repo_name in needed_owt:
    local_dir = CACHE / repo_name
    if (local_dir / "best.pt").exists():
        print(f"  SKIP {repo_name} (best.pt already present)")
        continue
    local_dir.mkdir(parents=True, exist_ok=True)
    repo_id = f"gaijingchu/{repo_name}"
    try:
        path = hf_hub_download(repo_id=repo_id, filename="best.pt",
                               local_dir=str(local_dir), repo_type="model")
        sz = os.path.getsize(path) / 1e9
        print(f"  ✓ {repo_name}/best.pt  ({sz:.2f} GB)", flush=True)
    except Exception as e:
        print(f"  ✗ {repo_name}: {type(e).__name__}: {e}", flush=True)

ts_dir = CACHE / "checkpoints_ts_archive"
ts_dir.mkdir(exist_ok=True)
ts_variants = ["vanilla_ts", "adam_ts", "adamw_ts", "tmm_ts", "yurii_ts",
               "yurii_sam_ts", "yurii_wsd_ts", "yurii_sawd_ts"]
for v in ts_variants:
    local = ts_dir / v / "best.pt"
    if local.exists():
        print(f"  SKIP ts_archive/{v}/best.pt")
        continue
    try:
        path = hf_hub_download(repo_id="gaijingchu/checkpoints_ts_archive",
                               filename=f"{v}/best.pt",
                               local_dir=str(ts_dir), repo_type="model")
        sz = os.path.getsize(path) / 1e9
        print(f"  ✓ ts_archive/{v}/best.pt  ({sz:.2f} GB)", flush=True)
    except Exception as e:
        print(f"  ✗ ts_archive/{v}: {type(e).__name__}: {e}", flush=True)

# MuonFormer (and SOAP) live in their own non-standard repo with a different
# internal layout: gaijingchu/ANLP-Yuriiformer-reproduce-MuonFormer holds
# checkpoints_muon/best.pt (TS-trained) at the repo root.  Map it into the
# ts_archive convention used by the eval scripts.
muon_target = ts_dir / "muon_ts" / "best.pt"
if muon_target.exists():
    print(f"  SKIP ts_archive/muon_ts/best.pt")
else:
    muon_target.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = hf_hub_download(repo_id="gaijingchu/ANLP-Yuriiformer-reproduce-MuonFormer",
                               filename="checkpoints_muon/best.pt",
                               local_dir=str(muon_target.parent.parent / "_muonformer_repo"),
                               repo_type="model")
        # hf_hub_download places it at <local_dir>/checkpoints_muon/best.pt;
        # symlink/copy into the ts_archive layout expected by VARIANTS tables.
        import shutil
        shutil.copy2(path, muon_target)
        sz = os.path.getsize(muon_target) / 1e9
        print(f"  ✓ ts_archive/muon_ts/best.pt  ({sz:.2f} GB)", flush=True)
    except Exception as e:
        print(f"  ✗ ts_archive/muon_ts: {type(e).__name__}: {e}", flush=True)

print("=== Download complete ===", flush=True)
