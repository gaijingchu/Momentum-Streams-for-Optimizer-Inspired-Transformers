"""Upload all checkpoints under $CACHE to HuggingFace, then delete those
whose variant has no active SLURM job (training OR eval)."""

import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, create_repo

CACHE = Path(os.environ["CACHE"])
HF_USER = "gaijingchu"
api = HfApi()


def active_job_names() -> list[str]:
    r = subprocess.run(
        ["squeue", "-u", os.environ["USER"], "-h", "-o", "%j"],
        capture_output=True, text=True, check=True,
    )
    return [n for n in r.stdout.strip().splitlines() if n]


def variant_has_active_job(dirname: str, jobs: list[str]) -> bool:
    # e.g. dirname = "checkpoints_tmm_sam_ts"  -> token = "tmm-sam-ts"
    token = dirname.replace("checkpoints_", "").replace("_", "-")
    return any(token in j for j in jobs)


def main() -> None:
    jobs = active_job_names()
    print(f"Active SLURM job names ({len(jobs)}):")
    for j in jobs:
        print(f"  {j}")
    print()

    dirs = sorted(p for p in CACHE.glob("checkpoints_*") if p.is_dir())
    print(f"Found {len(dirs)} checkpoint directories under {CACHE}")
    print()

    kept, deleted, failed = [], [], []
    for d in dirs:
        repo_id = f"{HF_USER}/{d.name}"
        print(f"=== {d.name} -> {repo_id} ===")
        try:
            create_repo(repo_id, private=True, exist_ok=True, repo_type="model")
            api.upload_folder(
                folder_path=str(d),
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"upload {d.name}",
            )
            print("  upload OK")
        except Exception as e:
            print(f"  UPLOAD FAILED: {e}")
            failed.append(d.name)
            continue

        if variant_has_active_job(d.name, jobs):
            print("  KEEP local (matches active queue entry)")
            kept.append(d.name)
        else:
            print("  DELETE local")
            shutil.rmtree(d)
            deleted.append(d.name)
        print()

    print("=== summary ===")
    print(f"uploaded + kept  : {len(kept)}")
    for n in kept:
        print(f"  {n}")
    print(f"uploaded + deleted: {len(deleted)}")
    for n in deleted:
        print(f"  {n}")
    print(f"upload failed    : {len(failed)}")
    for n in failed:
        print(f"  {n}")


if __name__ == "__main__":
    main()
