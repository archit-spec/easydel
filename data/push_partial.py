#!/usr/bin/env python3
import argparse, os, sys
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, create_repo, upload_folder

def push_to_hf(out_dir, hf_namespace, hf_dataset, hf_token):
    repo_id = f"{hf_namespace}/{hf_dataset}"
    print(f"Pushing to HF dataset: {repo_id}")
    api = HfApi()usls
    try:
        create_repo(repo_id=repo_id, token=hf_token, repo_type="dataset", exist_ok=True)
    except Exception:
        pass
    upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(out_dir),
        token=hf_token,
        commit_message=f"Add partial hyperswitch extracted dataset at {datetime.utcnow().isoformat()}Z"
    )
    print("Done ✅")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True, help="Working directory")
    ap.add_argument("--hf-namespace", required=True, help="HF username/org")
    ap.add_argument("--hf-dataset", required=True, help="HF dataset name")
    args = ap.parse_args()

    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        print("ERROR: set HF_TOKEN env var.", file=sys.stderr); exit(1)

    workdir = Path(args.workdir).absolute()
    out_dir = workdir / "dataset"

    if not out_dir.exists():
        print(f"ERROR: {out_dir} does not exist", file=sys.stderr); exit(1)

    push_to_hf(out_dir, args.hf_namespace, args.hf_dataset, HF_TOKEN)

if __name__ == "__main__":
    main()