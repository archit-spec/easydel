#!/usr/bin/env python3
import argparse, os, sys, shutil
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, create_repo, upload_folder

def push_incremental_update(workdir, hf_namespace, hf_dataset, hf_token, backup_name=None):
    """Push incremental update from backup or current data"""

    if backup_name:
        # Push from specific backup
        backup_dir = Path(workdir) / backup_name
        if not backup_dir.exists():
            print(f"❌ Backup directory not found: {backup_dir}")
            return False

        dataset_dir = backup_dir / "dataset"
        if not dataset_dir.exists():
            print(f"❌ No dataset in backup: {backup_dir}")
            return False

        source_dir = dataset_dir
        commit_msg = f"Incremental update from {backup_name} at {datetime.utcnow().isoformat()}Z"
    else:
        # Push from current dataset
        dataset_dir = Path(workdir) / "dataset"
        if not dataset_dir.exists():
            print(f"❌ No dataset directory found in {workdir}")
            return False
        source_dir = dataset_dir
        commit_msg = f"Incremental update at {datetime.utcnow().isoformat()}Z"

    # Check what's in the dataset
    commits_dir = source_dir / "commits"
    commit_count = 0
    if commits_dir.exists():
        import gzip
        for cf in commits_dir.glob("*.jsonl.gz"):
            try:
                with gzip.open(cf, 'rt') as f:
                    commit_count += sum(1 for line in f if line.strip())
            except:
                pass

    if commit_count == 0:
        print("❌ No commits found in dataset")
        return False

    print(f"📊 Pushing {commit_count} commits from {source_dir.name}")

    # Push to Hugging Face
    repo_id = f"{hf_namespace}/{hf_dataset}"

    api = HfApi()
    try:
        create_repo(repo_id=repo_id, token=hf_token, repo_type="dataset", exist_ok=True)
        print("✅ Repository ready")
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")
        return False

    try:
        upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(source_dir),
            token=hf_token,
            commit_message=commit_msg
        )
        print("✅ Incremental update pushed successfully!")
        print(f"🔗 View at: https://huggingface.co/datasets/{repo_id}")
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def list_backups(workdir):
    """List available backup directories"""
    workdir_path = Path(workdir)
    backups = sorted(workdir_path.glob("backup_batch_*"), key=lambda x: int(x.name.split('_')[-1]))

    if not backups:
        print("❌ No backup directories found")
        return

    print("📁 Available backups:")
    for backup in backups:
        dataset_dir = backup / "dataset"
        commit_count = 0

        if dataset_dir.exists():
            commits_dir = dataset_dir / "commits"
            if commits_dir.exists():
                import gzip
                for cf in commits_dir.glob("*.jsonl.gz"):
                    try:
                        with gzip.open(cf, 'rt') as f:
                            commit_count += sum(1 for line in f if line.strip())
                    except:
                        pass

        print(f"   {backup.name}: {commit_count} commits")

def main():
    ap = argparse.ArgumentParser(description="Push incremental updates to Hugging Face")
    ap.add_argument("--workdir", required=True, help="Working directory")
    ap.add_argument("--hf-namespace", required=True, help="HF username/org")
    ap.add_argument("--hf-dataset", required=True, help="HF dataset name")
    ap.add_argument("--backup", help="Specific backup to push (e.g., backup_batch_10)")
    ap.add_argument("--list-backups", action="store_true", help="List available backups")

    args = ap.parse_args()

    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        print("❌ ERROR: set HF_TOKEN environment variable", file=sys.stderr)
        sys.exit(1)

    workdir = Path(args.workdir).absolute()

    if args.list_backups:
        list_backups(workdir)
        return

    success = push_incremental_update(
        str(workdir),
        args.hf_namespace,
        args.hf_dataset,
        HF_TOKEN,
        args.backup
    )

    if success:
        print("\n🎉 Incremental update completed!")
        if args.backup:
            print(f"✅ Pushed data from {args.backup}")
        else:
            print("✅ Pushed current dataset")
    else:
        print("\n❌ Push failed")
        sys.exit(1)

if __name__ == "__main__":
    main()