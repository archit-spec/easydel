#!/usr/bin/env python3
import argparse, os, sys, shutil
from pathlib import Path

def find_latest_backup(workdir):
    """Find the latest backup directory"""
    workdir_path = Path(workdir)
    backups = list(workdir_path.glob("backup_batch_*"))

    if not backups:
        print("❌ No backup directories found")
        return None

    # Find the latest backup
    latest_backup = max(backups, key=lambda x: int(x.name.split('_')[-1]))
    print(f"📁 Found latest backup: {latest_backup.name}")
    return latest_backup

def resume_from_backup(workdir, latest_backup):
    """Resume extraction from backup"""
    dataset_dir = Path(workdir) / "dataset"
    backup_dataset = latest_backup / "dataset"

    if not backup_dataset.exists():
        print("❌ No dataset in backup directory")
        return False

    # Restore from backup
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    shutil.copytree(backup_dataset, dataset_dir)
    print(f"✅ Restored dataset from {latest_backup.name}")

    # Count existing commits
    commits_dir = dataset_dir / "commits"
    if commits_dir.exists():
        import gzip
        commit_files = list(commits_dir.glob("*.jsonl.gz"))
        commit_count = 0
        for cf in commit_files:
            try:
                with gzip.open(cf, 'rt') as f:
                    commit_count += sum(1 for line in f if line.strip())
            except:
                pass
        print(f"📊 Backup contains {commit_count} commits")
        return commit_count
    else:
        print("⚠️  No commits directory in backup")
        return 0

def main():
    ap = argparse.ArgumentParser(description="Resume extraction from backup")
    ap.add_argument("--workdir", required=True, help="Working directory")
    ap.add_argument("--continue-extraction", action="store_true", help="Continue extraction after resume")
    args = ap.parse_args()

    workdir = Path(args.workdir).absolute()

    if not workdir.exists():
        print(f"❌ Work directory does not exist: {workdir}")
        sys.exit(1)

    # Find latest backup
    latest_backup = find_latest_backup(workdir)
    if not latest_backup:
        print("❌ No backups found. Start a fresh extraction.")
        sys.exit(1)

    # Resume from backup
    commit_count = resume_from_backup(str(workdir), latest_backup)
    if commit_count == 0:
        print("❌ No valid data in backup")
        sys.exit(1)

    print(f"\n🎯 Backup restored with {commit_count} commits")
    print("You can now:"
    print("1. Push this data: uv run data/push_complete.py --workdir {workdir} --hf-namespace YOUR_NAMESPACE --hf-dataset YOUR_DATASET"
    print("2. Continue extraction: uv run data/extract.py --repo https://github.com/juspay/hyperswitch.git --workdir {workdir} [other args]"
    print("3. Just use the backup data as is"

if __name__ == "__main__":
    main()