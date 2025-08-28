#!/usr/bin/env python3
import argparse, os, sys
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, create_repo, upload_folder

def check_dataset_quality(out_dir):
    """Check what data is available in the dataset"""
    dataset_path = Path(out_dir)
    if not dataset_path.exists():
        print(f"❌ Dataset directory {out_dir} does not exist")
        return False, False

    print("📊 Dataset Quality Check:")
    has_commits = False
    has_patches = False

    # Check commits
    commits_dir = dataset_path / "commits"
    if commits_dir.exists():
        commit_files = list(commits_dir.glob("*.jsonl.gz"))
        print(f"   ✅ Commits: {len(commit_files)} files")
        if commit_files:
            # Check if files have content
            import gzip
            try:
                with gzip.open(commit_files[0], 'rt') as f:
                    first_line = f.readline().strip()
                    if first_line and len(first_line) > 10:  # Must have actual content
                        print("   ✅ Commit data appears to have content")
                        has_commits = True
                    else:
                        print("   ⚠️  Commit files appear to be empty")
            except:
                print("   ⚠️  Could not read commit files")
    else:
        print("   ❌ Commits: No commits directory found")

    # Check patches (actual code diffs)
    patches_dir = dataset_path / "patches"
    if patches_dir.exists():
        patch_files = list(patches_dir.rglob("*.patch.jsonl.gz"))
        print(f"   ✅ Patches: {len(patch_files)} files")
        if patch_files:
            print("   ✅ Code diffs available!")
            has_patches = True
        else:
            print("   ⚠️  No patch files found (no code diffs extracted)")
    else:
        print("   ❌ Patches: No patches directory found")

    # Check snapshots (file content)
    snapshots_dir = dataset_path / "snapshots"
    if snapshots_dir.exists():
        snapshot_files = list(snapshots_dir.rglob("*.gz"))
        print(f"   ✅ Snapshots: {len(snapshot_files)} files")
        if snapshot_files:
            print("   ✅ File content available!")
    else:
        print("   ❌ Snapshots: No snapshots directory found")

    # Check pretraining dataset
    pretraining_dir = dataset_path / "pretraining"
    if pretraining_dir.exists():
        pretraining_files = list(pretraining_dir.glob("*.jsonl.gz"))
        print(f"   ✅ Pretraining: {len(pretraining_files)} files")
        if pretraining_files:
            print("   ✅ Pretraining dataset available!")
    else:
        print("   ❌ Pretraining: No pretraining directory found")

    # Require at least commits to consider it a valid dataset
    is_valid = has_commits
    if not is_valid:
        print("   ❌ Dataset is incomplete - no valid commit data found")

    return is_valid, has_patches

def push_to_hf(out_dir, hf_namespace, hf_dataset, hf_token, force=False):
    """Push dataset to Hugging Face with quality checks"""
    dataset_path = Path(out_dir)

    is_valid, has_patches = check_dataset_quality(out_dir)
    if not is_valid:
        if not force:
            print("❌ Dataset quality check failed. Use --force to push anyway.")
            return False
        else:
            print("⚠️  Pushing despite quality issues (--force enabled)")

    if has_patches:
        print("🎉 Dataset includes actual code diffs - perfect for training!")
    else:
        print("⚠️  Dataset has commits but no code diffs. Consider re-running with patch extraction enabled.")

    repo_id = f"{hf_namespace}/{hf_dataset}"
    print(f"\n🚀 Pushing to HF dataset: {repo_id}")

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
            folder_path=str(dataset_path),
            token=hf_token,
            commit_message=f"Add complete hyperswitch dataset at {datetime.utcnow().isoformat()}Z"
        )
        print("✅ Upload completed successfully!")
        print(f"🔗 View at: https://huggingface.co/datasets/{repo_id}")
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def main():
    ap = argparse.ArgumentParser(description="Push complete dataset to Hugging Face")
    ap.add_argument("--workdir", required=True, help="Working directory containing dataset/")
    ap.add_argument("--hf-namespace", required=True, help="HF username/org")
    ap.add_argument("--hf-dataset", required=True, help="HF dataset name")
    ap.add_argument("--force", action="store_true", help="Force push even if quality checks fail")
    args = ap.parse_args()

    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        print("❌ ERROR: set HF_TOKEN environment variable", file=sys.stderr)
        sys.exit(1)

    workdir = Path(args.workdir).absolute()
    out_dir = workdir / "dataset"

    print(f"🔍 Checking dataset in: {out_dir}")

    if not out_dir.exists():
        print(f"❌ Dataset directory {out_dir} does not exist")
        print("Make sure the extraction completed successfully first")
        sys.exit(1)

    success = push_to_hf(str(out_dir), args.hf_namespace, args.hf_dataset, HF_TOKEN, args.force)

    if success:
        print("\n🎉 Dataset successfully pushed to Hugging Face!")
        print("\n📋 What you can now access:")
        print("   • Code diffs: patches/ directory")
        print("   • File content: snapshots/ directory")
        print("   • Pretraining data: pretraining/ directory")
        print("   • Commit metadata: commits/ directory")
        print("   • File changes: modifications/ directory")
    else:
        print("\n❌ Push failed. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()