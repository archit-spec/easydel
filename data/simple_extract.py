#!/usr/bin/env python3
import argparse, gzip, json, os, subprocess, sys
from pathlib import Path
from datetime import datetime

def run_cmd(cmd, cwd=None, check=True, timeout=30):
    """Run command with timeout"""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            timeout=timeout, check=check
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"⏰ Command timed out: {' '.join(cmd)}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        return None

def extract_commits_simple(repo_path, out_dir, max_commits=1000):
    """Extract commits using simple git commands"""
    print("📝 Extracting commits with git log...")

    commits_dir = out_dir / "commits"
    commits_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Get commit hashes first
        hash_cmd = ["git", "log", "--all", "--pretty=format:%H", "--no-merges", f"-{max_commits}"]
        commit_hashes = run_cmd(hash_cmd, cwd=repo_path, timeout=60)
        if not commit_hashes:
            print("❌ No commit hashes received")
            return []

        commit_list = commit_hashes.strip().split('\n')
        commits = []

        with gzip.open(commits_dir / "commits-0000.jsonl.gz", "wt") as f:
            for hash in commit_list:
                if not hash.strip():
                    continue

                # Get commit details
                show_cmd = ["git", "show", "--no-patch", "--pretty=format:%an|%ae|%at|%s", hash]
                commit_info = run_cmd(show_cmd, cwd=repo_path, timeout=30)

                if commit_info:
                    parts = commit_info.split('|')
                    if len(parts) >= 4:
                        commit_data = {
                            "hash": hash,
                            "author_name": parts[0],
                            "author_email": parts[1],
                            "authored_date": datetime.fromtimestamp(int(parts[2])).isoformat(),
                            "message": '|'.join(parts[3:]),
                            "in_main_branch": True,
                            "merge": False,
                            "project": "juspay/hyperswitch",
                            "modifications": []
                        }

                        # Get file modifications
                        name_status_cmd = ["git", "show", "--name-status", "--pretty=format:", hash]
                        file_changes = run_cmd(name_status_cmd, cwd=repo_path, timeout=30)

                        if file_changes:
                            for line in file_changes.strip().split('\n'):
                                if line.strip() and '\t' in line:
                                    status, filename = line.split('\t', 1)
                                    mod = {
                                        "commit": hash,
                                        "old_path": None,
                                        "new_path": filename,
                                        "change_type": status,
                                        "added": 0,  # We'll calculate this from diff later if needed
                                        "removed": 0,
                                        "filename": filename
                                    }
                                    commit_data["modifications"].append(mod)

                        commits.append(commit_data)
                        f.write(json.dumps(commit_data, ensure_ascii=False) + '\n')

        print(f"✅ Extracted {len(commits)} commits with modifications")
        return commits

    except Exception as e:
        print(f"❌ Error extracting commits: {e}")
        return []

def extract_patches_simple(repo_path, commits, out_dir):
    """Extract patches for commits"""
    print("🔧 Extracting patches...")

    patches_dir = out_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    patch_count = 0
    for commit in commits[:100]:  # Limit to first 100 commits for testing
        try:
            # Get patch
            patch_output = run_cmd(
                ["git", "show", "--no-merges", "--pretty=format:", commit["hash"]],
                cwd=repo_path, timeout=60
            )

            if patch_output and len(patch_output.strip()) > 10:
                # Save patch
                patch_data = {
                    "commit": commit["hash"],
                    "patch": patch_output
                }

                patch_prefix = commit["hash"][:2]
                patch_subdir = patches_dir / patch_prefix
                patch_subdir.mkdir(exist_ok=True)

                patch_file = patch_subdir / f"{commit['hash']}.patch.jsonl.gz"
                with gzip.open(patch_file, "wt") as f:
                    f.write(json.dumps(patch_data, ensure_ascii=False) + '\n')

                patch_count += 1

                if patch_count % 10 == 0:
                    print(f"   Processed {patch_count} patches...")

        except Exception as e:
            print(f"   ⚠️  Skipped patch for {commit['hash'][:8]}: {e}")
            continue

    print(f"✅ Extracted {patch_count} patches")
    return patch_count

def main():
    ap = argparse.ArgumentParser(description="Simple git extraction (no PyDriller)")
    ap.add_argument("--repo", required=True, help="Git repository path")
    ap.add_argument("--workdir", required=True, help="Working directory")
    ap.add_argument("--max-commits", type=int, default=500, help="Maximum commits to extract")
    args = ap.parse_args()

    workdir = Path(args.workdir).absolute()
    repo_path = Path(args.repo).absolute()
    out_dir = workdir / "dataset"

    # Ensure repo exists
    if not repo_path.exists():
        print(f"❌ Repository path does not exist: {repo_path}")
        sys.exit(1)

    if not (repo_path / ".git").exists():
        print(f"❌ Not a git repository: {repo_path}")
        sys.exit(1)

    print(f"🔍 Repository: {repo_path}")
    print(f"📁 Output: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract commits
    commits = extract_commits_simple(repo_path, out_dir, args.max_commits)

    if not commits:
        print("❌ No commits extracted. Exiting.")
        sys.exit(1)

    # Extract patches
    patch_count = extract_patches_simple(repo_path, commits, out_dir)

    print("\n📊 Simple Extraction Complete:")
    print(f"   Commits: {len(commits)}")
    print(f"   Patches: {patch_count}")
    print(f"   Output: {out_dir}")

    # Show sample data
    print("\n📋 Sample Data:")
    if commits:
        print(f"   Latest commit: {commits[0]['hash'][:8]} - {commits[0]['message'][:50]}...")

    print("\n✅ Ready for upload! Use:")
    print(f"   uv run data/push_complete.py --workdir {workdir} --hf-namespace YOUR_NAMESPACE --hf-dataset YOUR_DATASET")

if __name__ == "__main__":
    main()