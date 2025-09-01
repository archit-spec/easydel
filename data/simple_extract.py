#!/usr/bin/env python3
import argparse, gzip, json, os, re, subprocess, sys
import urllib.request, urllib.error
from pathlib import Path
from datetime import datetime
import dotenv

dotenv.load_dotenv()

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

def get_repo_remote_info(repo_path):
    url = run_cmd(["git", "config", "--get", "remote.origin.url"], cwd=repo_path, timeout=10)
    if not url:
        return None
    owner = repo = host = None
    u = url.strip()
    path = None
    if u.startswith("git@"):
        try:
            host_part, path_part = u.split(":", 1)
            host = host_part.split("@", 1)[1]
            path = path_part
        except ValueError:
            path = None
    elif u.startswith("http://") or u.startswith("https://"):
        try:
            parts = u.split("://", 1)[1].split("/", 1)
            host = parts[0]
            path = parts[1] if len(parts) > 1 else None
        except Exception:
            path = None
    if path:
        path = path.strip("/")
        if path.endswith(".git"):
            path = path[:-4]
        if path.count("/") >= 1:
            owner, repo = path.split("/", 1)
            repo = repo.strip("/")
    if host and owner and repo:
        return {"host": host, "owner": owner, "repo": repo, "html_url": f"https://{host}/{owner}/{repo}"}
    return None

def parse_issue_numbers_from_text(text, max_refs=3):
    if not text:
        return []
    numbers = []
    patterns = [
        r'#(\d+)',
        r'GH-(\d+)',
        r'PR\s*#?(\d+)',
        r'pull\s+request\s+#?(\d+)',
        r'issue\s+#?(\d+)',
        r'fixes\s+#?(\d+)',
        r'closes\s+#?(\d+)',
    ]
    for pat in patterns:
        for m in re.findall(pat, text, flags=re.IGNORECASE):
            try:
                n = int(m)
                if n not in numbers:
                    numbers.append(n)
            except ValueError:
                pass
            if len(numbers) >= max_refs:
                break
        if len(numbers) >= max_refs:
            break
    return numbers

def _github_api_get(owner, repo, subpath, token):
    url = f"https://api.github.com/repos/{owner}/{repo}{subpath}"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "simple-extract-script"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8", errors="replace")
            return json.loads(data)
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
            print(f"   ⚠️ GitHub API error {e.code} for {subpath}: {err_body[:200]}")
        except Exception:
            print(f"   ⚠️ GitHub API error {e.code} for {subpath}")
        return None
    except Exception as e:
        print(f"   ⚠️ GitHub API request failed for {subpath}: {e}")
        return None

def fetch_issue_pr_bundle(owner, repo, numbers, token):
    bundles = []
    for n in numbers:
        issue = _github_api_get(owner, repo, f"/issues/{n}", token)
        if not issue:
            continue
        item = {
            "number": n,
            "type": "pr" if "pull_request" in issue else "issue",
            "title": issue.get("title") or "",
            "body": issue.get("body") or "",
            "html_url": issue.get("html_url") or f"https://github.com/{owner}/{repo}/issues/{n}",
            "comments": []
        }
        comments = _github_api_get(owner, repo, f"/issues/{n}/comments", token) or []
        for c in comments[:10]:
            author = c.get("user", {}).get("login", "unknown")
            body = c.get("body") or ""
            item["comments"].append(f"{author}: {body}")
        if item["type"] == "pr":
            pr_comments = _github_api_get(owner, repo, f"/pulls/{n}/comments", token) or []
            for c in pr_comments[:20]:
                author = c.get("user", {}).get("login", "unknown")
                path = c.get("path") or ""
                body = c.get("body") or ""
                line = c.get("line") or c.get("original_line") or ""
                item["comments"].append(f"{author} [{path}:{line}]: {body}")
            pr = _github_api_get(owner, repo, f"/pulls/{n}", token)
            if pr and pr.get("html_url"):
                item["html_url"] = pr["html_url"]
        bundles.append(item)
    return bundles

def compute_numstat(repo_path, commit_hash):
    out = run_cmd(["git", "diff", "--numstat", f"{commit_hash}^", commit_hash], cwd=repo_path, timeout=60)
    files = []
    total_added = 0
    total_removed = 0
    if out:
        for line in out.strip().splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    added = 0 if parts[0] == "-" else int(parts[0])
                    removed = 0 if parts[1] == "-" else int(parts[1])
                except ValueError:
                    added = 0
                    removed = 0
                filename = parts[2]
                files.append({"filename": filename, "added": added, "removed": removed})
                total_added += added
                total_removed += removed
    return files, total_added, total_removed

def get_full_commit_message(repo_path, commit_hash):
    msg = run_cmd(["git", "show", "-s", "--format=%B", commit_hash], cwd=repo_path, timeout=30)
    return msg or ""

def get_commit_patch(repo_path, commit_hash):
    return run_cmd(["git", "show", "--no-merges", "--pretty=format:", commit_hash], cwd=repo_path, timeout=60) or ""

def build_semantic_chunk(commit, message_full, files_stats, totals, patch_text, repo_info, issue_bundles):
    host_owner_repo = ""
    html_repo = ""
    if repo_info:
        host_owner_repo = f"{repo_info['owner']}/{repo_info['repo']}@{commit['hash']}"
        html_repo = repo_info.get("html_url", "")
    subject = message_full.strip().splitlines()[0] if message_full.strip() else (commit.get("message") or "")
    body_lines = message_full.strip().splitlines()[1:] if message_full.strip() else []
    body = "\n".join(body_lines).strip()
    lines = []
    lines.append(f"ROUTE: {host_owner_repo}")
    if html_repo:
        lines.append(f"REPO: {html_repo}")
    lines.append(f"COMMIT: {commit['hash']}")
    lines.append(f"AUTHOR: {commit.get('author_name','')} <{commit.get('author_email','')}>")
    lines.append(f"DATE: {commit.get('authored_date','')}")
    lines.append("")
    if issue_bundles:
        lines.append("=== ISSUE_AND_PR ===")
        for b in issue_bundles:
            header = f"[{b['type'].upper()} #{b['number']}] {b['title']}".strip()
            lines.append(header)
            if b.get("html_url"):
                lines.append(f"URL: {b['html_url']}")
            if b.get("body"):
                lines.append("BODY:")
                lines.append(b["body"].strip())
            if b.get("comments"):
                lines.append("COMMENTS:")
                for c in b["comments"]:
                    lines.append(f"- {c}")
            lines.append("")
    lines.append("=== COMMIT ===")
    lines.append(f"SUBJECT: {subject}")
    if body:
        lines.append("BODY:")
        lines.append(body)
    lines.append("")
    lines.append("=== FILES ===")
    lines.append(f"Summary: {len(files_stats)} files changed, +{totals[0]} -{totals[1]}")
    for fs in files_stats:
        lines.append(f"- {fs['filename']} (+{fs['added']} -{fs['removed']})")
    lines.append("")
    lines.append("=== PATCH ===")
    lines.append(patch_text.strip())
    lines.append("")
    lines.append("----- END CHUNK -----")
    return "\n".join(lines)

def emit_semantic_text_files(repo_path, commits, out_dir, limit=200, enable_github=True):
    print("🧾 Emitting semantic raw text files...")
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    repo_info = get_repo_remote_info(repo_path)
    token = os.environ.get("GITHUB_TOKEN") if (enable_github and repo_info and "github.com" in (repo_info.get("host") or "")) else None
    count = 0
    for commit in commits[:limit]:
        try:
            h = commit["hash"]
            message_full = get_full_commit_message(repo_path, h)
            patch_text = get_commit_patch(repo_path, h)
            if not patch_text or len(patch_text.strip()) < 10:
                continue
            files_stats, total_added, total_removed = compute_numstat(repo_path, h)
            issue_nums = parse_issue_numbers_from_text(message_full)
            issue_bundles = []
            if issue_nums and token and repo_info and repo_info.get("host") == "github.com":
                issue_bundles = fetch_issue_pr_bundle(repo_info["owner"], repo_info["repo"], issue_nums, token)
            chunk = build_semantic_chunk(commit, message_full, files_stats, (total_added, total_removed), patch_text, repo_info, issue_bundles)
            subdir = raw_dir / h[:2]
            subdir.mkdir(exist_ok=True)
            out_file = subdir / f"{h}.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(chunk)
            count += 1
            if count % 10 == 0:
                print(f"   Wrote {count} raw chunks...")
        except Exception as e:
            print(f"   ⚠️  Skipped raw chunk for {commit.get('hash','')[:8]}: {e}")
            continue
    print(f"✅ Emitted {count} raw text chunks")
    return count

def main():
    ap = argparse.ArgumentParser(description="Simple git extraction (no PyDriller)")
    ap.add_argument("--repo", required=True, help="Git repository path")
    ap.add_argument("--workdir", required=True, help="Working directory")
    ap.add_argument("--max-commits", type=int, default=500, help="Maximum commits to extract")
    ap.add_argument("--raw-limit", type=int, default=200, help="Maximum raw text chunks to emit")
    ap.add_argument("--no-github", action="store_true", help="Disable GitHub API calls for issues/PRs")
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
    
    # Emit semantic raw text files
    raw_count = emit_semantic_text_files(repo_path, commits, out_dir, limit=args.raw_limit, enable_github=not args.no_github)
    
    print("\n📊 Simple Extraction Complete:")
    print(f"   Commits: {len(commits)}")
    print(f"   Patches: {patch_count}")
    print(f"   Raw text chunks: {raw_count}")
    print(f"   Output: {out_dir}")

    # Show sample data
    print("\n📋 Sample Data:")
    if commits:
        print(f"   Latest commit: {commits[0]['hash'][:8]} - {commits[0]['message'][:50]}...")

    print("\n✅ Ready for upload! Use:")
    print(f"   uv run data/push_complete.py --workdir {workdir} --hf-namespace YOUR_NAMESPACE --hf-dataset YOUR_DATASET")

if __name__ == "__main__":
    main()