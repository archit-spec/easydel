#!/usr/bin/env python3
import argparse, gzip, io, json, os, re, shutil, subprocess, sys, tarfile, tempfile, time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# deps: pip install pydriller gitpython huggingface_hub tqdm
import concurrent.futures
from tqdm import tqdm
from pydriller import Repository
from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder

def run(cmd, cwd=None, check=True):
    p = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{p.stderr}")
    return p.stdout

def gz_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return gzip.open(path, "wt", encoding="utf-8")

def shard_writer(base_dir: Path, base_name: str, shard_idx: int):
    return gz_writer(base_dir / f"{base_name}-{shard_idx:04d}.jsonl.gz")

def sha1_file(p: Path):
    import hashlib
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def process_commit(commit, args, repo_dir, out_dir):
    cobj = {
        "hash": commit.hash,
        "parents": [p for p in commit.parents],
        "author_name": commit.author.name if commit.author else None,
        "author_email": commit.author.email if commit.author else None,
        "authored_date": commit.author_date.isoformat() if commit.author_date else None,
        "committer_name": commit.committer.name if commit.committer else None,
        "committer_email": commit.committer.email if commit.committer else None,
        "committed_date": commit.committer_date.isoformat() if commit.committer_date else None,
        "message": commit.msg,
        "in_main_branch": commit.in_main_branch,
        "merge": commit.merge,
        "project": "juspay/hyperswitch",
    }
    mod_recs = []
    patch_data = []
    snap_data = []
    for m in commit.modified_files:
        rec = {
            "commit": commit.hash,
            "old_path": m.old_path,
            "new_path": m.new_path,
            "change_type": m.change_type,
            "added": m.added_lines,
            "removed": m.deleted_lines,
            "filename": m.filename,
        }
        mod_recs.append(rec)
        if args.extract_patches and m.diff is not None:
            if not (args.max_patch_lines and m.diff.count("\n") > args.max_patch_lines):
                patch_rec = {
                    "commit": commit.hash,
                    "path": m.new_path or m.old_path,
                    "patch": m.diff
                }
                patch_data.append(patch_rec)
        if args.extract_snapshots_changed and (m.new_path or m.old_path):
            path = m.new_path or m.old_path
            try:
                before = run(["git", "show", f"{commit.hash}^:{path}"], cwd=repo_dir, check=False).encode("utf-8", "ignore")
            except Exception:
                before = b""
            try:
                after = run(["git", "show", f"{commit.hash}:{path}"], cwd=repo_dir, check=False).encode("utf-8", "ignore")
            except Exception:
                after = b""
            snap_rec = {
                "commit": commit.hash,
                "path": path,
                "before": before,
                "after": after
            }
            snap_data.append(snap_rec)
    return cobj, mod_recs, patch_data, snap_data

def tar_directory(src_dir: Path, out_tar_gz: Path, exclude_globs=None):
    exclude_globs = exclude_globs or []
    out_tar_gz.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar_gz, "w:gz") as tar:
        for root, _, files in os.walk(src_dir):
            for fn in files:
                fp = Path(root) / fn
                rel = fp.relative_to(src_dir)
                skip = any(rel.match(g) for g in exclude_globs)
                if skip: 
                    continue
                tar.add(fp, arcname=str(rel))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Git URL or local path")
    ap.add_argument("--workdir", required=True, help="Working directory")
    ap.add_argument("--hf-namespace", required=True, help="HF username/org, e.g., archit11")
    ap.add_argument("--hf-dataset", required=True, help="HF dataset name, e.g., hyperswitch-code-history")
    ap.add_argument("--extract-patches", action="store_true", help="Export full unified diffs (can be large)")
    ap.add_argument("--extract-snapshots-changed", action="store_true", help="Store before/after contents of changed files")
    ap.add_argument("--max-patch-lines", type=int, default=2000, help="Skip patches longer than this many lines")
    ap.add_argument("--shard-size", type=int, default=200_000, help="Rows per .jsonl.gz shard")
    ap.add_argument("--repo_subdir", default="", help="If you only care about a path like 'crates/'")
    args = ap.parse_args()

    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        print("ERROR: set HF_TOKEN env var.", file=sys.stderr); sys.exit(1)

    workdir = Path(args.workdir).absolute()
    repo_dir = workdir / "repo"
    out_dir  = workdir / "dataset"
    shutil.rmtree(workdir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Clone full history
    print("Cloning repo...")
    run(["git", "clone", "--progress", "--mirror", args.repo, str(workdir / "repo_mirror")])
    # make a working tree to get files at HEAD
    run(["git", "clone", str(workdir / "repo_mirror"), str(repo_dir)])
    if args.repo_subdir:
        subpath = args.repo_subdir.rstrip("/") + "/"
    else:
        subpath = ""

    # 2) Export HEAD snapshot (codebase)
    print("Exporting HEAD codebase tar.gz and files index...")
    head_tar = out_dir / "HEAD_codebase.tar.gz"
    tar_directory(repo_dir, head_tar, exclude_globs=[".git/**", "**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.pdf", "**/*.mp4", "**/*.mov"])
    files_index_path = out_dir / "files_index.jsonl.gz"
    with gz_writer(files_index_path) as fw:
        for root, _, files in os.walk(repo_dir):
            if ".git" in root.split(os.sep): 
                continue
            for fn in files:
                p = Path(root) / fn
                rel = p.relative_to(repo_dir)
                size = p.stat().st_size
                try:
                    digest = sha1_file(p) if size <= 50_000_000 else None
                except Exception:
                    digest = None
                rec = {"path": str(rel), "size": size, "sha1": digest}
                fw.write(json.dumps(rec) + "\n")

    # Copy LICENSE if present
    lic_src = repo_dir / "LICENSE"
    if lic_src.exists():
        shutil.copy2(lic_src, out_dir / "LICENSE.txt")

    # 3) Traverse commits with PyDriller
    print("Traversing commits (PyDriller)...")
    commits_dir = out_dir / "commits"
    mods_dir     = out_dir / "modifications"
    patches_dir  = out_dir / "patches"      # optionally filled
    snaps_dir    = out_dir / "snapshots"    # optionally filled
    commits_dir.mkdir(parents=True, exist_ok=True)
    mods_dir.mkdir(parents=True, exist_ok=True)
    if args.extract_patches: patches_dir.mkdir(parents=True, exist_ok=True)
    if args.extract_snapshots_changed: snaps_dir.mkdir(parents=True, exist_ok=True)

    c_shard_idx = 0; c_rows = 0; c_fw = shard_writer(commits_dir, "commits", c_shard_idx)
    m_shard_idx = 0; m_rows = 0; m_fw = shard_writer(mods_dir, "modifications", m_shard_idx)

    repo = Repository(str(repo_dir), order='reverse')
    commits = list(repo.traverse_commits())
    print(f"Found {len(commits)} commits")

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_commit, commit, args, str(repo_dir), str(out_dir)) for commit in commits]
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(commits), desc="Processing commits"):
            results.append(future.result())

    for cobj, mod_recs, patch_data, snap_data in results:
        c_fw.write(json.dumps(cobj, ensure_ascii=False) + "\n")
        c_rows += 1
        if c_rows >= args.shard_size:
            c_fw.close(); c_shard_idx += 1; c_rows = 0
            c_fw = shard_writer(commits_dir, "commits", c_shard_idx)

        for rec in mod_recs:
            m_fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            m_rows += 1
            if m_rows >= args.shard_size:
                m_fw.close(); m_shard_idx += 1; m_rows = 0
                m_fw = shard_writer(mods_dir, "modifications", m_shard_idx)

        for patch_rec in patch_data:
            shard = (patches_dir / f"{patch_rec['commit'][:2]}").with_suffix("")
            shard.mkdir(parents=True, exist_ok=True)
            with gz_writer(shard / f"{patch_rec['commit']}_{(patch_rec['path'] or 'unknown').replace('/', '__')}.patch.jsonl.gz") as pw:
                pw.write(json.dumps(patch_rec) + "\n")

        for snap_rec in snap_data:
            base = snaps_dir / f"{snap_rec['commit'][:2]}" / snap_rec['commit']
            (base / "before").parent.mkdir(parents=True, exist_ok=True)
            for kind, blob in (("before", snap_rec['before']), ("after", snap_rec['after'])):
                if blob:
                    outp = base / kind / (snap_rec['path'].replace("/", "__") + ".gz")
                    outp.parent.mkdir(parents=True, exist_ok=True)
                    with gzip.open(outp, "wb") as fb:
                        fb.write(blob)

    c_fw.close(); m_fw.close()

    # 4) (Optional) Issues & PRs dump – left as stub for privacy; enable if you set GITHUB_TOKEN
    gh_token = os.environ.get("GITHUB_TOKEN")
    if gh_token:
        print("Fetching Issues/PRs via GitHub API...")
        import requests
        session = requests.Session(); session.headers["Authorization"] = f"token {gh_token}"
        def paged(url):
            page = 1
            while True:
                r = session.get(url, params={"state":"all", "per_page":100, "page":page}, timeout=30)
                if r.status_code != 200: break
                data = r.json()
                if not data: break
                for x in data: yield x
                page += 1
        issues = []
        for it in tqdm(paged("https://api.github.com/repos/juspay/hyperswitch/issues"), desc="Issues/PRs"):
            # GitHub returns both issues and PRs from /issues endpoint
            issues.append({
                "id": it.get("id"),
                "number": it.get("number"),
                "title": it.get("title"),
                "state": it.get("state"),
                "is_pull_request": "pull_request" in it,
                "created_at": it.get("created_at"),
                "closed_at": it.get("closed_at"),
                "user": it.get("user", {}).get("login"),
                "labels": [l.get("name") for l in it.get("labels", [])],
                "body": it.get("body"),
            })
        with gz_writer(out_dir / "issues_prs.jsonl.gz") as fw:
            for x in issues:
                fw.write(json.dumps(x) + "\n")

    # 5) Push to Hugging Face
    repo_id = f"{args.hf_namespace}/{args.hf_dataset}"
    print(f"Pushing to HF dataset: {repo_id}")
    api = HfApi()
    try:
        create_repo(repo_id=repo_id, token=HF_TOKEN, repo_type="dataset", exist_ok=True)
    except Exception:
        pass
    upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(out_dir),
        token=HF_TOKEN,
        commit_message=f"Add hyperswitch extracted dataset at {datetime.utcnow().isoformat()}Z"
    )
    print("Done ✅")

if __name__ == "__main__":
    main()
