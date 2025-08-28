#!/usr/bin/env python3
import argparse, gzip, io, json, os, re, shutil, subprocess, sys, tarfile, tempfile, time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# deps: pip install pydriller gitpython huggingface_hub tqdm pygit2 lz4 orjson
import concurrent.futures
from tqdm import tqdm
from pydriller import Repository
from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder

# Optional fast libraries
try:
    import pygit2
    HAS_PYGIT2 = True
except ImportError:
    HAS_PYGIT2 = False
    print("Note: Install pygit2 for faster extraction: pip install pygit2")

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

def run(cmd, cwd=None, check=True, timeout=None):
    p = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
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

def extract_repo_fast_pygit2(repo_path: Path, out_dir: Path, args):
    """Fast extraction using pygit2 - 10-100x faster than subprocess approach"""
    if not HAS_PYGIT2:
        raise ImportError("pygit2 not installed. Install with: pip install pygit2")

    print("🚀 Using fast pygit2 extraction...")
    repo = pygit2.Repository(str(repo_path))

    # Get all commits efficiently
    commits = []
    walker = repo.walk(repo.head.target, pygit2.GIT_SORT_TIME | pygit2.GIT_SORT_REVERSE)

    for commit in walker:
        commits.append(commit)

    print(f"Found {len(commits)} commits")

    # Process in parallel using ProcessPoolExecutor for CPU-bound operations
    max_workers = args.max_workers or min(32, os.cpu_count())
    batch_size = 500  # Larger batches for efficiency

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, len(commits), batch_size):
            batch = commits[i:i+batch_size]
            # Convert commit objects to OIDs for pickling
            batch_oids = [str(commit.id) for commit in batch]
            future = executor.submit(process_commit_batch_pygit2, str(repo_path), batch_oids, args)
            futures.append(future)

        # Collect results with progress
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing batches"):
            batch_results = future.result()
            results.extend(batch_results)

    # Write results using fast compression
    write_results_fast(results, out_dir, args)
    return results

def process_commit_batch_pygit2(repo_path, commit_oids, args):
    """Process a batch of commits using pygit2 - no subprocess calls!"""
    repo = pygit2.Repository(repo_path)
    results = []

    for oid in commit_oids:
        try:
            commit = repo[oid]
            commit_data = {
                "hash": str(commit.id),
                "parents": [str(p) for p in commit.parents],
                "author_name": commit.author.name,
                "author_email": commit.author.email,
                "authored_date": commit.author.when.isoformat() if hasattr(commit.author, 'when') and commit.author.when else None,
                "committer_name": commit.committer.name,
                "committer_email": commit.committer.email,
                "committed_date": commit.committer.when.isoformat() if hasattr(commit.committer, 'when') and commit.committer.when else None,
                "message": commit.message,
                "in_main_branch": True,  # Assume main branch for simplicity
                "merge": len(commit.parents) > 1,
                "project": "juspay/hyperswitch",
            }

            mod_recs = []
            patch_data = []
            snap_data = []

            # Skip large commits
            if len(commit.parents) == 0:
                # Initial commit - no diff
                continue

            parent = repo[commit.parents[0]]
            diff = parent.tree.diff_to_tree(commit.tree)

            for patch in diff:
                delta = patch.delta

                # Skip if too many modifications requested
                if len(list(diff)) > args.max_modifications:
                    break

                # Modification record
                mod_rec = {
                    "commit": str(commit.id),
                    "old_path": delta.old_file.path if delta.old_file.path else None,
                    "new_path": delta.new_file.path if delta.new_file.path else None,
                    "change_type": str(delta.status),
                    "added": patch.additions,
                    "removed": patch.deletions,
                    "filename": delta.new_file.path or delta.old_file.path,
                }
                mod_recs.append(mod_rec)

                # Patch data (actual diff)
                if not args.no_extract_patches:
                    try:
                        patch_text = patch.text
                        if patch_text and len(patch_text) < args.max_patch_lines * 100:  # Rough line count
                            patch_rec = {
                                "commit": str(commit.id),
                                "path": delta.new_file.path or delta.old_file.path,
                                "patch": patch_text
                            }
                            patch_data.append(patch_rec)
                    except:
                        pass

                # Snapshot data (file content)
                if not args.no_extract_snapshots_changed:
                    try:
                        before_content = b""
                        after_content = b""

                        if delta.old_file.oid != pygit2.GIT_OID_HEX_ZERO:
                            old_blob = repo[delta.old_file.oid]
                            before_content = old_blob.data

                        if delta.new_file.oid != pygit2.GIT_OID_HEX_ZERO:
                            new_blob = repo[delta.new_file.oid]
                            after_content = new_blob.data

                        snap_rec = {
                            "commit": str(commit.id),
                            "path": delta.new_file.path or delta.old_file.path,
                            "before": before_content,
                            "after": after_content
                        }
                        snap_data.append(snap_rec)
                    except:
                        pass

            results.append((commit_data, mod_recs, patch_data, snap_data))

        except Exception as e:
            print(f"Error processing commit {oid}: {e}")
            continue

    return results

def write_results_fast(results, out_dir, args):
    """Write results using fast compression and serialization"""
    commits_dir = out_dir / "commits"
    mods_dir = out_dir / "modifications"
    patches_dir = out_dir / "patches" if not args.no_extract_patches else None
    snaps_dir = out_dir / "snapshots" if not args.no_extract_snapshots_changed else None

    commits_dir.mkdir(parents=True, exist_ok=True)
    mods_dir.mkdir(parents=True, exist_ok=True)
    if patches_dir: patches_dir.mkdir(parents=True, exist_ok=True)
    if snaps_dir: snaps_dir.mkdir(parents=True, exist_ok=True)

    # Use fast JSON if available
    json_dumps = orjson.dumps if HAS_ORJSON else json.dumps
    json_kwargs = {} if HAS_ORJSON else {"ensure_ascii": False}

    # Write commits
    with gz_writer(commits_dir / "commits-0000.jsonl.gz") as f:
        for commit_data, _, _, _ in results:
            f.write(json_dumps(commit_data, **json_kwargs) + b"\n")

    # Write modifications
    with gz_writer(mods_dir / "modifications-0000.jsonl.gz") as f:
        for _, mod_recs, _, _ in results:
            for rec in mod_recs:
                f.write(json_dumps(rec, **json_kwargs) + b"\n")

    # Write patches
    if patches_dir:
        patch_shards = {}
        for _, _, patch_data, _ in results:
            for patch_rec in patch_data:
                commit_prefix = patch_rec["commit"][:2]
                if commit_prefix not in patch_shards:
                    patch_shards[commit_prefix] = []

                shard_dir = patches_dir / commit_prefix
                shard_dir.mkdir(exist_ok=True)

                patch_file = shard_dir / f"{patch_rec['commit']}_{(patch_rec['path'] or 'unknown').replace('/', '__')}.patch.jsonl.gz"
                with gz_writer(patch_file) as f:
                    f.write(json_dumps(patch_rec, **json_kwargs) + b"\n")

    # Write snapshots
    if snaps_dir:
        for _, _, _, snap_data in results:
            for snap_rec in snap_data:
                base = snaps_dir / snap_rec["commit"][:2] / snap_rec["commit"]
                base.mkdir(parents=True, exist_ok=True)

                for kind, content in [("before", snap_rec["before"]), ("after", snap_rec["after"])]:
                    if content:
                        outp = base / kind / (snap_rec["path"].replace("/", "__") + ".gz")
                        outp.parent.mkdir(parents=True, exist_ok=True)
                        with gzip.open(outp, "wb") as f:
                            f.write(content)

def process_commit(commit, args, repo_dir, out_dir):
    try:
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

        # Skip commits with too many modifications (they're usually bulk operations)
        if len(commit.modified_files) > args.max_modifications:
            print(f"Skipping large commit {commit.hash[:8]} with {len(commit.modified_files)} modifications (max: {args.max_modifications})")
            return None, [], [], []

        # Log progress for large commits
        if len(commit.modified_files) > 50:
            print(f"Processing large commit {commit.hash[:8]} with {len(commit.modified_files)} modifications")

        for m in commit.modified_files:
            try:
                rec = {
                    "commit": commit.hash,
                    "old_path": m.old_path,
                    "new_path": m.new_path,
                    "change_type": str(m.change_type),
                    "added": m.added_lines,
                    "removed": m.deleted_lines,
                    "filename": m.filename,
                }
                mod_recs.append(rec)

                if not args.no_extract_patches:
                    try:
                        # Try to get diff, but don't fail if it doesn't work
                        if hasattr(m, 'diff') and m.diff is not None:
                            if not (args.max_patch_lines and m.diff.count("\n") > args.max_patch_lines):
                                patch_rec = {
                                    "commit": commit.hash,
                                    "path": m.new_path or m.old_path,
                                    "patch": m.diff
                                }
                                patch_data.append(patch_rec)
                    except Exception:
                        # Skip patches that can't be extracted
                        pass

                if not args.no_extract_snapshots_changed and (m.new_path or m.old_path):
                    path = m.new_path or m.old_path
                    try:
                        before = run(["git", "show", f"{commit.hash}^:{path}"], cwd=repo_dir, check=False, timeout=30).encode("utf-8", "ignore")
                    except Exception:
                        before = b""
                    try:
                        after = run(["git", "show", f"{commit.hash}:{path}"], cwd=repo_dir, check=False, timeout=30).encode("utf-8", "ignore")
                    except Exception:
                        after = b""
                    snap_rec = {
                        "commit": commit.hash,
                        "path": path,
                        "before": before,
                        "after": after
                    }
                    snap_data.append(snap_rec)
            except Exception as e:
                # Skip individual file modifications that fail
                print(f"Skipping modification in commit {commit.hash[:8]}: {e}")
                continue
        return cobj, mod_recs, patch_data, snap_data
    except Exception as e:
        print(f"Error processing commit {commit.hash}: {e}")
        # Return empty data for failed commits
        return None, [], [], []

def create_pretraining_dataset(out_dir, results, repo_dir):
    """Create a dataset optimized for language model pretraining"""
    pretraining_dir = out_dir / "pretraining"
    pretraining_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    shard_size = 1000  # commits per shard
    current_shard = []

    def write_shard(shard_data, idx):
        with gz_writer(pretraining_dir / f"pretraining-{idx:04d}.jsonl.gz") as fw:
            for item in shard_data:
                fw.write(json.dumps(item, ensure_ascii=False) + "\n")

    for cobj, mod_recs, patch_data, snap_data in results:
        # Combine commit info with patches and code content
        pretraining_item = {
            "commit_hash": cobj["hash"],
            "commit_message": cobj["message"],
            "author": cobj["author_name"],
            "date": cobj["authored_date"],
            "changes": []
        }

        # Add each file change with diff and content
        for patch_rec in patch_data:
            change = {
                "file_path": patch_rec["path"],
                "diff": patch_rec["patch"]
            }
            pretraining_item["changes"].append(change)

        # Add snapshot content for changed files
        for snap_rec in snap_data:
            # Find the corresponding change
            for change in pretraining_item["changes"]:
                if change["file_path"] == snap_rec["path"]:
                    try:
                        change["before_content"] = snap_rec["before"].decode("utf-8", errors="ignore") if snap_rec["before"] else ""
                        change["after_content"] = snap_rec["after"].decode("utf-8", errors="ignore") if snap_rec["after"] else ""
                    except:
                        change["before_content"] = ""
                        change["after_content"] = ""
                    break

        if pretraining_item["changes"]:  # Only include commits with changes
            current_shard.append(pretraining_item)

        if len(current_shard) >= shard_size:
            write_shard(current_shard, shard_idx)
            current_shard = []
            shard_idx += 1

    # Write remaining items
    if current_shard:
        write_shard(current_shard, shard_idx)

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
    ap.add_argument("--no-extract-patches", action="store_true", help="Skip exporting full unified diffs")
    ap.add_argument("--no-extract-snapshots-changed", action="store_true", help="Skip storing before/after contents of changed files")
    ap.add_argument("--max-patch-lines", type=int, default=2000, help="Skip patches longer than this many lines")
    ap.add_argument("--shard-size", type=int, default=200_000, help="Rows per .jsonl.gz shard")
    ap.add_argument("--repo_subdir", default="", help="If you only care about a path like 'crates/'")
    ap.add_argument("--create-pretraining-dataset", action="store_true", help="Create a combined dataset optimized for language model pretraining")
    ap.add_argument("--max-workers", type=int, default=None, help="Maximum number of worker threads (default: min(64, cpu_count))")
    ap.add_argument("--commit-timeout", type=int, default=120, help="Timeout in seconds for processing a single commit (default: 120)")
    ap.add_argument("--final-commit-timeout", type=int, default=60, help="Timeout in seconds for final commits (default: 60)")
    ap.add_argument("--max-modifications", type=int, default=100, help="Skip commits with more than this many modifications (default: 100)")
    ap.add_argument("--use-fast-extraction", action="store_true", help="Use pygit2 for much faster extraction (requires: pip install pygit2)")
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

    # Add safe directory to prevent ownership errors
    print("Configuring git safe directory...")
    run(["git", "config", "--global", "--add", "safe.directory", str(repo_dir)], check=False)
    # Also add it for the current user
    run(["git", "config", "--add", "safe.directory", str(repo_dir)], check=False, cwd=repo_dir)
    # Try to set ownership if possible
    try:
        run(["git", "config", "core.repositoryformatversion", "0"], check=False, cwd=repo_dir)
    except:
        pass
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

    # 3) Extract commits using fast or regular method
    if args.use_fast_extraction:
        if not HAS_PYGIT2:
            print("❌ pygit2 not installed. Install with: pip install pygit2")
            print("Falling back to regular extraction...")
            args.use_fast_extraction = False
        else:
            results = extract_repo_fast_pygit2(repo_dir, out_dir, args)
            # Skip regular extraction path
            commits = []  # Define empty to avoid UnboundLocalError
    else:
        print("Traversing commits (PyDriller)...")
        commits_dir = out_dir / "commits"
        mods_dir     = out_dir / "modifications"
        patches_dir  = out_dir / "patches"      # optionally filled
        snaps_dir    = out_dir / "snapshots"    # optionally filled
        commits_dir.mkdir(parents=True, exist_ok=True)
        mods_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_extract_patches: patches_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_extract_snapshots_changed: snaps_dir.mkdir(parents=True, exist_ok=True)

        c_shard_idx = 0; c_rows = 0; c_fw = shard_writer(commits_dir, "commits", c_shard_idx)
        m_shard_idx = 0; m_rows = 0; m_fw = shard_writer(mods_dir, "modifications", m_shard_idx)

        repo = Repository(str(repo_dir), order='reverse')
        commits = list(repo.traverse_commits())
        print(f"Found {len(commits)} commits")

    # Use fewer threads for I/O bound git operations - too many can cause contention
    if args.max_workers:
        max_workers = args.max_workers
    else:
        max_workers = min(64, os.cpu_count())
    print(f"Using {max_workers} threads for parallel processing")

    # Skip regular processing if fast extraction was used
    if args.use_fast_extraction:
        pass  # Results already obtained from fast extraction
    else:
        results = []
        batch_size = 50  # Smaller batches for more frequent saves
        save_interval = 10  # Save every 10 batches

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            def process_commit_with_timeout(commit):
                try:
                    print(f"Processing commit {commit.hash[:8]} ({len(commit.modified_files)} files)")
                    return process_commit(commit, args, str(repo_dir), str(out_dir))
                except Exception as e:
                    print(f"Commit {commit.hash[:8]} failed: {e}")
                    return None, [], [], []

                # Process commits in batches to save incrementally
                for i in range(0, len(commits), batch_size):
                    batch_commits = commits[i:i+batch_size]
                    print(f"Processing batch {i//batch_size + 1}/{(len(commits) + batch_size - 1)//batch_size} ({len(batch_commits)} commits)")

                    futures = [executor.submit(process_commit_with_timeout, commit) for commit in batch_commits]
                    completed_futures = set()
                    stuck_message_count = 0

                    # Process futures as they complete, with progress tracking
                    pbar = tqdm(total=len(batch_commits), desc=f"Batch {i//batch_size + 1}")
                    try:
                        while len(completed_futures) < len(futures):
                            # Use shorter timeout for final commits
                            remaining = len(futures) - len(completed_futures)
                            is_final_phase = remaining <= max(5, len(batch_commits) // 20)  # Last 5% of batch
                            is_very_final = remaining <= 3  # Last 3 commits in entire repo

                            if is_very_final:
                                current_timeout = 10  # Very aggressive timeout for final commits
                                print(f"🚨 VERY FINAL: {remaining} commits remaining, using {current_timeout}s timeout")
                            elif is_final_phase:
                                current_timeout = args.final_commit_timeout
                                if remaining < len(futures) - len(completed_futures):
                                    print(f"🔥 Final phase: {remaining} commits remaining in batch, using {current_timeout}s timeout")
                            else:
                                current_timeout = args.commit_timeout

                            # Wait for at least one future to complete
                            done, _ = concurrent.futures.wait(
                                [f for f in futures if f not in completed_futures],
                                timeout=current_timeout,
                                return_when=concurrent.futures.FIRST_COMPLETED
                            )

                            for future in done:
                                completed_futures.add(future)
                                try:
                                    result = future.result(timeout=1)  # Short timeout since it should already be done
                                    if result[0] is not None:  # Only add successful results
                                        results.append(result)
                                        # Save incrementally - write this result immediately
                                        cobj, mod_recs, patch_data, snap_data = result
                                        try:
                                            c_fw.write(json.dumps(cobj, ensure_ascii=False) + "\n")
                                            c_rows += 1
                                            if c_rows >= args.shard_size:
                                                c_fw.close(); c_shard_idx += 1; c_rows = 0
                                                c_fw = shard_writer(commits_dir, "commits", c_shard_idx)
                                        except Exception as e:
                                            print(f"Error writing commit {cobj.get('hash', 'unknown')[:8]}: {e}")

                                        for rec in mod_recs:
                                            try:
                                                m_fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                                m_rows += 1
                                                if m_rows >= args.shard_size:
                                                    m_fw.close(); m_shard_idx += 1; m_rows = 0
                                                    m_fw = shard_writer(mods_dir, "modifications", m_shard_idx)
                                            except Exception as e:
                                                print(f"Error writing modification for commit {rec.get('commit', 'unknown')[:8]}: {e}")

                                        for patch_rec in patch_data:
                                            try:
                                                shard = (patches_dir / f"{patch_rec['commit'][:2]}").with_suffix("")
                                                shard.mkdir(parents=True, exist_ok=True)
                                                with gz_writer(shard / f"{patch_rec['commit']}_{(patch_rec['path'] or 'unknown').replace('/', '__')}.patch.jsonl.gz") as pw:
                                                    pw.write(json.dumps(patch_rec) + "\n")
                                            except Exception as e:
                                                print(f"Error writing patch for commit {patch_rec.get('commit', 'unknown')[:8]}: {e}")

                                        for snap_rec in snap_data:
                                            try:
                                                base = snaps_dir / f"{snap_rec['commit'][:2]}" / snap_rec['commit']
                                                (base / "before").parent.mkdir(parents=True, exist_ok=True)
                                                for kind, blob in (("before", snap_rec['before']), ("after", snap_rec['after'])):
                                                    if blob:
                                                        outp = base / kind / (snap_rec['path'].replace("/", "__") + ".gz")
                                                        outp.parent.mkdir(parents=True, exist_ok=True)
                                                        with gzip.open(outp, "wb") as fb:
                                                            fb.write(blob)
                                            except Exception as e:
                                                print(f"Error writing snapshot for commit {snap_rec.get('commit', 'unknown')[:8]}: {e}")
                                except concurrent.futures.TimeoutError:
                                    if is_very_final:
                                        print(f"🚨 Final commit timed out - skipping to complete extraction")
                                        pbar.update(1)  # Mark as completed even if failed
                                    else:
                                        print(f"Commit processing timed out after {current_timeout}s")
                                except Exception as e:
                                    if is_very_final:
                                        print(f"🚨 Final commit failed - skipping to complete extraction")
                                        pbar.update(1)  # Mark as completed even if failed
                                    else:
                                        print(f"Error getting result: {e}")
                                pbar.update(1)

                        # Check for stuck futures and cancel them
                        stuck_futures = [f for f in futures if f not in completed_futures and f.running()]
                        if stuck_futures and stuck_message_count < 3:  # Limit spam
                            print(f"Waiting for {len(stuck_futures)} slow commits (timeout: {args.commit_timeout}s)...")
                            stuck_message_count += 1
                        elif stuck_futures and stuck_message_count == 3:
                            print(f"Still processing {len(stuck_futures)} slow commits...")
                            stuck_message_count += 1

                    except KeyboardInterrupt:
                        print("\n⚠️  KeyboardInterrupt detected! Saving current progress...")
                        # Cancel remaining futures
                        for future in futures:
                            if not future.done():
                                future.cancel()
                        print(f"Progress saved: {len(results)} commits processed so far")
                        break

                    pbar.close()
                    print(f"Batch {i//batch_size + 1} completed. Total processed: {len(results)} commits")

                    # Incremental save and push every few batches
                    batch_num = i//batch_size + 1
                    if batch_num % save_interval == 0 and len(results) > 0:
                        print(f"💾 Saving progress... ({len(results)} commits so far)")
                        c_fw.flush()
                        m_fw.flush()

                        # Create incremental backup
                        backup_dir = out_dir.parent / f"backup_batch_{batch_num}"
                        if backup_dir.exists():
                            shutil.rmtree(backup_dir)
                        shutil.copytree(out_dir, backup_dir)

                        print(f"✅ Progress saved to backup_batch_{batch_num}")

                c_fw.close(); m_fw.close()

    print(f"Successfully processed {len(results)} out of {len(commits)} commits")
    failed_commits = len(commits) - len(results)
    if failed_commits > 0:
        print(f"Skipped {failed_commits} problematic commits")

    # Summary of what was extracted
    print("\n📊 Extraction Summary:")
    print(f"   Commits: {len(results)}")
    print(f"   Patches: {'Yes' if not args.no_extract_patches else 'No'}")
    print(f"   Snapshots: {'Yes' if not args.no_extract_snapshots_changed else 'No'}")
    print(f"   Pretraining dataset: {'Yes' if args.create_pretraining_dataset else 'No'}")
    print(f"   Issues/PRs: {'Yes' if os.environ.get('GITHUB_TOKEN') else 'No (set GITHUB_TOKEN)'}")

    # 4) Create pretraining dataset if requested
    if args.create_pretraining_dataset:
        print("Creating pretraining dataset...")
        create_pretraining_dataset(out_dir, results, repo_dir)

    # 6) (Optional) Issues & PRs dump – left as stub for privacy; enable if you set GITHUB_TOKEN
    gh_token = os.environ.get("GITHUB_TOKEN")
    if gh_token:
        try:
            print("Fetching Issues/PRs via GitHub API...")
            import requests
            session = requests.Session()
            session.headers["Authorization"] = f"token {gh_token}"
            session.headers["Accept"] = "application/vnd.github.v3+json"

            def paged(url, desc="Fetching"):
                page = 1
                while True:
                    try:
                        r = session.get(url, params={"state":"all", "per_page":100, "page":page}, timeout=30)
                        if r.status_code != 200:
                            print(f"GitHub API error: {r.status_code} - {r.text}")
                            break
                        data = r.json()
                        if not data: break
                        for x in data: yield x
                        page += 1
                        if page > 50:  # Safety limit
                            print("Reached page limit (50), stopping...")
                            break
                    except Exception as e:
                        print(f"Error fetching page {page}: {e}")
                        break

            issues = []
            print("Fetching issues and pull requests...")
            for it in tqdm(paged("https://api.github.com/repos/juspay/hyperswitch/issues", "Issues/PRs"),
                          desc="Fetching Issues/PRs"):
                try:
                    # GitHub returns both issues and PRs from /issues endpoint
                    issue_data = {
                        "id": it.get("id"),
                        "number": it.get("number"),
                        "title": it.get("title"),
                        "state": it.get("state"),
                        "is_pull_request": "pull_request" in it,
                        "created_at": it.get("created_at"),
                        "updated_at": it.get("updated_at"),
                        "closed_at": it.get("closed_at"),
                        "user": it.get("user", {}).get("login"),
                        "assignee": it.get("assignee", {}).get("login") if it.get("assignee") else None,
                        "labels": [l.get("name") for l in it.get("labels", [])],
                        "body": it.get("body"),
                        "comments_count": it.get("comments", 0),
                        "reactions": it.get("reactions", {}),
                    }

                    # If it's a PR, add PR-specific data
                    if issue_data["is_pull_request"]:
                        pr_data = it.get("pull_request", {})
                        if pr_data:
                            # Fetch additional PR details
                            try:
                                pr_url = pr_data.get("url")
                                if pr_url:
                                    pr_response = session.get(pr_url, timeout=30)
                                    if pr_response.status_code == 200:
                                        pr_details = pr_response.json()
                                        issue_data["pr_details"] = {
                                            "merged": pr_details.get("merged", False),
                                            "mergeable": pr_details.get("mergeable"),
                                            "merged_at": pr_details.get("merged_at"),
                                            "base": pr_details.get("base", {}).get("ref"),
                                            "head": pr_details.get("head", {}).get("ref"),
                                            "commits": pr_details.get("commits", 0),
                                            "additions": pr_details.get("additions", 0),
                                            "deletions": pr_details.get("deletions", 0),
                                            "changed_files": pr_details.get("changed_files", 0),
                                        }
                            except Exception as e:
                                print(f"Error fetching PR details for #{issue_data['number']}: {e}")

                    issues.append(issue_data)
                except Exception as e:
                    print(f"Error processing issue/PR: {e}")
                    continue

            print(f"Fetched {len(issues)} issues/PRs")
            with gz_writer(out_dir / "issues_prs.jsonl.gz") as fw:
                for x in issues:
                    fw.write(json.dumps(x, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Failed to fetch issues/PRs: {e}")
            print("Continuing without issues/PRs data...")
    else:
        print("Note: Set GITHUB_TOKEN environment variable to include issues and PRs data")

    # 7) Push to Hugging Face
    if len(results) > 0:
        push_to_hf(out_dir, args.hf_namespace, args.hf_dataset, HF_TOKEN)
    else:
        print("❌ No commits were successfully processed. Skipping upload to Hugging Face.")
        print("Check the error messages above for details on what went wrong.")

def push_to_hf(out_dir, hf_namespace, hf_dataset, hf_token):
    repo_id = f"{hf_namespace}/{hf_dataset}"
    print(f"Pushing to HF dataset: {repo_id}")
    api = HfApi()
    try:
        create_repo(repo_id=repo_id, token=hf_token, repo_type="dataset", exist_ok=True)
    except Exception:
        pass
    upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(out_dir),
        token=hf_token,
        commit_message=f"Add hyperswitch extracted dataset at {datetime.utcnow().isoformat()}Z"
    )
    print("Done ✅")

if __name__ == "__main__":
    main()
