#!/usr/bin/env python3
import subprocess, os, sys
from pathlib import Path

def run_cmd(cmd, cwd=None, check=True):
    """Run command and return output"""
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30)
        if check and result.returncode != 0:
            print(f"❌ Command failed: {' '.join(cmd)}")
            print(f"STDERR: {result.stderr}")
            return None
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"⏰ Command timed out: {' '.join(cmd)}")
        return None
    except Exception as e:
        print(f"💥 Error running command: {e}")
        return None

def diagnose_repository(repo_path):
    """Diagnose repository issues"""
    print(f"🔍 Diagnosing repository: {repo_path}")

    if not repo_path.exists():
        print(f"❌ Repository path does not exist: {repo_path}")
        return False

    # Check if it's a git repository
    git_dir = repo_path / ".git"
    if not git_dir.exists():
        print(f"❌ Not a git repository (no .git directory): {repo_path}")
        return False

    print("✅ Git repository found")

    # Check git status
    print("\n📋 Git Status:")
    status = run_cmd(["git", "status", "--porcelain"], cwd=repo_path)
    if status is not None:
        print(f"   Status: {'Clean' if not status else f'{len(status.split(chr(10)))} changes'}")

    # Check current branch
    branch = run_cmd(["git", "branch", "--show-current"], cwd=repo_path)
    if branch:
        print(f"   Branch: {branch}")

    # Check number of commits
    commit_count = run_cmd(["git", "rev-list", "--count", "HEAD"], cwd=repo_path)
    if commit_count:
        print(f"   Total commits: {commit_count}")

    # Check recent commits
    print("\n📝 Recent commits:")
    recent_commits = run_cmd(["git", "log", "--oneline", "-10"], cwd=repo_path)
    if recent_commits:
        for line in recent_commits.split('\n')[:5]:
            if line.strip():
                print(f"   {line}")

    # Test PyDriller import
    print("\n🐍 Testing PyDriller:")
    try:
        from pydriller import Repository
        print("   ✅ PyDriller imported successfully")

        # Try to create repository object
        try:
            repo = Repository(str(repo_path))
            commits = list(repo.traverse_commits())
            print(f"   ✅ Repository object created")
            print(f"   ✅ Found {len(commits)} commits via PyDriller")

            if len(commits) > 0:
                print(f"   📋 Sample commit: {commits[0].hash[:8]} - {commits[0].msg[:50]}...")

        except Exception as e:
            print(f"   ❌ PyDriller repository creation failed: {e}")
            return False

    except ImportError:
        print("   ❌ PyDriller not installed")
        return False

    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python diagnose_repo.py <repo_path>")
        print("Example: python diagnose_repo.py data/_work_hyperswitch/repo")
        sys.exit(1)

    repo_path = Path(sys.argv[1]).absolute()

    if diagnose_repository(repo_path):
        print("\n🎉 Repository diagnosis completed successfully!")
        print("The repository appears to be in good condition.")
    else:
        print("\n❌ Repository diagnosis found issues.")
        print("Check the error messages above.")

if __name__ == "__main__":
    main()