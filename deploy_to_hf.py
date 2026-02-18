"""
deploy_to_hf.py
Run once:  python deploy_to_hf.py
Requires:  pip install huggingface_hub
Prompts for a HuggingFace write-access token, then:
  1. Creates (or reuses) Space  singhrituraj114/agriyield-ai
  2. Clones the Space repo to a temp folder
  3. Copies all project files + the 4 .pkl models  
  4. Tracks *.pkl via git-lfs (large files)
  5. Commits + pushes everything
  6. Prints the live URL
"""

import os, sys, shutil, subprocess, tempfile
from pathlib import Path

# ── Config ─────────────────────────────────────────────────
HF_USER      = "RiturajDS"              # HuggingFace username
SPACE_NAME   = "agriyield-ai"
SPACE_SDK    = "docker"
REPO_ID      = f"{HF_USER}/{SPACE_NAME}"
SPACE_URL    = f"https://huggingface.co/spaces/{HF_USER}/{SPACE_NAME}"
CLONE_URL    = f"https://huggingface.co/spaces/{REPO_ID}"
PROJECT_ROOT = Path(r"C:\Users\singh\AgriYieldAI")

# Files to copy (relative to PROJECT_ROOT)
COPY_FILES = [
    "backend/main.py",
    "frontend/index.html",
    "crop_production.csv",
    "requirements.txt",
    "Dockerfile",
    "README.md",
]
COPY_DIRS = [
    "backend",
    "frontend",
]
MODEL_FILES = [
    "agri_yield_model.pkl",
    "state_encoder.pkl",
    "season_encoder.pkl",
    "crop_encoder.pkl",
]

def run(cmd, cwd=None, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=True, text=True)
    if result.stdout.strip(): print("   ", result.stdout.strip())
    if result.stderr.strip(): print("   ERR:", result.stderr.strip()[:200])
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result

def main():
    print("\n🌾  AgriYield AI — HuggingFace Spaces Deployer")
    print("=" * 52)

    # ── Get token ──────────────────────────────────────────
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("\nGet a WRITE token from: https://huggingface.co/settings/tokens/new?tokenType=write")
        token = input("Paste your HuggingFace write token: ").strip()
    if not token:
        sys.exit("❌ No token provided.")

    # ── Login via huggingface_hub ──────────────────────────
    try:
        from huggingface_hub import HfApi, login
        login(token=token, add_to_git_credential=True)
        api = HfApi(token=token)
    except Exception as e:
        sys.exit(f"❌ Login failed: {e}")

    # ── Create Space (skip if exists) ──────────────────────
    print(f"\n📦 Creating Space: {REPO_ID}")
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk=SPACE_SDK, exist_ok=True)
        print(f"   ✅ Space ready: {SPACE_URL}")
    except Exception as e:
        sys.exit(f"❌ Could not create Space: {e}")

    # ── Clone Space repo to temp dir ───────────────────────
    tmpdir = Path(tempfile.mkdtemp(prefix="hf_agri_"))
    clone_url_auth = f"https://user:{token}@huggingface.co/spaces/{REPO_ID}"
    print(f"\n📥 Cloning Space repo to {tmpdir}")
    try:
        run(["git", "clone", clone_url_auth, str(tmpdir)])
    except:
        # If clone fails (empty repo), init manually
        shutil.rmtree(tmpdir, ignore_errors=True)
        tmpdir.mkdir(parents=True, exist_ok=True)
        run(["git", "init"], cwd=tmpdir)
        run(["git", "remote", "add", "origin", clone_url_auth], cwd=tmpdir)

    # ── Set up git-lfs for large pkl files ────────────────
    print("\n🔧 Setting up git-lfs for .pkl model files")
    run(["git", "lfs", "install"], cwd=tmpdir)
    # Write .gitattributes
    attrs = tmpdir / ".gitattributes"
    attrs.write_text("*.pkl filter=lfs diff=lfs merge=lfs -text\n")

    # ── Copy source files ──────────────────────────────────
    print("\n📋 Copying project files")
    for d in COPY_DIRS:
        src = PROJECT_ROOT / d
        dst = tmpdir / d
        if src.exists():
            if dst.exists(): shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"   📁 {d}/")

    for f in ["requirements.txt", "Dockerfile", "README.md", "crop_production.csv"]:
        src = PROJECT_ROOT / f
        if src.exists():
            shutil.copy2(src, tmpdir / f)
            print(f"   📄 {f}")

    # ── Copy model files ───────────────────────────────────
    print("\n🤖 Copying model files (large — via git-lfs)")
    missing = []
    for mf in MODEL_FILES:
        src = PROJECT_ROOT / mf
        if src.exists():
            shutil.copy2(src, tmpdir / mf)
            size_mb = src.stat().st_size / 1024 / 1024
            print(f"   🧠 {mf}  ({size_mb:.0f} MB)")
        else:
            missing.append(mf)
            print(f"   ⚠️  MISSING: {mf}")

    if missing:
        print(f"\n⚠️  Missing model files: {missing}")
        print("   These must be present to run predictions.")
        ans = input("Continue anyway? (y/n): ").strip().lower()
        if ans != 'y': sys.exit("Aborted.")

    # ── Set git user ───────────────────────────────────────
    run(["git", "config", "user.email", "singhrituraj114@outlook.com"], cwd=tmpdir)
    run(["git", "config", "user.name", "Singhrituraj114"], cwd=tmpdir)

    # ── Commit & push ──────────────────────────────────────
    print("\n🚀 Committing and pushing to HuggingFace Spaces")
    run(["git", "add", "-A"], cwd=tmpdir)

    # Check if there's anything to commit
    status = run(["git", "status", "--porcelain"], cwd=tmpdir, check=False)
    if not status.stdout.strip():
        print("   Nothing to commit — already up to date.")
    else:
        run(["git", "commit", "-m", "deploy: AgriYield AI v2 — FastAPI + Three.js + RandomForest"], cwd=tmpdir)
        run(["git", "push", "origin", "main", "--force"], cwd=tmpdir)

    # ── Cleanup ────────────────────────────────────────────
    shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"""
╔══════════════════════════════════════════════════════╗
║  ✅  DEPLOYMENT COMPLETE                             ║
╠══════════════════════════════════════════════════════╣
║  🤖  Backend  (builds in ~5 min):                   ║
║      {SPACE_URL:<46} ║
║                                                      ║
║  🌐  Frontend (live now):                            ║
║      https://singhrituraj114.github.io/AgriYield-AI/ ║
╚══════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    main()
