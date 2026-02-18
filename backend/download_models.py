"""
download_models.py
Downloads model .pkl files from HuggingFace model repo at container startup.
Only downloads if files are not already present (caches across restarts).
"""
import os, sys
from pathlib import Path

REPO_ID   = "RiturajDS/agriyield-models"
MODEL_DIR = Path(__file__).parent.parent  # /app
PKL_FILES = [
    "agri_yield_model.pkl",
    "state_encoder.pkl",
    "season_encoder.pkl",
    "crop_encoder.pkl",
]

def main():
    all_present = all((MODEL_DIR / f).exists() for f in PKL_FILES)
    if all_present:
        print("✅ Model files already present — skipping download.")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub…")
        os.system(f"{sys.executable} -m pip install huggingface_hub -q")
        from huggingface_hub import hf_hub_download

    for fname in PKL_FILES:
        dest = MODEL_DIR / fname
        if dest.exists():
            print(f"  ✓ {fname} already exists")
            continue
        print(f"  ⬇ Downloading {fname} from {REPO_ID} …")
        local = hf_hub_download(
            repo_id=REPO_ID,
            filename=fname,
            repo_type="model",
            local_dir=str(MODEL_DIR),
        )
        print(f"  ✅ {fname} → {local}")

    print("🎉 All model files ready.")

if __name__ == "__main__":
    main()
