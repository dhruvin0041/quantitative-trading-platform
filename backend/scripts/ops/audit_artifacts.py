import hashlib
import subprocess
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

def get_file_hash(path: Path) -> str:
    if not path.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_git_commit(path: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "log", "-n", "1", "--pretty=format:%h %cd (%s)", "--", str(path)],
            cwd=str(BACKEND_DIR.parent),
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip() if res.stdout else "Untracked / Local"
    except Exception as e:
        return f"Error: {e}"

artifacts = [
    "artifacts/xgb_ensemble.json",
    "artifacts/latest_fusion_weights.weights.h5",
    "artifacts/lgbm_agent.joblib",
    "artifacts/dqn_model.pth",
    "artifacts/latest_scaler.joblib",
    "artifacts/model_calibrator.joblib",
    "artifacts/meta_ensemble.joblib",
    "artifacts/tft_quantile_weights.weights.h5",
    "artifacts/regime_detector.joblib",
    "artifacts/X_train_tabular.joblib",
    "artifacts/X_val_tabular.joblib",
    "artifacts/y_train_sig.joblib",
    "artifacts/y_val_sig.joblib",
    "configs/model_params.yaml",
    "configs/optimized_params_AAPL.json",
    "configs/kept_features.json",
]

print("=" * 120)
print(f"{'ARTIFACT':<35} | {'SIZE (BYTES)':<12} | {'SHA256 (FIRST 16)':<18} | {'GIT COMMIT'}")
print("=" * 120)

for art in artifacts:
    p = BACKEND_DIR / art
    size = p.stat().st_size if p.exists() else 0
    f_hash = get_file_hash(p)
    git_c = get_git_commit(p)
    print(f"{art:<35} | {size:<12} | {f_hash[:16]:<18} | {git_c[:45]}")
