# src/execution/strategy_governance.py
"""
Strategy Governance and Anti-Overfitting Lock Engine.
Enforces that frozen models, hyperparameters, feature pipelines, and rules
cannot be silently mutated or overfitted during prospective validation.
Any parameter change requires creating a new strategy version and resetting
the prospective validation holdout.
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("StrategyGovernance")


class StrategyLockError(Exception):
    """Raised when an attempt is made to execute an altered strategy under an active lock."""
    pass


class StrategyGovernanceEngine:
    """
    Immutable Strategy Versioning and Anti-Overfitting Lock Guardian.
    """

    def __init__(self, manifest_path: Optional[str] = None):
        if manifest_path is None:
            self.manifest_path = Path(__file__).resolve().parent.parent.parent / "artifacts" / "frozen_strategy_manifest.json"
        else:
            self.manifest_path = Path(manifest_path)
        self.backend_dir = Path(__file__).resolve().parent.parent.parent
        self._manifest_cache: Optional[Dict[str, Any]] = None

    def load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Frozen strategy manifest not found at {self.manifest_path}")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self._manifest_cache = json.load(f)
        return self._manifest_cache

    @staticmethod
    def compute_file_hash(path: Path) -> str:
        if not path.exists():
            return "FILE_NOT_FOUND"
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def verify_integrity(self) -> Tuple[bool, List[str]]:
        """
        Verifies that all active model weights, configurations, and core code files
        match the cryptographic hashes in the frozen manifest.
        Returns:
            is_valid: bool
            violations: List[str] of mismatched artifacts
        """
        manifest = self.load_manifest()
        violations = []

        # 1. Model Hashes
        for rel_path, expected_hash in manifest.get("model_hashes", {}).items():
            full_path = self.backend_dir / "artifacts" / rel_path.replace("artifacts/", "")
            actual_hash = self.compute_file_hash(full_path)
            if actual_hash != expected_hash:
                violations.append(
                    f"Model artifact {rel_path} mutated! Expected SHA-256 {expected_hash[:12]}..., got {actual_hash[:12]}..."
                )

        # 2. Config Hashes
        for rel_path, expected_hash in manifest.get("config_hashes", {}).items():
            full_path = self.backend_dir / "configs" / rel_path.replace("configs/", "")
            actual_hash = self.compute_file_hash(full_path)
            if actual_hash != expected_hash:
                violations.append(
                    f"Configuration {rel_path} mutated! Expected SHA-256 {expected_hash[:12]}..., got {actual_hash[:12]}..."
                )

        # 3. Code Hashes
        for rel_path, expected_hash in manifest.get("code_hashes", {}).items():
            full_path = self.backend_dir / rel_path
            if not full_path.exists():
                full_path = self.backend_dir / "src" / "execution" / rel_path.split("/")[-1]
            actual_hash = self.compute_file_hash(full_path)
            if actual_hash != expected_hash:
                violations.append(
                    f"Execution code {rel_path} modified! Expected SHA-256 {expected_hash[:12]}..., got {actual_hash[:12]}..."
                )

        is_valid = len(violations) == 0
        return is_valid, violations

    def enforce_anti_overfitting_lock(self) -> None:
        """
        Strict check: raises StrategyLockError if any frozen parameter or artifact was modified.
        """
        is_valid, violations = self.verify_integrity()
        if not is_valid:
            error_msg = (
                f"[ANTI-OVERFITTING LOCK ENGAGED] Strategy mutation rejected! "
                f"{len(violations)} integrity violations detected:\n" + "\n".join(violations)
            )
            logger.critical(error_msg)
            raise StrategyLockError(error_msg)

    def get_strategy_version(self) -> str:
        manifest = self.load_manifest()
        return manifest.get("strategy_version", "HYDRA_PROSPECTIVE_V1.0")

    def get_governance_status(self) -> Dict[str, Any]:
        manifest = self.load_manifest()
        is_valid, violations = self.verify_integrity()
        return {
            "strategy_version": manifest.get("strategy_version"),
            "freeze_timestamp_utc": manifest.get("freeze_timestamp_utc"),
            "freeze_timestamp_new_york": manifest.get("freeze_timestamp_new_york"),
            "freeze_display_new_york": manifest.get("freeze_display_new_york"),
            "freeze_session_edt": manifest.get("freeze_session_edt"),
            "git_commit": manifest.get("git_commit"),
            "lock_active": manifest.get("anti_overfitting_lock", True),
            "integrity_verified": is_valid,
            "violations_count": len(violations),
            "violations": violations,
            "model_provenance": manifest.get("model_provenance", {}),
            "prospective_sequence": manifest.get("prospective_sequence", {}),
            "frozen_hyperparameters": manifest.get("frozen_hyperparameters", {}),
            "validation_policy": manifest.get("validation_policy", {}),
        }
