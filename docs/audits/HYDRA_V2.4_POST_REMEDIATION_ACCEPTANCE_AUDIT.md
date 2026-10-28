# HYDRA V2.4.1 — MASTER POST-REMEDIATION ACCEPTANCE AUDIT & INDEPENDENT CODE-LEVEL VERIFICATION

**Audit Date:** 2026-10-11T21:55:00+05:30  
**Evaluated Target:** HYDRA V2.4.1 Working-Tree Remediation Package (Legacy Manifest Digest Hardening, Single-Buffer Manifest Parsing, Verbatim Dry-Run Isolation, Artifact Byte Provenance, and Governance Threshold Restoration)  
**Evaluation HEAD:** `88447a6c3c4fd8dfd818cb8c2a9eebde4f437e01`  
**Local `main` Commit:** `f69fccf74b1224caccdbb41d65386a7f927edd2b`  
**Remote `origin/main` Commit:** `6626496196ec34d8457efe3312980cd14ebbc626` (verified via `git ls-remote`)  
**Current Working Branch:** `research/v2.3_phase_3a_remediation`  
**Governance Designation:** `HYDRA_PROSPECTIVE_V2.4.1` (Candidate status: `PROPOSED_CANDIDATE_PENDING_AUTHORIZATION`)  
**Acceptance Decision:** **NOT APPROVED FOR COMMIT, MERGE, OR PUSH TO MAIN**  
**Live Capital Status:** **STRICTLY BLOCKED** (Research simulation and paper trading only)  

---

## 1. Executive Summary & Review Finding Reconciliation

This acceptance audit report records the adversarial, code-level verification of the uncommitted working-tree modifications addressing the four critical findings and blockers identified in the HYDRA V2.4.1 acceptance assessment:

1. **Strict Legacy Manifest Hash Validation (Blocker A — RESOLVED & VERIFIED):**
   - **Vulnerability Identified:** The initial implementation used `if actual_m_hash and actual_m_hash != V24_EXPECTED_MANIFEST_HASH: return False`. If `_manifest_file_hash` was missing (`None`), empty (`""`), or whitespace, the check failed open, allowing an unverified manifest to claim the legacy exception.
   - **Hardened Fix:** The check now strictly fails closed unless `_manifest_file_hash` is present, is a non-empty string, consists of exactly 64 valid lowercase hexadecimal characters, and exactly matches the authoritative SHA-256 digest `d580b50a441c2f5a3881bb72eb0c8b23d90077dea61ddc506193affcfe993da5`. Furthermore, if raw bytes are attached, the SHA-256 digest recomputed over those bytes must also match.
   - **Verification:** Unit test `test_14` in `backend/tests/test_isolated_ledger_safety.py` now includes 19 comprehensive cases covering missing hash, empty hash, whitespace hash, non-string types, malformed lengths (63, 65), non-hex characters, altered digests, altered raw bytes, target version mismatches, and authentic on-disk V2.4 manifest byte loading.

2. **Single-Buffer Raw-Byte Manifest Parsing (Blocker A / Architecture — RESOLVED & VERIFIED):**
   - **Vulnerability Identified:** The earlier implementation read `raw_bytes` to compute the digest, but then performed a separate file read via `with open(p) as f: json.load(f)`. This introduced a double-read file I/O risk and relied on post-populating the dictionary.
   - **Hardened Fix:** `_load_manifest_data()` in `backend/src/execution/inference_service.py` reads `raw_bytes = p.read_bytes()`, computes `m_hash = hashlib.sha256(raw_bytes).hexdigest()`, and deserializes the exact same byte buffer via `data = json.loads(raw_bytes.decode("utf-8"))`. The authentic byte digest is bound to `data["_manifest_file_hash"]` and `data["_manifest_raw_bytes"] = raw_bytes`.
   - **Verification:** Case 19 of `test_14` verifies that `service._load_manifest_data("Institutional_Mesh_V2.4")` attaches both `_manifest_file_hash` and `_manifest_raw_bytes`, and that recomputing SHA-256 over `_manifest_raw_bytes` exactly matches `_manifest_file_hash`.

3. **Restoration of Authoritative Governance Thresholds (Blocker D — RESOLVED & VERIFIED):**
   - **Discrepancy Identified:** The previous audit report referenced a downgraded threshold of $\ge 30$ closed prospective trades. The authoritative HYDRA governance requirement established in V2.3 and V2.4 is **at least 50 independent closed prospective trades (Target: 100)** across SPY SMA200 market regimes.
   - **Correction:** The downgraded 30-trade threshold has been completely repudiated and removed from all status tables and limitation sections. The authoritative requirements ($\ge 50$ closed trades, target 100, Brier $\le 0.20$, ECE $\le 0.08$) are restored across all documentation and formally asserted in unit test `test_17` in `test_isolated_ledger_safety.py`.

4. **Candidate-Status Validation Hardening (Finding 1 — RESOLVED & VERIFIED):**
   - **Vulnerability Identified:** The previous dry-run implementation checked substring `"CANDIDATE" in cand_status`, which inadvertently accepted prohibited states like `REVOKED_CANDIDATE`, `SUSPENDED_CANDIDATE`, `REJECTED_CANDIDATE`, and `UNAUTHORIZED_CANDIDATE`.
   - **Hardened Fix:** Replaced substring checking with strict two-stage validation: (1) Rejection of any status containing prohibited terms (`PROHIBITED_CANDIDATE_STATUS_TERMS`), and (2) Strict allowlist validation against `permitted_candidate_statuses` defined in `SUPPORTED_RESEARCH_CANDIDATE_VERSIONS["Institutional_Mesh_V2.4.1"]`.
   - **Verification:** Unit test `test_18` asserts that `REVOKED_CANDIDATE`, `SUSPENDED_CANDIDATE`, `REJECTED_CANDIDATE`, `UNAUTHORIZED_CANDIDATE`, empty/missing statuses, and non-permitted states are strictly rejected, while authentic pending candidate states succeed.

5. **Cross-Directory Artifact Resolution & Digest Isolation (Finding 2 — RESOLVED & VERIFIED):**
   - **Vulnerability Identified:** In-memory artifacts with mismatched hashes could cause confusing fallback behavior, and path resolution across different execution roots (repo root vs. `backend/`) required deterministic multi-root search without implicit version fallback.
   - **Hardened Fix:** Implemented multi-root search (`manifest_path.parent`, `artifacts/`, `backend/artifacts/`) with immediate discard of in-memory artifacts whose `_artifact_hash` differs from the candidate manifest. If disk artifacts are missing, returns `MISSING_ARTIFACTS`. If disk or in-memory artifacts carry altered/V3 hashes, returns `CRYPTOGRAPHIC_DIGEST_MISMATCH`. Feature pair dimensions (scaler vs. XGB) and policy dimensions (31) are strictly verified.
   - **Verification:** Unit test `test_20` proves authentic resolution, missing artifact rejection (`MISSING_ARTIFACTS`), tampered digest rejection (`CRYPTOGRAPHIC_DIGEST_MISMATCH`), cross-version V3 substitution rejection, and feature dimension mismatch rejections (`DIMENSION_MISMATCH`, `POLICY_DIMENSION_MISMATCH`).

6. **Candidate-Version Mapping & Baseline Prohibition (Finding 3 — RESOLVED & VERIFIED):**
   - **Vulnerability Identified:** Dry-run evaluation lacked an explicit whitelist policy, leaving open whether non-candidates or unvetted versions could be evaluated.
   - **Hardened Fix:** Restricts research dry-run strictly to `SUPPORTED_RESEARCH_CANDIDATE_VERSIONS`. Immutable baselines like `Institutional_Mesh_V2.4`, `Institutional_Mesh_V3`, or unrecognized versions are strictly rejected with `UNSUPPORTED_RESEARCH_CANDIDATE_VERSION`. Furthermore, `strategy_version` in the manifest must exactly match the expected candidate mapping (`HYDRA_PROSPECTIVE_V2.4.1`) or fail closed with `VERSION_LABEL_MISMATCH`.
   - **Verification:** Unit test `test_19` verifies that `Institutional_Mesh_V2.4` (baseline), `Institutional_Mesh_V3`, and unvetted versions are strictly rejected, and that label mismatches fail closed.

7. **Dry-Run Exit Logic Look-Ahead Bias Resolution (Finding 4 — RESOLVED & VERIFIED):**
   - **Vulnerability Identified:** The research candidate dry-run previously evaluated exit decisions using the next bar's open (`open_next < entry_price * 0.95 or open_next > entry_price * 1.10`) before executing at that same next open. This violated temporal causality by allowing future price data ($Open[t+1]$) to decide whether to exit at bar $t$.
   - **Hardened Fix:** The exit decision is evaluated strictly and exclusively on information available by bar $t$ close: `close_t <= entry_price * 0.95 or close_t >= entry_price * 1.10 or p_sell >= 0.50`. The next-bar open (`open_next = float(df.loc[idx_next, "Open"])`) is strictly quarantined to order pricing (`exit_price = open_next * 0.9995`) for orders already decided at bar $t$.
   - **Verification:** Regression unit test `test_21` in `backend/tests/test_isolated_ledger_safety.py` proves that mutating $Open[t+1]$ (e.g. crashing to 50.0 or surging to 400.0) cannot induce an exit decision at bar $t$, and that genuine bar $t$ exit decisions execute invariantly at the next open with the respective fill price.

8. **Independent Test Reproduction & Ruff Validation (RESOLVED & VERIFIED):**
   - **Verification:** All tests were independently executed against the actual working tree:
     - `test_isolated_ledger_safety.py`: **21 tests passed in 4.966s (Exit code 0)**.
     - Full 5-suite matrix (`test_isolated_ledger_safety.py`, `test_signal_recovery_regression.py`, `test_causality_and_execution_timing.py`, `test_signal_integrity.py`, `test_execution_cost_governance.py`): **40 tests passed in 6.784s (Exit code 0)**.
     - `ruff check`: **All checks passed! 0 errors (Exit code 0)**.

---

## 2. Master Review-Finding Status Table

| Blocker / Finding | Authoritative Governance Requirement | Hardened Architectural Remediation | Verification Status | Exact Verifiable Evidence |
|---|---|---|---|---|
| **Blocker A: Legacy Authorization Exception** | Strictly require verified manifest hash; reject missing, empty, malformed, or mismatched digests; parse exact byte buffer; reject all versions except frozen V2.4 baseline. | Scoped exception in `is_manifest_formally_authorized` to `Institutional_Mesh_V2.4` and `HYDRA_PROSPECTIVE_V2.4` with mandatory valid 64-char hex SHA-256 digest matching `d580b50a...`. Single byte buffer parsed via `json.loads(raw_bytes.decode('utf-8'))`. | **PASS** | `test_14` (Cases 1–19) in `backend/tests/test_isolated_ledger_safety.py`. All negative cases strictly fail closed. |
| **Blocker B: Dry-Run Isolation** | Inspect verbatim `replay_research_candidate_dry_run()` and transitive call tree. Verify zero database writes, zero prospective ledger modifications, zero order creation, and distinct dry-run markers. | Audited verbatim implementation. Entire pathway operates in memory with zero calls to `SignalLedger`, `paper_engine`, or SQLite. Output tagged with `is_dry_run: True` and `model_version: Institutional_Mesh_V2.4.1_RESEARCH_DRY_RUN`. | **PASS** | `test_16` in `backend/tests/test_isolated_ledger_safety.py`. Spies on all database and order routing methods confirmed 0 calls across all scenarios. |
| **Blocker C: Artifact Provenance** | Enumerate all model/scaler loading sites. Verify deserialization from exact raw bytes hashed. Confirm missing or mismatched provenance fails closed with no disk fallback. | Deserialization switched to `io.BytesIO(raw_bytes)` and `bytearray(raw_bytes)` across `ModelLoader` and `InferenceService`. `_verify_manifest_hashes()` fails closed if `_artifact_hash` is missing or mismatched. | **PASS** | `test_15` in `test_isolated_ledger_safety.py` and `test_2` in `test_signal_recovery_regression.py`. Tested unbound objects, swapped digests, and non-string hashes. |
| **Blocker D: Governance Thresholds** | Enforce authoritative thresholds: Brier $\le 0.20$, ECE $\le 0.08$, $\ge 50$ independent closed prospective trades (target 100), SPY SMA200 regime coverage, next-bar Open execution. Reject downgraded 30-trade threshold. | Restored authoritative threshold across code, tests, and audit documentation. Added `test_17` formally asserting that 30 trades is rejected and that current records fail closed. | **PASS (Gates Enforced) / FAIL (Strategy Gating)** | `test_17` in `test_isolated_ledger_safety.py`. Current platform records (Brier 0.5394, ECE 0.2076, 0 closed trades) fail gates; live capital routing remains strictly blocked. |
| **Finding 1: Candidate Status Validation** | Explicit allowlist of permitted candidate states. Separately and explicitly reject revoked, rejected, suspended, and unauthorized states. | Configured `PROHIBITED_CANDIDATE_STATUS_TERMS` and `permitted_candidate_statuses` in `SUPPORTED_RESEARCH_CANDIDATE_VERSIONS`. Fails closed if any prohibited term is present or status not allowlisted. | **PASS** | `test_18` in `test_isolated_ledger_safety.py`. Verified rejection of `REVOKED`, `SUSPENDED`, `REJECTED`, `UNAUTHORIZED`, etc. |
| **Finding 2: Artifact Fallback & Isolation** | Cross-directory resolution (`repo root` vs `backend/`), explicit fail-closed detection of missing files vs cryptographic digest mismatches, dimension verification. | Multi-root directory search, in-memory artifact isolation, distinct error codes (`MISSING_ARTIFACTS`, `CRYPTOGRAPHIC_DIGEST_MISMATCH`, `DIMENSION_MISMATCH`). | **PASS** | `test_20` in `test_isolated_ledger_safety.py`. Verified authentic load, missing file, tampered hash, V3 substitution, and dimension mismatches. |
| **Finding 3: Candidate Version Mapping Policy** | Explicit expected strategy version for every candidate; reject unknown, unsupported versions and immutable baselines like V2.4. | Enforced `SUPPORTED_RESEARCH_CANDIDATE_VERSIONS`. Fails closed with `UNSUPPORTED_RESEARCH_CANDIDATE_VERSION` or `VERSION_LABEL_MISMATCH`. | **PASS** | `test_19` in `test_isolated_ledger_safety.py`. Verified rejection of V2.4 baseline, V3, and label mismatch. |
| **Finding 4: Exit Decision Temporal Causality** | Exit decisions at bar $t$ must depend exclusively on bar $t$ information (`close_t`, `p_sell`, `entry_price`). Next-bar open $Open[t+1]$ must only price fill execution. | Replaced look-ahead check `open_next < entry_price * 0.95` with strictly causal `close_t <= entry_price * 0.95 or close_t >= entry_price * 1.10`. Quarantined `open_next` to `exit_price = open_next * 0.9995`. | **PASS** | `test_21` in `test_isolated_ledger_safety.py`. Verified Open[t+1] gap invariance and next-open pricing consistency. |
| **Linter & Code Hygiene** | Zero errors under `ruff check`. | Clean Ruff run across all modified backend sources and tests. | **PASS** | `ruff check src/ tests/...` returned exit code 0 (`All checks passed!`). |

---

## 3. Forensic Code Audits & Implementation Verification

### 3.1 Hardened Legacy Authorization Exception & Manifest Byte Parsing

In `backend/src/execution/inference_service.py`:

```python
    VERSION_CONFIGS: Dict[str, Any] = {
        "Institutional_Mesh_V2.4.1": {
            "manifest_candidates": [
                Path("artifacts/frozen_strategy_manifest_v2.4.1.json"),
                Path("backend/artifacts/frozen_strategy_manifest_v2.4.1.json"),
            ],
            "allowed_dims": {31},
            "model_key": "xgb_ensemble.json",
            "scaler_key": "latest_scaler.joblib",
        },
        "Institutional_Mesh_V2.4": {
            "manifest_candidates": [
                Path("artifacts/frozen_strategy_manifest_v2.4.json"),
                Path("backend/artifacts/frozen_strategy_manifest_v2.4.json"),
            ],
            "allowed_dims": {27, 31},
            "model_key": "xgb_ensemble.json",
            "scaler_key": "latest_scaler.joblib",
        },
        "Institutional_Mesh_V3": {
            "manifest_candidates": [
                Path("artifacts/frozen_strategy_manifest_v3.0.json"),
                Path("backend/artifacts/frozen_strategy_manifest_v3.0.json"),
            ],
            "allowed_dims": {31},
            "model_key": "v3/xgb_ensemble.json",
            "scaler_key": "v3/latest_scaler.joblib",
        },
    }

    def _load_manifest_data(self, ver: str) -> Optional[Dict[str, Any]]:
        """
        Loads manifest JSON by reading raw file bytes once, calculating SHA-256
        directly over those raw bytes, and deserializing the exact same byte buffer.
        Attaches _manifest_file_hash, _manifest_raw_bytes, and _manifest_path.
        """
        cfg = self.VERSION_CONFIGS.get(ver, {})
        for p in cfg.get("manifest_candidates", []):
            if p.exists():
                try:
                    raw_bytes = p.read_bytes()
                    m_hash = hashlib.sha256(raw_bytes).hexdigest()
                    data = json.loads(raw_bytes.decode("utf-8"))
                    if isinstance(data, dict):
                        data["_manifest_file_hash"] = m_hash
                        data["_manifest_raw_bytes"] = raw_bytes
                        data["_manifest_path"] = str(p)
                        return data
                    return None
                except Exception:
                    return None
        return None

    @staticmethod
    def is_manifest_formally_authorized(
        manifest_data: Dict[str, Any], target_version: Optional[str] = None
    ) -> bool:
        """
        Evaluates formal governance authorization with strict negative precedence.
        """
        if not isinstance(manifest_data, dict):
            return False

        unauthorized_terms = {
            "PROPOSED",
            "PENDING",
            "REVOKED",
            "REJECTED",
            "UNAUTHORIZED",
            "DRAFT",
            "SUSPENDED",
            "CANDIDATE",
        }

        # 1. Candidate status negative check
        cand_status = str(manifest_data.get("candidate_status", "")).strip().upper()
        if any(term in cand_status for term in unauthorized_terms):
            return False

        # 2. Top-level status negative check
        top_status = str(manifest_data.get("status", "")).strip().upper()
        if any(term in top_status for term in unauthorized_terms):
            return False

        # 3. Authorization record inspection
        auth_record = manifest_data.get("authorization_record")
        if auth_record is not None:
            if not isinstance(auth_record, dict):
                return False
            auth_status = str(auth_record.get("status", "")).strip().upper()
            if any(term in auth_status for term in unauthorized_terms):
                return False
            if auth_record.get("formal_authorization_granted") is not True:
                return False
            return True

        # If candidate status was set but no authorization record exists, fail closed
        if cand_status:
            return False

        # 4. Narrowly scoped legacy baseline exception:
        strat_ver = str(manifest_data.get("strategy_version", "")).strip()
        resolved_target = (target_version or "").strip()

        V24_EXPECTED_MANIFEST_HASH = "d580b50a441c2f5a3881bb72eb0c8b23d90077dea61ddc506193affcfe993da5"
        actual_m_hash = manifest_data.get("_manifest_file_hash")
        if not isinstance(actual_m_hash, str) or not actual_m_hash.strip():
            return False
        actual_m_hash_clean = actual_m_hash.strip().lower()
        if len(actual_m_hash_clean) != 64 or not all(c in "0123456789abcdef" for c in actual_m_hash_clean):
            return False
        if actual_m_hash_clean != V24_EXPECTED_MANIFEST_HASH.lower():
            return False

        raw_bytes = manifest_data.get("_manifest_raw_bytes")
        if not isinstance(raw_bytes, (bytes, bytearray)) or not raw_bytes:
            return False
        computed_from_bytes = hashlib.sha256(raw_bytes).hexdigest().lower()
        if computed_from_bytes != V24_EXPECTED_MANIFEST_HASH.lower():
            return False
        if computed_from_bytes != actual_m_hash_clean:
            return False

        try:
            parsed_raw = json.loads(raw_bytes.decode("utf-8"))
            if not isinstance(parsed_raw, dict):
                return False
            if str(parsed_raw.get("strategy_version", "")).strip() != strat_ver:
                return False
            if str(parsed_raw.get("git_commit", "")).strip() != "e687e2321da9159dca2b10c6744f2166fb44a506":
                return False
            if str(parsed_raw.get("freeze_timestamp_utc", "")).strip() != "2026-10-01T09:39:11Z":
                return False
            if str(parsed_raw.get("status", "")).strip() != "FROZEN_FOR_PROSPECTIVE_VALIDATION":
                return False
        except Exception:
            return False

        is_designated_v24_baseline = (
            (resolved_target == "Institutional_Mesh_V2.4" or (not resolved_target and strat_ver == "HYDRA_PROSPECTIVE_V2.4"))
            and strat_ver == "HYDRA_PROSPECTIVE_V2.4"
            and str(manifest_data.get("git_commit", "")).strip() == "e687e2321da9159dca2b10c6744f2166fb44a506"
            and str(manifest_data.get("freeze_timestamp_utc", "")).strip() == "2026-10-01T09:39:11Z"
            and top_status == "FROZEN_FOR_PROSPECTIVE_VALIDATION"
            and manifest_data.get("anti_overfitting_lock") is True
            and not cand_status
        )
        if is_designated_v24_baseline:
            return True

        return False
```

#### Test Evidence: `test_14` (Cases 1–19, with Cases 17a–17f)
- **Case 1 (Conflicting candidate manifest):** `None` (`[GOVERNANCE BLOCKED]`).
- **Case 2 (Revoked candidate):** `None` (`[GOVERNANCE BLOCKED]`).
- **Case 3 (Malformed authorization record types):** `None` across string, int, list, bool.
- **Case 4 (Candidate status set without authorization block):** `None`.
- **Case 5 (Authentic V2.4 baseline with bound raw bytes):** Returns `"Institutional_Mesh_V2.4"`.
- **Case 6 (Identical manifest with `strategy_version: "HYDRA_PROSPECTIVE_V2.5"`):** Returns `None`.
- **Case 7 (Identical manifest with altered `git_commit`):** Returns `None`.
- **Case 8 (Identical manifest with altered `freeze_timestamp_utc`):** Returns `None`.
- **Case 9 (Identical manifest with altered `_manifest_file_hash`):** Returns `None`.
- **Case 10 (Missing `_manifest_file_hash` key):** `is_manifest_formally_authorized` returns `False`.
- **Case 11 (None `_manifest_file_hash` value):** `is_manifest_formally_authorized` returns `False`.
- **Case 12 (Empty string `_manifest_file_hash`):** `is_manifest_formally_authorized` returns `False`.
- **Case 13 (Whitespace-only `_manifest_file_hash`):** `is_manifest_formally_authorized` returns `False`.
- **Case 14 (Non-string `_manifest_file_hash` types):** `is_manifest_formally_authorized` returns `False` for int, list, dict, bool.
- **Case 15 (Malformed hex digest lengths 63, 65):** `is_manifest_formally_authorized` returns `False`.
- **Case 16 (Non-hex characters in digest):** `is_manifest_formally_authorized` returns `False`.
- **Case 17a (Missing `_manifest_raw_bytes` key entirely):** `is_manifest_formally_authorized` returns `False` (zero caller-side assumptions).
- **Case 17b (None `_manifest_raw_bytes`):** `is_manifest_formally_authorized` returns `False`.
- **Case 17c (Empty `_manifest_raw_bytes` `b""`):** `is_manifest_formally_authorized` returns `False`.
- **Case 17d (Non-bytes `_manifest_raw_bytes` type):** `is_manifest_formally_authorized` returns `False`.
- **Case 17e (Altered raw bytes attached):** `is_manifest_formally_authorized` returns `False`.
- **Case 17f (Authentic raw bytes, but in-memory dictionary metadata tampered):** `is_manifest_formally_authorized` returns `False`.
- **Case 18 (Target version mismatch `Institutional_Mesh_V2.4.1` or `V3`):** `is_manifest_formally_authorized` returns `False`.
- **Case 19 (Authentic on-disk V2.4 byte loading):** Verified that `_load_manifest_data` attaches `_manifest_file_hash` and `_manifest_raw_bytes`, that computing SHA-256 over `_manifest_raw_bytes` reproduces `_manifest_file_hash`, and that it passes authorization.

---

### 3.2 Complete Verbatim Code Listing of `replay_research_candidate_dry_run()`

Below is the complete, unabridged, verbatim method implementation from `backend/src/execution/inference_service.py`:

```python
    def replay_research_candidate_dry_run(
        self,
        ticker: str,
        ticker_df: pd.DataFrame,
        spy_df: Optional[pd.DataFrame] = None,
        candidate_version: str = "Institutional_Mesh_V2.4.1",
    ) -> Dict[str, Any]:
        """
        Isolated Research-Only Candidate Dry-Run Pathway.

        Strict Governance Invariants:
        1. Zero Production Routing: This method is strictly isolated for offline/research evaluation.
        2. Zero Ledger Writes: NEVER writes or modifies records in SignalLedger or prospective_signals SQLite tables.
        3. Zero Live Orders: NEVER routes or creates live execution orders.
        4. Cryptographic Provenance Verification: Model and scaler digests are verified against the candidate manifest.
        5. Fails closed if artifact digests do not match candidate manifest.
        """
        result: Dict[str, Any] = {
            "symbol": ticker,
            "candidate_version": candidate_version,
            "is_research_dry_run": True,
            "production_routing_blocked": True,
            "ledger_writes_blocked": True,
            "status": "FAILED",
            "marker_count": 0,
            "markers": [],
            "error": None,
        }

        df = clean_multiindex_columns(ticker_df.copy())
        if df.empty or len(df) < 30:
            result["error"] = "INSUFFICIENT_BARS"
            return result

        # 1. Version Policy: Strictly restrict to supported research candidates
        cand_policy = self.SUPPORTED_RESEARCH_CANDIDATE_VERSIONS.get(candidate_version)
        if not cand_policy:
            result["error"] = (
                f"UNSUPPORTED_RESEARCH_CANDIDATE_VERSION: '{candidate_version}'. "
                f"Dry-run evaluation is strictly restricted to designated research candidates: "
                f"{sorted(self.SUPPORTED_RESEARCH_CANDIDATE_VERSIONS.keys())}. "
                f"Production baselines and unrecognized versions are prohibited."
            )
            return result

        manifest_data = self._load_manifest_data(candidate_version)
        if not manifest_data:
            result["error"] = f"CANDIDATE_MANIFEST_NOT_FOUND: {candidate_version}"
            return result

        # 2. Candidate Status Validation: Explicit rejection of prohibited terms + strict allowlist check
        cand_status = str(manifest_data.get("candidate_status", "")).strip().upper()
        if not cand_status:
            result["error"] = (
                f"MISSING_CANDIDATE_STATUS: candidate_version='{candidate_version}' has empty candidate_status"
            )
            return result

        if any(term in cand_status for term in self.PROHIBITED_CANDIDATE_STATUS_TERMS):
            result["error"] = f"PROHIBITED_CANDIDATE_STATUS: '{cand_status}' contains prohibited governance terms"
            return result

        permitted_statuses = cand_policy["permitted_candidate_statuses"]
        if cand_status not in permitted_statuses:
            result["error"] = (
                f"UNPERMITTED_CANDIDATE_STATUS: '{cand_status}'. "
                f"Permitted research statuses for {candidate_version}: {sorted(permitted_statuses)}"
            )
            return result

        # 3. Strategy Version Identity Validation
        declared_strat = str(manifest_data.get("strategy_version", "")).strip()
        expected_strat = cand_policy["expected_strategy_version"]
        if declared_strat != expected_strat:
            result["error"] = (
                f"VERSION_LABEL_MISMATCH: requested '{candidate_version}', "
                f"expected strategy '{expected_strat}', but manifest declared '{declared_strat}'"
            )
            return result

        # 4. Manifest Model Hashes
        model_key = cand_policy["model_key"]
        scaler_key = cand_policy["scaler_key"]
        expected_xgb = manifest_data.get("model_hashes", {}).get(model_key)
        expected_scaler = manifest_data.get("model_hashes", {}).get(scaler_key)

        if not expected_xgb or not expected_scaler:
            result["error"] = f"MANIFEST_MISSING_MODEL_HASHES: {model_key}, {scaler_key}"
            return result

        # 5. Artifact Resolution across Working Directories (Repo-Root and Backend/)
        manifest_path = manifest_data.get("_manifest_path")
        search_dirs: List[Path] = []
        if manifest_path:
            m_parent = Path(manifest_path).parent
            if m_parent not in search_dirs:
                search_dirs.append(m_parent)
        for base_p in [Path("artifacts"), Path("backend/artifacts")]:
            if base_p not in search_dirs:
                search_dirs.append(base_p)

        # Discard or reload in-memory artifacts if their hashes do not match candidate manifest
        scaler = getattr(self.mm, "scaler", None)
        in_memory_scaler_bad = False
        if scaler is not None and getattr(scaler, "_artifact_hash", None) != expected_scaler:
            in_memory_scaler_bad = True
            scaler = None

        if scaler is None:
            for d in search_dirs:
                sp = d / scaler_key
                if sp.exists():
                    try:
                        loaded = load_bound_scaler(sp)
                        scaler = loaded
                        if getattr(loaded, "_artifact_hash", None) == expected_scaler:
                            break
                    except Exception:
                        pass

        xgb_model = getattr(self.mm, "xgb_model", None)
        in_memory_xgb_bad = False
        if xgb_model is not None and getattr(xgb_model, "_artifact_hash", None) != expected_xgb:
            in_memory_xgb_bad = True
            xgb_model = None

        if xgb_model is None:
            for d in search_dirs:
                mp = d / model_key
                if mp.exists():
                    try:
                        loaded = load_bound_xgb(mp)
                        xgb_model = loaded
                        if getattr(loaded, "_artifact_hash", None) == expected_xgb:
                            break
                    except Exception:
                        pass

        if scaler is None or xgb_model is None:
            if in_memory_scaler_bad or in_memory_xgb_bad:
                result["error"] = "CRYPTOGRAPHIC_DIGEST_MISMATCH"
            else:
                result["error"] = "MISSING_ARTIFACTS"
            return result

        actual_xgb = getattr(xgb_model, "_artifact_hash", None)
        actual_scaler = getattr(scaler, "_artifact_hash", None)

        if not actual_xgb or actual_xgb != expected_xgb or not actual_scaler or actual_scaler != expected_scaler:
            result["error"] = "CRYPTOGRAPHIC_DIGEST_MISMATCH"
            return result

        # 6. Strict Dimension & Pair Integrity Verification
        scaler_n = getattr(scaler, "n_features_in_", None)
        xgb_n = getattr(xgb_model, "n_features_in_", None)
        expected_dim = cand_policy["expected_feature_count"]
        manifest_dim = manifest_data.get("feature_schema", {}).get("feature_count")

        if scaler_n is None or xgb_n is None or scaler_n != xgb_n:
            result["error"] = f"DIMENSION_MISMATCH: scaler={scaler_n}, model={xgb_n}"
            return result
        if scaler_n != expected_dim:
            result["error"] = f"POLICY_DIMENSION_MISMATCH: scaler={scaler_n}, policy_expected={expected_dim}"
            return result
        if manifest_dim is not None and scaler_n != manifest_dim:
            result["error"] = f"SCHEMA_DIMENSION_MISMATCH: scaler={scaler_n}, manifest_schema={manifest_dim}"
            return result

        # Run feature calculation and dry-run inference
        spy = clean_multiindex_columns(spy_df.copy()) if spy_df is not None else None
        df_feat = add_upgraded_features(df.copy(), spy, None)
        df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()].copy()

        active_cols = (
            FEATURE_COLUMNS_V30
            if getattr(scaler, "n_features_in_", 0) == len(FEATURE_COLUMNS_V30)
            else FEATURE_COLUMNS
        )
        df_filtered = df_feat.reindex(columns=active_cols).dropna()
        if df_filtered.empty or len(df_filtered) < 15:
            result["error"] = "INSUFFICIENT_VALID_BARS"
            return result

        scaled_rows = scaler.transform(df_filtered.values)
        xgb_probs = xgb_model.predict_proba(scaled_rows)

        # In-memory signal calculation ONLY (NO persistence, NO SignalLedger calls)
        staged_markers = []
        in_pos = False
        entry_price = 0.0
        pos_type = None

        valid_indices = df_filtered.index
        for i in range(len(valid_indices) - 1):
            idx_t = valid_indices[i]
            idx_next = valid_indices[i + 1]
            p_sell, p_hold, p_buy = xgb_probs[i]
            close_t = float(df.loc[idx_t, "Close"])
            open_next = float(df.loc[idx_next, "Open"])

            if not in_pos:
                if p_buy >= 0.55 and p_buy > p_sell:
                    in_pos = True
                    pos_type = "LONG"
                    entry_price = open_next * 1.0005
                    staged_markers.append({
                        "symbol": ticker,
                        "action": "BUY",
                        "bar_timestamp": (
                            idx_t.strftime("%Y-%m-%d") if hasattr(idx_t, "strftime") else str(idx_t)[:10]
                        ),
                        "model_version": f"{candidate_version}_RESEARCH_DRY_RUN",
                        "strategy_version": declared_strat,
                        "manifest_file_hash": str(manifest_data.get("_manifest_file_hash", "")),
                        "scaler_hash": actual_scaler,
                        "xgb_hash": actual_xgb,
                        "execution_price": entry_price,
                        "is_dry_run": True,
                    })
            else:
                if pos_type == "LONG" and (
                    p_sell >= 0.50 or close_t <= entry_price * 0.95 or close_t >= entry_price * 1.10
                ):
                    in_pos = False
                    exit_price = open_next * 0.9995
                    staged_markers.append({
                        "symbol": ticker,
                        "action": "SELL",
                        "bar_timestamp": (
                            idx_t.strftime("%Y-%m-%d") if hasattr(idx_t, "strftime") else str(idx_t)[:10]
                        ),
                        "model_version": f"{candidate_version}_RESEARCH_DRY_RUN",
                        "strategy_version": declared_strat,
                        "manifest_file_hash": str(manifest_data.get("_manifest_file_hash", "")),
                        "scaler_hash": actual_scaler,
                        "xgb_hash": actual_xgb,
                        "execution_price": exit_price,
                        "is_dry_run": True,
                    })
                    pos_type = None

        result["status"] = "SUCCESS"
        result["strategy_version"] = declared_strat
        result["manifest_file_hash"] = str(manifest_data.get("_manifest_file_hash", ""))
        result["scaler_hash"] = actual_scaler
        result["xgb_hash"] = actual_xgb
        result["feature_dimension"] = scaler_n
        result["marker_count"] = len(staged_markers)
        result["markers"] = staged_markers
        return result
```

#### Transitive Call-Tree Audit
1. `clean_multiindex_columns(df)`: Read-only pandas column manipulation. Zero I/O or network.
2. `self._load_manifest_data(candidate_version)`: Reads manifest JSON via single-buffer `read_bytes` and `json.loads`. Attaches `_manifest_file_hash` and `_manifest_raw_bytes`. Read-only.
3. Candidate Version & Strategy Label Policy (Finding 3): Checks `candidate_version` against `SUPPORTED_RESEARCH_CANDIDATE_VERSIONS`. Fails closed with `UNSUPPORTED_RESEARCH_CANDIDATE_VERSION` if non-candidate baseline (e.g. `Institutional_Mesh_V2.4` or V3) is passed. Cross-checks `manifest.strategy_version == expected_strategy_version` (`HYDRA_PROSPECTIVE_V2.4.1`), failing closed with `VERSION_LABEL_MISMATCH` if altered.
4. Candidate Status Allowlist & Prohibition (Finding 1): Validates `candidate_status` against `PROHIBITED_CANDIDATE_STATUS_TERMS` (strictly rejecting `REVOKED_CANDIDATE`, `SUSPENDED_CANDIDATE`, `REJECTED_CANDIDATE`, `UNAUTHORIZED_CANDIDATE`, etc.) and enforces membership in `cand_policy["permitted_candidate_statuses"]`. Non-allowlisted or prohibited statuses return explicit error without inference.
5. Cross-Directory Artifact Resolution & Digest Isolation (Finding 2): Resolves artifact locations across `manifest_path.parent`, `artifacts/`, and `backend/artifacts/`. In-memory artifacts with mismatched digests are immediately discarded. Missing artifacts return `MISSING_ARTIFACTS`. Altered or cross-version substituted artifacts return `CRYPTOGRAPHIC_DIGEST_MISMATCH`. Feature pair dimensions (scaler vs. XGB) and policy dimensions (31) are strictly verified (`DIMENSION_MISMATCH`, `POLICY_DIMENSION_MISMATCH`).
6. `add_upgraded_features(df, spy, None)`: In-memory mathematical feature calculation. Zero I/O.
7. `scaler.transform(df.values)`: In-memory Scikit-learn array transformation. Zero I/O.
8. `xgb_model.predict_proba(scaled_rows)`: In-memory XGBoost C++ matrix evaluation. Zero I/O.
9. Marker generation loop: Appends dictionaries containing `model_version`, `strategy_version`, `manifest_file_hash`, `scaler_hash`, `xgb_hash`, and `is_dry_run` to a local Python list.
10. Return value: Pure in-memory dictionary. Zero calls to `SignalLedger`, `paper_engine`, `journal`, or SQLite.
11. **Exit Decision Temporal Causality & Open[t+1] Invariance (Finding 4):** The exit decision is evaluated strictly using bar $t$ finalized information (`close_t = float(df.loc[idx_t, "Close"])`, model `p_sell`, and past `entry_price`). Next bar open (`open_next = float(df.loc[idx_next, "Open"])`) is used exclusively to price fill execution (`exit_price = open_next * 0.9995`) if and only if an exit order was already determined at bar $t$. Mutating $Open[t+1]$ cannot induce, alter, or suppress an exit decision at bar $t$.

---

### 3.3 Complete Load-Path Audit & Raw-Byte Provenance Binding

| Loading Route | Implementation Location | Deserialization Source | Bound Provenance Attributes | Mismatch / Missing Behavior |
|---|---|---|---|---|
| Primary Scaler | `ModelLoader._load_scaler` | `io.BytesIO(raw_bytes)` | `_artifact_hash`, `_artifact_source_path` | Fails closed (`logger.warning`) |
| Primary XGBoost | `ModelLoader._load_xgb` | `bytearray(raw_bytes)` | `_artifact_hash`, `_artifact_source_path` | Fails closed (`logger.warning`) |
| Inference Scaler | `inference_service.load_bound_scaler` | `io.BytesIO(raw_bytes)` | `_artifact_hash`, `_artifact_source_path` | `FileNotFoundError` / fails closed |
| Inference XGBoost | `inference_service.load_bound_xgb` | `bytearray(raw_bytes)` | `_artifact_hash`, `_artifact_source_path` | `FileNotFoundError` / fails closed |
| Verification Gate | `inference_service._verify_manifest_hashes` | In-memory attribute inspection | Requires `_artifact_hash == manifest_hash` | Fails closed with `[PROVENANCE UNVERIFIABLE]` or `[HASH MISMATCH]` |

---

## 4. Authoritative HYDRA Governance Thresholds & Statistical Gating

The authoritative HYDRA governance requirements for production graduation and live capital enablement are:

| Metric / Requirement | Authoritative Threshold | Current Strategy Record | Evaluation Status | Impact on Integration |
|---|---|---|---|---|
| **Multiclass Brier Score** | $\le 0.20$ | **0.5394** | **FAILED** | Absolute blocker for live trading |
| **Expected Calibration Error (ECE)** | $\le 0.08$ | **0.2076** | **FAILED** | Absolute blocker for live trading |
| **Closed Prospective Trades** | $\ge 50$ (Target: 100) | **0 closed trades** | **PENDING / INSUFFICIENT** | Absolute blocker for live trading |
| **Market Regime Coverage** | Verified across Bull ($\text{SPY} > \text{SMA200}$) & Bear ($\text{SPY} \le \text{SMA200}$) | 0 forward observations | **PENDING** | Required for Stage 3 validation |
| **Execution Cost Conventions** | Next-bar Open[t+1] with 5 bps adverse slippage per side + $0.005/share commission (min $1.00) | Implemented in manifests, simulator, & governance tests | **VERIFIED** | Enforced in code (`test_execution_cost_governance.py`) |
| **Forming-Bar Causality** | Forming bar strictly excluded | Implemented in filter | **VERIFIED** | Enforced in code |

> [!CRITICAL]
> **Repudiation of 30-Trade Downgrade:** The previously mentioned 30-trade threshold is formally repudiated as an unapproved downgrade. Authoritative HYDRA governance mandates at least **50 independent closed prospective trades (with 100 as the institutional target)** before any live capital allocation may be considered. Unit test `test_17` formally verifies this constraint.

---

## 5. Independent Test Execution & Verification Results

### Test Reproduction Commands and Outputs

```powershell
# 1. Run atomic ledger safety, provenance, candidate-state, artifact resolution, and causality invariance suite (21 tests)
.\venv\Scripts\python.exe -m unittest tests/test_isolated_ledger_safety.py

# Output:
Ran 21 tests in 4.966s
OK

# 2. Run full 5-suite governance, regression, execution-cost, and signal integrity matrix (40 tests)
.\venv\Scripts\python.exe -m unittest tests/test_isolated_ledger_safety.py tests/test_signal_recovery_regression.py tests/test_causality_and_execution_timing.py tests/test_signal_integrity.py tests/test_execution_cost_governance.py

# Output:
Ran 40 tests in 6.784s
OK

# 3. Verify Ruff linter across backend codebase and test suites (zero errors required)
.\venv\Scripts\ruff.exe check src/ tests/test_isolated_ledger_safety.py tests/test_signal_recovery_regression.py tests/test_causality_and_execution_timing.py tests/test_signal_integrity.py tests/test_execution_cost_governance.py

# Output:
All checks passed!

# 4. Check git status to verify zero committed, pushed, or modified baseline files
git status --short
```

### Git Working Tree Audit
- `backend/artifacts/frozen_strategy_manifest_v2.4.json`: **100% UNTOUCHED (0 diff lines)**. SHA-256 remains `d580b50a441c2f5a3881bb72eb0c8b23d90077dea61ddc506193affcfe993da5`.
- Working Tree Modifications: Exclusively confined to uncommitted working tree of branch `research/v2.3_phase_3a_remediation`.
- Commits / Merges / Pushes: **0 Git operations executed**.

---

## 6. Final Acceptance Assessment & Integration Recommendation

### Blocker-by-Blocker Verdicts
1. **Blocker A (Legacy Authorization Exception & Manifest Byte Parsing):** **PASS**  
   Strict 64-char hex SHA-256 digest required; missing, empty, malformed, or mismatched digests strictly fail closed; manifest parsed from exact raw bytes buffer via `json.loads(raw_bytes.decode('utf-8'))`. Tested by 19 negative and positive test cases in `test_14`.
2. **Blocker B (Dry-Run Isolation):** **PASS**  
   Verbatim `replay_research_candidate_dry_run()` and transitive call tree verified. 100% in-memory calculation with zero SQLite writes, zero prospective ledger modifications, and zero order routing. Verified by spy unit test `test_16`.
3. **Blocker C (Artifact Provenance):** **PASS**  
   Direct deserialization from raw byte buffer via `io.BytesIO(raw_bytes)` and `bytearray(raw_bytes)` across all loading routes. Cryptographic `_artifact_hash` bound at load time. Unbound objects fail closed without disk fallback. Verified by `test_15`.
4. **Blocker D (Authoritative Governance Thresholds):** **PASS (Governance Enforcement) / FAIL (Strategy Readiness)**  
   The authoritative requirements (Brier $\le 0.20$, ECE $\le 0.08$, $\ge 50$ closed trades target 100, SPY SMA200 regime coverage) are strictly enforced and verified by `test_17`. Downgraded 30-trade threshold is rejected.
5. **Finding 1 (Candidate Status Validation):** **PASS**  
   Strict allowlist check (`permitted_candidate_statuses`) and explicit prohibition check (`PROHIBITED_CANDIDATE_STATUS_TERMS`) replaces substring check. Fails closed on `REVOKED_CANDIDATE`, `SUSPENDED_CANDIDATE`, `REJECTED_CANDIDATE`, `UNAUTHORIZED_CANDIDATE`, empty strings, and unauthorized states. Verified by `test_18`.
6. **Finding 2 (Artifact Fallback Resolution & Isolation):** **PASS**  
   Multi-root directory search (`manifest_path.parent`, `artifacts/`, `backend/artifacts/`) resolves consistently across working directories. Contaminated in-memory objects are safely discarded. Fails closed with explicit error codes (`MISSING_ARTIFACTS`, `CRYPTOGRAPHIC_DIGEST_MISMATCH`, `DIMENSION_MISMATCH`, `POLICY_DIMENSION_MISMATCH`). Tested by `test_20`.
7. **Finding 3 (Candidate-Version Mapping Policy):** **PASS**  
   Strict check against `SUPPORTED_RESEARCH_CANDIDATE_VERSIONS`. Fails closed with `UNSUPPORTED_RESEARCH_CANDIDATE_VERSION` if immutable baseline `Institutional_Mesh_V2.4`, `Institutional_Mesh_V3`, or unrecognized versions are evaluated. Strict label check fails closed with `VERSION_LABEL_MISMATCH` if candidate strategy version is altered. Tested by `test_19`.
8. **Finding 4 (Exit Decision Temporal Causality & Open[t+1] Invariance):** **PASS**  
   Exit decisions at bar $t$ are evaluated strictly on information finalized by bar $t$ close (`close_t <= entry_price * 0.95 or close_t >= entry_price * 1.10 or p_sell >= 0.50`). Next-bar open $Open[t+1]$ is strictly quarantined to fill price computation. Regression unit test `test_21` in `test_isolated_ledger_safety.py` proves that extreme future open gaps (-75% crash or +100% surge) cannot trigger look-ahead exit signals, and that genuine bar $t$ exits are executed at next open with identical timestamp and invariant decision status.

### Explicit Integration Recommendation
**RECOMMENDATION: NOT APPROVED FOR INTEGRATION / MERGE / COMMIT / PUSH TO MAIN.**

- **Rationale:** While the software architecture, cryptographic bindings, dry-run isolation, candidate status governance, artifact resolution logic, and causal execution timing have achieved institutional-grade hardening and 100% pass rates across all 40 automated tests and Ruff linting, **software safety does not constitute statistical validity**.
- The quantitative strategy remains statistically uncalibrated (Brier 0.5394 vs max 0.20, ECE 0.2076 vs max 0.08) and prospectively unproven (0 closed trades vs minimum 50, target 100).
- In accordance with HYDRA Governance Mandates:
  - V2.4.1 candidate status remains `PROPOSED_CANDIDATE_PENDING_AUTHORIZATION`.
  - Live capital routing remains **STRICTLY BLOCKED**.
  - All modifications remain strictly in the uncommitted working tree.
  - Baseline `backend/artifacts/frozen_strategy_manifest_v2.4.json` remains completely untouched (0 diff lines).
