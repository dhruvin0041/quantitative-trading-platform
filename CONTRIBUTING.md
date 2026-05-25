# CONTRIBUTING TO HYDRA TERMINAL

Welcome to the Hydra research team. To maintain the SOTA 2026 standards of this quantitative system, all contributors must adhere to the following protocols.

## 🧠 1. The "Zero-State" Workflow
Hydra enforces a strict separation of research contexts to prevent neural pathway contamination.

**MANDATORY**: Before switching from research on one ticker to another, you must run:
```bash
python backend/scripts/ops/clean_artifacts.py
```
This script systematically wipes transient model weights and scalers, ensuring the next research session starts from a clean baseline.

## 🛠 2. Coding Standards

### Python (Backend)
- **Style**: Strict PEP 8 enforced via `ruff`.
- **Typing**: Use Pydantic models for all data exchange.
- **Async**: Prefer `async`/`await` for all I/O bound tasks.
- **Safety**: Never use bare `except:`. Always log exceptions with full context.
- **Verification**: Run `ruff check .` before every commit.

### TypeScript (Frontend)
- **Type Safety**: Zero `any` policy. All components and API responses must be strictly typed.
- **Architecture**: Follow the Next.js App Router conventions and Server Components by default.
- **Styling**: Adhere to the **Intrinsic Sizing Model** in Tailwind. Avoid hardcoding pixel heights for containers.

## 📂 3. Repository Discipline

### Git Protocol
- **Single-Branch**: Work exclusively on the `main` branch unless a long-running feature requires a dedicated branch.
- **Commits**: Use **Conventional Commits** (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`).
- **No AI Attribution**: Do not include "Co-authored-by: Claude" or similar markers. All contributions appear as authored by the user.

### Exclusion Policy
- **Local Assets**: NEVER commit `graphify-out/` or `GEMINI.md`. These are local-only assets.
- **ML Artifacts**: Ensure `.gitignore` properly excludes large weight files (`.h5`, `.pth`) and local databases (`mlflow.db`).

## 🧪 4. Testing & Validation
- **Alpha Research**: New features must be validated via Walk-Forward Optimization (WFO).
- **Integrity Audit**: After modifying ingestion or feature engineering, run the `IntegrityAudit` component in the frontend to verify data provenance.
- **Graph Consistency**: Run `graphify update .` after significant structural changes to keep the system dependency map accurate.

---
**Hydra Terminal** — built for precision, grounded in data, executed by agents.
