#!/usr/bin/env bash
# ==============================================================================
# Manual On-Demand Daily Execution Runner (POSIX)
#
# Strictly manual and on-demand execution. No background daemons or loops.
# Executes a single daily execution cycle and terminates immediately.
#
# Usage:
#   ./run_daily_eod.sh              (Live paper execution with Pure XGBoost flagship)
#   ./run_daily_eod.sh --dry-run    (Simulated run with zero broker/db mutations)
#   ./run_daily_eod.sh --use-veto   (Enables secondary asymmetric veto consensus)
#   ./run_daily_eod.sh --status     (Read-only status dashboard: balance, positions, stops)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${BACKEND_DIR}"

if [ -f "venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "venv/bin/activate"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Manual On-Demand Daily Execution Run..."
python execution/paper_runner.py "$@"
EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Manual On-Demand Daily Execution Run completed successfully."
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Daily Execution Run failed with exit code ${EXIT_CODE}."
fi

exit ${EXIT_CODE}
