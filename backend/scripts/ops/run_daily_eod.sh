#!/usr/bin/env bash
# ==============================================================================
# Institutional Automated Daily Execution Runner (POSIX / Cron)
#
# Cron Schedule:
#   Mon-Fri at 16:15 EST (15 minutes after US market close: 21:15 UTC standard)
#   Cron Entry:
#   15 16 * * 1-5 /path/to/backend/scripts/ops/run_daily_eod.sh >> /path/to/backend/artifacts/cron_eod.log 2>&1
#
# Usage:
#   ./run_daily_eod.sh              (Live paper execution with Pure XGBoost flagship)
#   ./run_daily_eod.sh --dry-run    (Simulated run with zero broker/db mutations)
#   ./run_daily_eod.sh --use-veto   (Enables secondary asymmetric veto consensus)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${BACKEND_DIR}"

if [ -f "venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "venv/bin/activate"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Automated Daily EOD Execution Run..."
python execution/paper_runner.py "$@"
EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Automated Daily EOD Execution Run completed successfully."
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Daily EOD Execution Run failed with exit code ${EXIT_CODE}."
fi

exit ${EXIT_CODE}
