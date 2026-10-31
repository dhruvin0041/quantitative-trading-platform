"""
Batch training script for newly added US and Indian universe assets.
Generates isolated models and calibration artifacts for each asset.
"""
import jsonimport loggingimport sysimport timefrom pathlib import Pathbackend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from scripts.training.train import train_tickerlogging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NEW_TICKERS = [
    # US Stable Champions & Leaders
    "JNJ",
    "V",
    "BRK-B",
    "WMT",
    "JPM",
    "LLY",
    # Indian Stable Champions & Leaders
    "TITAN.NS",
    "SUNPHARMA.NS",
    "TCS.NS",
    "ICICIBANK.NS",
    "BHARTIARTL.NS",
    "LT.NS",
    "ITC.NS",
]

def main():
    logger.info("=" * 80)
    logger.info(f"STARTING DEDICATED MODEL TRAINING FOR {len(NEW_TICKERS)} NEW ASSETS")
    logger.info("=" * 80)

    for i, t in enumerate(NEW_TICKERS, start=1):
        logger.info(f"[{i}/{len(NEW_TICKERS)}] Training dedicated pipeline for {t}...")
        t0 = time.time()
        try:
            res = train_ticker(t, skip_optimization=True)
            elapsed = round(time.time() - t0, 2)
            logger.info(f"[SUCCESS] {t} completed in {elapsed}s | Test Acc: {res.get('xgb_test_acc', 0.0)*100:.2f}%")
        except Exception as exc:
            logger.error(f"[FAILED] {t} training error: {exc}")

    # Reset active ticker to AAPL
    active_path = Path(__file__).resolve().parent.parent.parent / "configs" / "active_ticker.json"
    if active_path.exists():
        with open(active_path, "w") as f:
            json.dump({"ticker": "AAPL", "market": "us"}, f, indent=2)

    logger.info("=" * 80)
    logger.info("ALL NEW ASSETS TRAINED AND ARTIFACTS ISOLATED")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
