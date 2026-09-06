import logging
import os
import sys

# Ensure absolute imports work when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agents.orchestrator import InstitutionalOrchestrator
from src.models.generative.timegan import MarketTimeGAN

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def run_stress_test():
    logger.info("Initializing Generative Market TimeGAN Stress Test...")
    timegan = MarketTimeGAN(seq_len=10, num_features=5, latent_dim=12)
    orchestrator = InstitutionalOrchestrator()

    num_assets = 500

    logger.info("--- TEST PHASE 1: STANDARD MARKET CONDITIONS ---")
    standard_panel = timegan.generate_synthetic_data(num_assets=num_assets)

    # Mocking AlphaAgent output: Under standard conditions, AI generates 40 buy signals randomly
    standard_agreement_data = []
    for ticker in standard_panel.index[:40]:
        standard_agreement_data.append({
            "ticker": ticker,
            "dominant_idx": 2, # BUY
            "agreement_score": 75.0 # 75% confidence
        })

    result_standard = orchestrator.run_panel_consensus(
        panel_agreement_data=standard_agreement_data,
        panel_data=standard_panel,
        market_regime="BULL"
    )

    logger.info(f"Standard Market Consensus Status: {result_standard['consensus_status']}")
    logger.info(f"Approved Trades: {result_standard['approved_trades_count']} / 40")
    logger.info(f"Portfolio VaR: {result_standard['portfolio_var']*100:.2f}%")
    print("\n")


    logger.info("--- TEST PHASE 2: BLACK SWAN INJECTION (MARKET CRASH) ---")
    logger.info("Injecting massive volatility spikes and 1.0 cross-asset correlation breakdown...")
    black_swan_panel = timegan.inject_black_swan(num_assets=num_assets)

    # Mocking AlphaAgent output: The AI attempts to "buy the dip" heavily due to massive price drops,
    # issuing Buy signals across the board.
    catastrophic_agreement_data = []
    for ticker in black_swan_panel.index[:200]: # AI attempts to buy 200 crashing stocks
        catastrophic_agreement_data.append({
            "ticker": ticker,
            "dominant_idx": 2, # BUY
            "agreement_score": 85.0 # High confidence to buy the dip
        })

    result_black_swan = orchestrator.run_panel_consensus(
        panel_agreement_data=catastrophic_agreement_data,
        panel_data=black_swan_panel,
        market_regime="BEAR"
    )

    logger.info(f"Black Swan Consensus Status: {result_black_swan['consensus_status']}")
    logger.info(f"Approved Trades: {result_black_swan['approved_trades_count']} / 200")
    logger.info(f"Veto Reason: {result_black_swan.get('rejected_trades')[0].get('veto_reason') if result_black_swan['consensus_status'] == 'VETOED' else 'None'}")

    # Validate the mathematical safeguard
    if result_black_swan['consensus_status'] == "VETOED":
        logger.info("SUCCESS: RiskAgent mathematically detected the Black Swan and successfully severed execution.")
    else:
        logger.error("FAILURE: RiskAgent allowed trades during a catastrophic market breakdown.")


if __name__ == "__main__":
    run_stress_test()
