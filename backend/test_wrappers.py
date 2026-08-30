from src.agents.langgraph_orchestrator import technical_prediction_tool, timegan_stress_test_tool
import json

def test_tools():
    print("Testing technical_prediction_tool...")
    tech_result = technical_prediction_tool.invoke({"ticker": "AAPL"})
    print("Result:", tech_result)
    assert "AAPL" in tech_result
    print("OK.")

    print("\nTesting timegan_stress_test_tool...")
    gan_result = timegan_stress_test_tool.invoke({"ticker": "AAPL"})
    print("Result:", gan_result)
    assert "value_at_risk_99_percent" in gan_result
    print("OK.")

if __name__ == "__main__":
    test_tools()
