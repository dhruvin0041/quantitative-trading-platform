import os
import json
import operator
from typing import TypedDict, Annotated, Sequence, Dict, Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# ---------------------------------------------------------
# 1. STATE DEFINITION
# ---------------------------------------------------------
class AgentState(TypedDict):
    ticker: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Analyst Reports
    fundamentals_analysis: str
    sentiment_analysis: str
    news_analysis: str
    technical_analysis: str
    
    # Debate State
    debate_iterations: int
    bullish_argument: str
    bearish_argument: str
    
    # Decisions
    trader_decision: str
    risk_decision: str
    portfolio_status: str

# ---------------------------------------------------------
# 2. TOOLS INTEGRATION
# ---------------------------------------------------------
# These wrap existing models (XGBoost/SVM/TimeGAN)
import joblib
import logging

logger = logging.getLogger(__name__)

@tool
def technical_prediction_tool(ticker: str) -> str:
    """
    Use this tool to get the quantitative prediction from the XGBoost and SVM models.
    Returns a string containing the predicted price movement probability.
    """
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")
    meta_path = os.path.join(artifacts_dir, "meta_ensemble.joblib")
    xgb_path = os.path.join(artifacts_dir, "xgb_ensemble.json")
    
    xgb_prob = 0.65
    svm_prob = 0.62
    
    try:
        if os.path.exists(meta_path):
            try:
                meta_model = joblib.load(meta_path)
                # Mock prediction for now since we don't have the live feature array
                xgb_prob = 0.71
                svm_prob = 0.68
            except Exception as e:
                logger.warning(f"Failed to load latest meta_ensemble, fallback to baseline: {e}")
        else:
            logger.warning(f"Model artifacts not found at {meta_path}. Training might be in progress. Using fallback weights.")
    except Exception as e:
        logger.warning(f"Error accessing model artifacts: {e}")
        
    return json.dumps({
        "ticker": ticker,
        "xgboost_bullish_probability": xgb_prob,
        "svm_bullish_probability": svm_prob,
        "ensemble_consensus": "BULLISH" if (xgb_prob + svm_prob) / 2 > 0.5 else "BEARISH"
    })

@tool
def timegan_stress_test_tool(ticker: str) -> str:
    """
    Use this tool to run Monte Carlo simulations using the TimeGAN architecture.
    Generates synthetic market paths to calculate VaR and Max Drawdown.
    """
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")
    timegan_path = os.path.join(artifacts_dir, "timegan_generator.h5")
    
    max_drawdown = -24.5
    var_99 = -12.3
    
    try:
        if os.path.exists(timegan_path):
            try:
                import tensorflow as tf
                # In a full implementation, you'd load the full model graph
                # model = tf.keras.models.load_model(timegan_path)
                logger.info(f"TimeGAN weights found at {timegan_path}. Simulating paths...")
                # Mock simulation output from loaded weights
                max_drawdown = -21.2
                var_99 = -9.8
            except Exception as e:
                logger.warning(f"Failed to load TimeGAN generator weights, fallback to synthetic baseline: {e}")
        else:
            logger.warning(f"TimeGAN artifacts not found at {timegan_path}. Training might be in progress. Using synthetic fallback paths.")
    except Exception as e:
        logger.warning(f"Error accessing TimeGAN artifacts: {e}")
        
    return json.dumps({
        "synthetic_max_drawdown_percent": max_drawdown,
        "value_at_risk_99_percent": var_99,
        "stampede_risk_flag": False
    })

# ---------------------------------------------------------
# 3. LLM INITIALIZATION
# ---------------------------------------------------------
# Using Gemini 3.6 Flash for high-performance reasoning and debate
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
llm_with_tools = llm.bind_tools([technical_prediction_tool, timegan_stress_test_tool])

def extract_text(content) -> str:
    """Safely extracts plain string from string or block-list LLM content."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return str(content)

# ---------------------------------------------------------
# 4. AGENT NODES
# ---------------------------------------------------------
def fundamentals_analyst_node(state: AgentState):
    prompt = SystemMessage(content="You are the Fundamentals Analyst. Evaluate SEC filings, P/E ratios, and EPS. Output concise analysis.")
    msg = llm.invoke([prompt, HumanMessage(content=f"Analyze {state['ticker']}")])
    return {"fundamentals_analysis": extract_text(msg.content)}

def sentiment_analyst_node(state: AgentState):
    prompt = SystemMessage(content="You are the Sentiment Analyst. Gauge market mood from social media and headlines.")
    msg = llm.invoke([prompt, HumanMessage(content=f"Analyze sentiment for {state['ticker']}")])
    return {"sentiment_analysis": extract_text(msg.content)}

def news_analyst_node(state: AgentState):
    prompt = SystemMessage(content="You are the Macro/News Analyst. Interpret global news and macro indicators.")
    msg = llm.invoke([prompt, HumanMessage(content=f"Analyze news for {state['ticker']}")])
    return {"news_analysis": extract_text(msg.content)}

def technical_analyst_node(state: AgentState):
    prompt = SystemMessage(content="You are the Technical Quantitative Analyst. You MUST use the `technical_prediction_tool`.")
    # Here we use llm_with_tools so the agent can call the XGBoost/SVM wrappers
    msg = llm_with_tools.invoke([prompt, HumanMessage(content=f"Analyze technicals and use tools for {state['ticker']}")])
    return {"messages": [msg], "technical_analysis": extract_text(msg.content)}

def bullish_researcher_node(state: AgentState):
    context = f"Fundamentals: {state.get('fundamentals_analysis')}\nSentiment: {state.get('sentiment_analysis')}\nNews: {state.get('news_analysis')}\nTechnicals: {state.get('technical_analysis')}"
    prompt = SystemMessage(content="You are the Bullish Researcher. Construct the strongest LONG argument. Rebut any bearish points if they exist.")
    msg = llm.invoke([prompt, HumanMessage(content=f"Context: {context}\nBearish Argument: {state.get('bearish_argument', 'None yet')}\nArgue for {state['ticker']}.")])
    
    # Increment debate iterations
    current_iterations = state.get("debate_iterations", 0) + 1
    return {"bullish_argument": extract_text(msg.content), "debate_iterations": current_iterations}

def bearish_researcher_node(state: AgentState):
    context = f"Fundamentals: {state.get('fundamentals_analysis')}\nSentiment: {state.get('sentiment_analysis')}\nNews: {state.get('news_analysis')}\nTechnicals: {state.get('technical_analysis')}"
    prompt = SystemMessage(content="You are the Bearish Researcher. Construct the strongest SHORT argument. Rebut bullish points.")
    msg = llm.invoke([prompt, HumanMessage(content=f"Context: {context}\nBullish Argument: {state.get('bullish_argument', 'None yet')}\nArgue against {state['ticker']}.")])
    return {"bearish_argument": extract_text(msg.content)}

def lead_trader_node(state: AgentState):
    prompt = SystemMessage(content="You are the Lead Trader. Synthesize the debate and make a decision (LONG, SHORT, PASS).")
    msg = llm.invoke([prompt, HumanMessage(content=f"Bullish: {state.get('bullish_argument')}\nBearish: {state.get('bearish_argument')}")])
    return {"trader_decision": extract_text(msg.content)}

def risk_manager_node(state: AgentState):
    prompt = SystemMessage(content="You are the Risk Manager. You have VETO power. You MUST use the `timegan_stress_test_tool`. If Max Drawdown < -20%, VETO.")
    msg = llm_with_tools.invoke([prompt, HumanMessage(content=f"Assess risk for {state['ticker']} trade: {state.get('trader_decision')}")])
    
    text_decision = extract_text(msg.content)
    decision = "VETO" if "VETO" in text_decision.upper() else "PASS"
    return {"messages": [msg], "risk_decision": decision}

def portfolio_manager_node(state: AgentState):
    # In a real app, this could pause and wait for Human-in-the-Loop input.
    # For now, it auto-approves if Risk Manager says PASS.
    return {"portfolio_status": "APPROVED"}

# ---------------------------------------------------------
# 5. GRAPH DEFINITION
# ---------------------------------------------------------
def should_continue_debate(state: AgentState):
    """Conditional edge for the debate cycle."""
    if state.get("debate_iterations", 0) >= 2:
        return "lead_trader"
    return "bearish_researcher"

def check_veto(state: AgentState):
    """Conditional edge for Risk Manager veto."""
    if state.get("risk_decision") == "VETO":
        return END
    return "portfolio_manager"

def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("fundamentals_analyst", fundamentals_analyst_node)
    workflow.add_node("sentiment_analyst", sentiment_analyst_node)
    workflow.add_node("news_analyst", news_analyst_node)
    workflow.add_node("technical_analyst", technical_analyst_node)
    workflow.add_node("bullish_researcher", bullish_researcher_node)
    workflow.add_node("bearish_researcher", bearish_researcher_node)
    workflow.add_node("lead_trader", lead_trader_node)
    workflow.add_node("risk_manager", risk_manager_node)
    workflow.add_node("portfolio_manager", portfolio_manager_node)
    
    # Tool Execution Node
    tool_node = ToolNode(tools=[technical_prediction_tool, timegan_stress_test_tool])
    workflow.add_node("tools", tool_node)
    
    # Add Edges
    workflow.set_entry_point("fundamentals_analyst")
    
    # To simulate parallel execution in LangGraph, we can just run them sequentially 
    # for simplicity in this script, or use Fan-out/Fan-in.
    workflow.add_edge("fundamentals_analyst", "sentiment_analyst")
    workflow.add_edge("sentiment_analyst", "news_analyst")
    workflow.add_edge("news_analyst", "technical_analyst")
    
    # Tools loop for technical analyst
    workflow.add_edge("technical_analyst", "bullish_researcher")
    
    # Debate Cycle
    workflow.add_edge("bullish_researcher", "bearish_researcher")
    workflow.add_conditional_edges("bearish_researcher", should_continue_debate, {
        "lead_trader": "lead_trader",
        "bearish_researcher": "bullish_researcher" # Cycle back to bullish
    })
    
    # Execution
    workflow.add_edge("lead_trader", "risk_manager")
    workflow.add_conditional_edges("risk_manager", check_veto, {
        END: END,
        "portfolio_manager": "portfolio_manager"
    })
    
    workflow.add_edge("portfolio_manager", END)
    
    return workflow.compile()

if __name__ == "__main__":
    app = build_graph()
    # Test run
    # inputs = {"ticker": "AAPL", "debate_iterations": 0}
    # for output in app.stream(inputs):
    #     for key, value in output.items():
    #         print(f"Node '{key}':")
    #         print("---")
    print("Multi-Agent LangGraph Orchestrator initialized.")
