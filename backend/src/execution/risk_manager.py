import numpy as np

def calculate_full_kelly(win_prob, win_loss_ratio):
    """
    Calculates the Full Kelly Criterion for optimal position sizing.
    Formula: K% = W - [(1 - W) / R]
    W = Win Probability
    R = Win/Loss Ratio (Avg Win / Avg Loss)
    """
    if win_loss_ratio <= 0:
        return 0.0
        
    kelly = win_prob - ((1 - win_prob) / win_loss_ratio)
    
    # Constrain between 0% and 25% for safety, even if 'Full Kelly' is requested
    # Real-world institutional desks rarely exceed 20-25% on a single ticker
    return float(np.clip(kelly, 0.0, 0.25))

def get_position_sizing(model_confidence, historical_win_rate=0.55, avg_win_loss=1.2):
    """
    Determines the suggested capital allocation using a blend of confidence and Kelly.
    """
    kelly_fraction = calculate_full_kelly(historical_win_rate, avg_win_loss)
    
    # Scale Kelly fraction by model confidence (0.0 to 1.0)
    # If the model is only 70% sure, we only take 70% of the Kelly-suggested bet
    suggested_allocation = kelly_fraction * model_confidence
    
    return {
        "kelly_basis": f"{round(kelly_fraction * 100, 1)}%",
        "suggested_allocation": f"{round(suggested_allocation * 100, 1)}%",
        "raw_fraction": suggested_allocation
    }
