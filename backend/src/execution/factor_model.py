import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class FactorModel:
    """
    Cross-Sectional Factor Modeling to decompose returns into 
    systematic risk factors vs idiosyncratic alpha.
    """
    def __init__(self, n_factors=3):
        self.n_factors = n_factors
        self.pca = PCA(n_components=self.n_factors)
        
    def fit_statistical_factors(self, returns_df: pd.DataFrame):
        """
        Fits PCA on a panel of asset returns to extract statistical factors.
        returns_df: columns are tickers, rows are dates.
        """
        # Drop rows with NaN to ensure clean PCA
        clean_returns = returns_df.dropna()
        if len(clean_returns) < 10:
            return None
            
        factor_returns = self.pca.fit_transform(clean_returns)
        self.factor_loadings = self.pca.components_
        self.explained_variance = self.pca.explained_variance_ratio_
        
        return factor_returns
        
    def calculate_idiosyncratic_risk(self, ticker, ticker_returns, factor_returns):
        """
        Calculates the specific risk (volatility not explained by factors).
        """
        if factor_returns is None or len(ticker_returns) != len(factor_returns):
            return np.std(ticker_returns)
            
        # Regress ticker returns on factor returns
        X = np.column_stack((np.ones(len(factor_returns)), factor_returns))
        y = ticker_returns.values
        
        # OLS estimation: beta = (X^T X)^-1 X^T y
        try:
            beta = np.linalg.inv(X.T @ X) @ X.T @ y
            predictions = X @ beta
            residuals = y - predictions
            idiosyncratic_vol = np.std(residuals)
            return float(idiosyncratic_vol)
        except Exception:
            return float(np.std(y))
