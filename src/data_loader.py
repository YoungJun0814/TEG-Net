import yfinance as yf
import pandas as pd
import numpy as np

def get_vix_term_structure():
    """
    Downloads and preprocesses US Market Data for VIX Term Structure analysis.
    
    This function fetches volatility indices for different maturities to construct
    the 'spatial' dimension of the volatility surface and the SKEW index as a 
    proxy for tail-risk sentiment.
    
    Returns:
        pd.DataFrame: A dataframe containing VIX term structure (1M, 3M, 6M) 
                      and the SKEW index.
    """
    print(">>> [Data] Downloading US Market Data (VIX, VIX3M, VIX6M, SKEW)...")
    
    # Define Tickers
    # VIX (Spot), VIX3M (3-Month), VIX6M (6-Month) form the Term Structure.
    # SKEW is used as the external trigger for the 'Reaction' term in the PDE.
    tickers = {
        '^VIX': 'VIX_1M',   
        '^VIX3M': 'VIX_3M', 
        '^VIX6M': 'VIX_6M', 
        '^SKEW': 'SKEW'     
    }
    
    # Download Data (Fetching last 5-10 years for robust training)
    try:
        # Note: yfinance download format may vary by version. 
        # Using a list of tickers usually returns a MultiIndex or simple DataFrame.
        df = yf.download(list(tickers.keys()), start="2015-01-01", progress=False)['Close']
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None
    
    # Rename columns to standard internal names
    # Handling potential column mismatch from yfinance
    df = df.rename(columns=tickers)
    
    # Data Cleaning: Handle missing values via forward/backward filling
    df = df.ffill().bfill().dropna()
    
    # Ensure column order for spatial calculation (Gradient/Laplacian in Physics Loss)
    # Order: Short-term -> Medium-term -> Long-term -> External Trigger
    target_cols = ['VIX_1M', 'VIX_3M', 'VIX_6M', 'SKEW']
    df = df[target_cols]
    
    print(f">>> Download Complete. Shape: {df.shape}")
    print(">>> Sample Data:\n", df.tail())
    return df