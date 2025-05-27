import numpy as np
import pandas as pd

def calculate_cagr(df):
    years = (df.index[-1] - df.index[0]).days / 365.25
    cumulative_return = df.iloc[-1] / df.iloc[0]
    return (cumulative_return ** (1 / years)) - 1

def calculate_volatility(df):
    returns = df.pct_change().dropna()
    return returns.std() * np.sqrt(252)

def calculate_sharpe_ratio(df, risk_free_rate=0.02):
    returns = df.pct_change().dropna()
    excess_return = returns.mean() - (risk_free_rate / 252)
    return (excess_return / returns.std()) * np.sqrt(252)

def calculate_max_drawdown(df):
    cumulative = (1 + df.pct_change()).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

