"""
Trading Performance Metrics

This module provides functions for calculating various trading performance metrics
commonly used in quantitative finance to evaluate trading strategies.
"""

import numpy as np
from typing import Union, List


def sharpe_ratio(
    returns: Union[np.ndarray, List[float]],
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252
) -> float:
    """
    Calculate the annualized Sharpe ratio.
    
    The Sharpe ratio measures the excess return per unit of risk (volatility).
    Higher values indicate better risk-adjusted returns.
    
    Parameters:
    -----------
    returns : array-like
        Array of periodic returns (e.g., daily returns)
    risk_free_rate : float
        The risk-free rate (annualized), default 0
    annualization_factor : float
        Factor to annualize returns (252 for daily, 52 for weekly, 12 for monthly)
    
    Returns:
    --------
    float : Annualized Sharpe ratio
    """
    returns = np.array(returns)
    
    if len(returns) < 2:
        return 0.0
    
    # Convert annual risk-free rate to periodic rate
    periodic_rf = risk_free_rate / annualization_factor
    
    excess_returns = returns - periodic_rf
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(annualization_factor)


def max_drawdown(portfolio_values: Union[np.ndarray, List[float]]) -> float:
    """
    Calculate the maximum drawdown from a series of portfolio values.
    
    Maximum drawdown measures the largest peak-to-trough decline,
    representing the worst-case loss scenario.
    
    Parameters:
    -----------
    portfolio_values : array-like
        Array of portfolio values over time
    
    Returns:
    --------
    float : Maximum drawdown as a decimal (e.g., 0.20 = 20% drawdown)
    """
    portfolio_values = np.array(portfolio_values)
    
    if len(portfolio_values) < 2:
        return 0.0
    
    # Calculate running maximum (peak)
    peak = np.maximum.accumulate(portfolio_values)
    
    # Calculate drawdown at each point
    drawdown = (peak - portfolio_values) / peak
    
    return np.max(drawdown)


def cumulative_return(portfolio_values: Union[np.ndarray, List[float]]) -> float:
    """
    Calculate the total cumulative return of a portfolio.
    
    Parameters:
    -----------
    portfolio_values : array-like
        Array of portfolio values over time
    
    Returns:
    --------
    float : Cumulative return as a decimal (e.g., 0.50 = 50% return)
    """
    portfolio_values = np.array(portfolio_values)
    
    if len(portfolio_values) < 2 or portfolio_values[0] == 0:
        return 0.0
    
    return (portfolio_values[-1] / portfolio_values[0]) - 1


def sortino_ratio(
    returns: Union[np.ndarray, List[float]],
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252
) -> float:
    """
    Calculate the annualized Sortino ratio.
    
    Similar to Sharpe ratio but only considers downside volatility,
    making it more appropriate for asymmetric return distributions.
    
    Parameters:
    -----------
    returns : array-like
        Array of periodic returns
    risk_free_rate : float
        The risk-free rate (annualized)
    annualization_factor : float
        Factor to annualize returns
    
    Returns:
    --------
    float : Annualized Sortino ratio
    """
    returns = np.array(returns)
    
    if len(returns) < 2:
        return 0.0
    
    periodic_rf = risk_free_rate / annualization_factor
    excess_returns = returns - periodic_rf
    
    # Only consider negative returns for downside deviation
    downside_returns = np.minimum(excess_returns, 0)
    downside_deviation = np.std(downside_returns)
    
    if downside_deviation == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0
    
    return np.mean(excess_returns) / downside_deviation * np.sqrt(annualization_factor)


def calmar_ratio(
    portfolio_values: Union[np.ndarray, List[float]],
    annualization_factor: float = 252
) -> float:
    """
    Calculate the Calmar ratio.
    
    The Calmar ratio is the annualized return divided by the maximum drawdown.
    It measures the trade-off between returns and maximum loss.
    
    Parameters:
    -----------
    portfolio_values : array-like
        Array of portfolio values over time
    annualization_factor : float
        Number of periods in a year
    
    Returns:
    --------
    float : Calmar ratio
    """
    portfolio_values = np.array(portfolio_values)
    
    if len(portfolio_values) < 2:
        return 0.0
    
    # Calculate annualized return
    total_return = cumulative_return(portfolio_values)
    num_periods = len(portfolio_values) - 1
    annualized_return = (1 + total_return) ** (annualization_factor / num_periods) - 1
    
    # Calculate max drawdown
    mdd = max_drawdown(portfolio_values)
    
    if mdd == 0:
        return np.inf if annualized_return > 0 else 0.0
    
    return annualized_return / mdd


def win_rate(trades: List[dict]) -> float:
    """
    Calculate the win rate (percentage of profitable trades).
    
    Parameters:
    -----------
    trades : list of dict
        List of trade dictionaries with 'profit' key
    
    Returns:
    --------
    float : Win rate as a decimal (e.g., 0.60 = 60% win rate)
    """
    if not trades:
        return 0.0
    
    profitable_trades = sum(1 for t in trades if t.get("profit", 0) > 0)
    return profitable_trades / len(trades)


def profit_factor(trades: List[dict]) -> float:
    """
    Calculate the profit factor (gross profit / gross loss).
    
    A profit factor > 1 indicates a profitable trading system.
    
    Parameters:
    -----------
    trades : list of dict
        List of trade dictionaries with 'profit' key
    
    Returns:
    --------
    float : Profit factor
    """
    if not trades:
        return 0.0
    
    gross_profit = sum(t.get("profit", 0) for t in trades if t.get("profit", 0) > 0)
    gross_loss = abs(sum(t.get("profit", 0) for t in trades if t.get("profit", 0) < 0))
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_all_metrics(
    portfolio_values: Union[np.ndarray, List[float]],
    risk_free_rate: float = 0.0
) -> dict:
    """
    Calculate all performance metrics for a portfolio.
    
    Parameters:
    -----------
    portfolio_values : array-like
        Array of portfolio values over time
    risk_free_rate : float
        The annualized risk-free rate
    
    Returns:
    --------
    dict : Dictionary containing all calculated metrics
    """
    portfolio_values = np.array(portfolio_values)
    
    # Calculate returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    return {
        "cumulative_return": cumulative_return(portfolio_values),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate),
        "max_drawdown": max_drawdown(portfolio_values),
        "calmar_ratio": calmar_ratio(portfolio_values),
        "volatility": np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0,
        "final_value": portfolio_values[-1] if len(portfolio_values) > 0 else 0.0
    }