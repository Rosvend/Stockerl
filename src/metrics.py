import numpy as np 

#sharpe ratio
def sharpe_ratio(returns, risk_free_rate, volatility):
    excess_returns = returns - risk_free_rate
    return excess_returns / volatility


#max drawdown
def max_drawdown(portfolio_values):
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    return np.max(drawdown)

#cumulative return
def cumulative_return(portfolio_values):
    return (portfolio_values[-1] / portfolio_values[0]) - 1

