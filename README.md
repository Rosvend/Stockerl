# Stockerl

## Project Overview

Stockerl is a **Reinforcement Learning (RL)** agent trained to execute optimal trading strategies for Amazon (AMZN) stock. The agent is built using the **Proximal Policy Optimization (PPO)** algorithm and a custom-built environment following the **Gymnasium** (formerly OpenAI Gym) standard.

The goal is to maximize the portfolio's risk-adjusted returns by dynamically managing cash reserves and stock holdings.

---

## 1. Environment Architecture

This project utilizes a **from-scratch custom Gymnasium environment**. This allows for strict adherence to the portfolio math required for academic evaluation.

### State Space (Observation)

The state $S_t$ is an 8-dimensional vector represented as:


$$S_t = \{P_t, P_{t-1}, P_{t-2}, P_{t-3}, P_{t-4}, P_{t-5}, V_{invested}, V_{cash}\}$$

* **$P_{t...t-5}$:** The closing price of AMZN for the current day and the five preceding days (Lookback window = 6).
* **$V_{invested}$:** The current market value of the shares held ($Shares \times P_t$).
* **$V_{cash}$:** The liquid capital available for new purchases.

### Action Space (Continuous)

While the initial requirement suggested discrete actions, this implementation uses a **continuous action space** $A \in [-1, 1]$ to allow for granular capital allocation.

| Action Value ($a$) | Interpretation | Logic |
| --- | --- | --- |
| $a \in [-1, 0)$ | **Sell** | Sells a percentage $ |
| $a = 0$ | **Hold** | No market transaction; portfolio value only fluctuates with $P_t$. |
| $a \in (0, 1]$ | **Buy** | Uses a percentage $a$ of available $V_{cash}$ to purchase shares. |

---

## 2. Reward Function

To ensure training stability and account for compounding growth, the reward $r_t$ is calculated using the **logarithmic return** of the Total Net Worth ($V_{total}$).

$$V_{total, t} = V_{invested, t} + V_{cash, t}$$

$$r_t = \ln\left(\frac{V_{total, t}}{V_{total, t-1}}\right)$$

**Rationale:** Log returns are time-additive and more symmetric than raw percentage changes, preventing the PPO agent from being biased toward high-price regimes.

---

## 3. Trading Logic & Constraints

* **Transaction Execution:** All trades are executed at the current step's closing price.
* **Portfolio Update:** The $V_{invested}$ value is updated at every step regardless of the action, reflecting the real-time market volatility of the held assets.
* **Capital Constraints:** The agent cannot spend more than $V_{cash}$ and cannot sell more shares than it currently owns (no short-selling in this version).

---

## 4. Evaluation Metrics

The agent is evaluated against a **Buy and Hold (B&H)** baseline on out-of-sample test data using the following metrics:

1. **Cumulative Return:** Total percentage growth of the portfolio.
2. **Sharpe Ratio:** $\frac{R_p - R_f}{\sigma_p}$ (Risk-adjusted performance).
3. **Max Drawdown:** The maximum observed loss from a peak to a trough.
4. **Action Distribution:** Analysis of how often the agent chooses to be aggressive vs. conservative.

---

## 5. Tech stack

* **Language:** Python 3.x
* **RL Framework:** Stable Baselines3 (PPO implementation)
* **Environment:** Gymnasium
* **Data Source:** `yfinance` (Amazon historical daily data)
* **Analysis:** NumPy, Pandas, Matplotlib

---
