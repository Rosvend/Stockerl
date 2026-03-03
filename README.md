# Stockerl

A reinforcement learning agent that learns to trade Amazon (AMZN) stock using **Proximal Policy Optimization (PPO)** with a continuous action space. Built with a custom **Gymnasium** environment and trained via **Stable Baselines3**.

---

## Project Structure

```
src/
├── main.py              # Training and evaluation pipeline (entry point)
├── environment.py       # Custom Gymnasium trading environment
├── train.py             # PPO model configuration and training loop
├── evaluate.py          # Agent evaluation and plotting
├── metrics.py           # Financial performance metrics (Sharpe, Sortino, etc.)
└── data/
    └── market_data.py   # Stock data fetching via yfinance
```

---

## How It Works

### 1. Data Pipeline

Historical AMZN daily data (2020-01-01 to 2024-12-31) is fetched from Yahoo Finance using `yfinance`. The data is split 80/20 into training and test sets.

### 2. Environment (`environment.py`)

A custom Gymnasium environment simulates a single-stock trading account.

**Observation space** — a vector of `window_size + 4` features:

| Feature | Description |
|---|---|
| Log returns | `window_size - 1` consecutive log price returns |
| RSI | Relative Strength Index, normalized to [0, 1] |
| MACD signal | MACD vs signal line direction (-1, 0, +1) |
| Relative volume | Current volume / 20-day average volume |
| Portfolio ratio | Invested value / initial balance |
| Cash ratio | Cash / initial balance |

**Action space** — continuous, $a \in [-1, 1]$:

| Range | Action |
|---|---|
| $a > 0.01$ | **Buy** — use fraction $a$ of available cash |
| $-0.01 \leq a \leq 0.01$ | **Hold** — no transaction |
| $a < -0.01$ | **Sell** — sell fraction $|a|$ of held shares |

**Reward** — logarithmic return of total portfolio value:

$$r_t = \ln\left(\frac{V_t}{V_{t-1}}\right)$$

Log returns are time-additive and symmetric, which provides a stable training signal for PPO.

**Constraints:**
- Transaction fee of 0.1% per trade
- No short selling
- Bankruptcy terminates the episode (portfolio < 10% of initial balance)
- Random episode start points during training for better generalization

### 3. Training (`train.py`, `main.py`)

The agent is trained with PPO using the following configuration:

| Parameter | Value |
|---|---|
| Total timesteps | 500,000 |
| Learning rate | 3e-4 (linear decay to 0) |
| Discount factor (gamma) | 0.96 |
| Clip range | 0.15 |
| Entropy coefficient | 0.01 |
| Target KL | 0.02 |
| Batch size | 128 |
| Steps per update | 2,048 |

Key training features:
- **VecNormalize** wraps the environment to normalize observations and rewards
- **EvalCallback** evaluates on the test set every 10,000 steps and saves the best model
- **NormSyncCallback** syncs normalization statistics from the training env to the eval env
- **TensorBoard** logging for portfolio value and returns

### 4. Evaluation (`evaluate.py`, `metrics.py`)

The trained agent is evaluated over 5 episodes on the held-out test set and compared against a **fee-adjusted buy-and-hold baseline**. Evaluation uses `random_start=False` to cover the full test period for a fair comparison.

Metrics reported:
- **Cumulative Return** — total portfolio growth
- **Sharpe Ratio** — risk-adjusted return (annualized)
- **Max Drawdown** — worst peak-to-trough decline
- **Number of Trades** — trading activity level

A plot comparing the agent's portfolio value vs. buy-and-hold is saved to `trading_results.png`.

---

## Getting Started

### Prerequisites

- Python >= 3.11

### Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install gymnasium stable-baselines3 yfinance numpy pandas matplotlib tensorboard tqdm rich
```

### Running

```bash
cd src
python main.py
```

This will:
1. Download AMZN historical data
2. Train the PPO agent for 500k timesteps
3. Evaluate on the test set
4. Print a summary comparing agent vs. buy-and-hold returns
5. Save a results plot to `trading_results.png`

### Monitoring Training

```bash
tensorboard --logdir logs/ppo_trading
```

---

## Tech Stack

- **Language:** Python 3.11+
- **RL Framework:** Stable Baselines3 (PPO)
- **Environment:** Gymnasium
- **Data Source:** yfinance
- **Analysis:** NumPy, Pandas, Matplotlib, TensorBoard
