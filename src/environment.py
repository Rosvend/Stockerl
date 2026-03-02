import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TradingEnv(gym.Env):
    """
    A Gymnasium-compatible trading environment for reinforcement learning.

    State: [log_returns (window_size-1), rsi, macd_signal, relative_volume,
            portfolio_ratio, cash_ratio]

    Action: Continuous [-1, 1]
        - -1: Sell all shares
        -  0: Hold
        - +1: Buy with all available cash
        - Values in between: proportional transactions

    Reward: Log returns of portfolio value
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        initial_balance: float = 1_000_000,
        window_size: int = 10,
        transaction_fee: float = 0.001,
        hold_penalty: float = 0.0,          # default → 0
        bankruptcy_threshold: float = 0.1,
        random_start: bool = False,          # random start support
        render_mode: str = None,
    ):
        super().__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.transaction_fee = transaction_fee
        self.hold_penalty = hold_penalty
        self.bankruptcy_threshold = bankruptcy_threshold
        self.random_start = random_start
        self.render_mode = render_mode

        # Extract closing prices
        if "Close" in df.columns:
            self.prices = df["Close"].values.flatten()
        elif "Adj Close" in df.columns:
            self.prices = df["Adj Close"].values.flatten()
        else:
            self.prices = df.iloc[:, 0].values.flatten()

        # Extract volume data (for relative volume indicator)
        if "Volume" in df.columns:
            self.volumes = df["Volume"].values.flatten().astype(np.float64)
        else:
            self.volumes = np.ones(len(self.prices), dtype=np.float64)

        # Action space: continuous [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation space with correct bounds
        # Features: log_returns (window_size-1) + rsi + macd_signal + rel_volume
        #           + portfolio_ratio + cash_ratio
        obs_size = (window_size - 1) + 3 + 2  # = window_size + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Episode tracking
        self.current_step = None
        self.cash = None
        self.shares_held = None
        self.cost_basis = 0.0               # weighted average cost basis
        self.portfolio_values = []
        self.trade_history = []

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average."""
        ema = np.empty_like(data, dtype=np.float64)
        ema[0] = data[0]
        alpha = 2.0 / (period + 1)
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    def _compute_rsi(self, period: int = 14) -> np.ndarray:
        """RSI normalised to [0, 1]."""
        deltas = np.diff(self.prices, prepend=self.prices[0])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = self._ema(gains, period)
        avg_loss = self._ema(losses, period)
        safe_loss = np.where(avg_loss != 0, avg_loss, 1.0)
        rs = np.where(avg_loss != 0, avg_gain / safe_loss, 100.0)
        rsi = 1.0 - 1.0 / (1.0 + rs)
        return rsi  # already in [0, 1]

    def _compute_macd_signal(self) -> np.ndarray:
        """MACD signal direction: -1 / 0 / +1."""
        ema12 = self._ema(self.prices.astype(np.float64), 12)
        ema26 = self._ema(self.prices.astype(np.float64), 26)
        macd_line = ema12 - ema26
        signal_line = self._ema(macd_line, 9)
        diff = macd_line - signal_line
        return np.sign(diff)

    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Construct observation from current state."""
        #Price window = log returns
        start_idx = max(0, self.current_step - self.window_size + 1)
        price_window = self.prices[start_idx : self.current_step + 1]

        # Pad if necessary
        if len(price_window) < self.window_size:
            padding = np.full(self.window_size - len(price_window), price_window[0])
            price_window = np.concatenate([padding, price_window])

        # Log returns (length = window_size - 1)
        log_returns = np.diff(np.log(np.maximum(price_window, 1e-8)))

        # --- Technical indicators (#14) ---
        rsi_all = self._compute_rsi()
        macd_all = self._compute_macd_signal()

        rsi_val = rsi_all[self.current_step]
        macd_val = macd_all[self.current_step]

        # Relative volume (current / rolling 20-day mean)
        vol_start = max(0, self.current_step - 19)
        vol_window = self.volumes[vol_start : self.current_step + 1]
        mean_vol = np.mean(vol_window) if len(vol_window) > 0 else 1.0
        rel_volume = (self.volumes[self.current_step] / mean_vol) if mean_vol > 0 else 1.0

        # --- Portfolio features ---
        current_price = self.prices[self.current_step]
        invested_value = self.shares_held * current_price
        portfolio_ratio = invested_value / self.initial_balance
        cash_ratio = self.cash / self.initial_balance

        observation = np.concatenate([
            log_returns,
            [rsi_val, macd_val, rel_volume],
            [portfolio_ratio, cash_ratio],
        ]).astype(np.float32)

        return observation

    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + invested)."""
        current_price = self.prices[self.current_step]
        return self.cash + self.shares_held * current_price

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # #4: random start support
        min_start = self.window_size - 1
        if self.random_start and len(self.prices) > min_start + 1:
            self.current_step = int(
                self.np_random.integers(min_start, len(self.prices) - 1)
            )
        else:
            self.current_step = min_start

        self.cash = self.initial_balance
        self.shares_held = 0.0
        self.cost_basis = 0.0  # #13
        self.portfolio_values = [self.initial_balance]
        self.trade_history = []

        observation = self._get_observation()
        info = {
            "portfolio_value": self.initial_balance,
            "cash": self.cash,
            "shares_held": self.shares_held,
            "current_price": self.prices[self.current_step],
        }

        return observation, info

    def step(self, action):
        """Execute one time step within the environment."""
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        action_value = np.clip(action_value, -1.0, 1.0)

        current_price = self.prices[self.current_step]
        previous_portfolio_value = self._get_portfolio_value()

        transaction_cost = 0.0
        trade_occurred = False

        if action_value > 0.01:  # BUY
            cash_to_use = self.cash * action_value
            if cash_to_use > 0:
                cash_after_fee = cash_to_use * (1 - self.transaction_fee)
                shares_to_buy = cash_after_fee / current_price

                # #13: Update cost basis (weighted average)
                old_value = self.shares_held * self.cost_basis
                new_value = shares_to_buy * current_price
                total_shares = self.shares_held + shares_to_buy
                if total_shares > 0:
                    self.cost_basis = (old_value + new_value) / total_shares

                self.shares_held += shares_to_buy
                self.cash -= cash_to_use
                transaction_cost = cash_to_use * self.transaction_fee
                trade_occurred = True

                self.trade_history.append({
                    "step": self.current_step,
                    "action": "BUY",
                    "amount": shares_to_buy,
                    "price": current_price,
                    "fee": transaction_cost,
                    "profit": 0.0,  # #13: no profit on buy
                })

        elif action_value < -0.01:  # SELL
            shares_to_sell = self.shares_held * abs(action_value)
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price
                revenue_after_fee = revenue * (1 - self.transaction_fee)

                # #13: Per-trade PnL
                profit = revenue_after_fee - shares_to_sell * self.cost_basis

                self.shares_held -= shares_to_sell
                self.cash += revenue_after_fee
                transaction_cost = revenue * self.transaction_fee
                trade_occurred = True

                self.trade_history.append({
                    "step": self.current_step,
                    "action": "SELL",
                    "amount": shares_to_sell,
                    "price": current_price,
                    "fee": transaction_cost,
                    "profit": profit,  # #13
                })

        # Move to next time step
        self.current_step += 1

        # Termination conditions
        terminated = False
        truncated = False

        if self.current_step >= len(self.prices):
            truncated = True
            self.current_step = len(self.prices) - 1

        new_portfolio_value = self._get_portfolio_value()
        self.portfolio_values.append(new_portfolio_value)

        # Bankruptcy check
        if new_portfolio_value < self.initial_balance * self.bankruptcy_threshold:
            terminated = True

        # Reward: log return
        if previous_portfolio_value > 0:
            log_return = np.log(new_portfolio_value / previous_portfolio_value)
        else:
            log_return = -10.0

        reward = log_return

        if not trade_occurred:
            reward -= self.hold_penalty

        observation = self._get_observation()

        info = {
            "portfolio_value": new_portfolio_value,
            "cash": self.cash,
            "shares_held": self.shares_held,
            "current_price": self.prices[self.current_step],
            "transaction_cost": transaction_cost,
            "log_return": log_return,
            "trade_occurred": trade_occurred,
            "total_return": (new_portfolio_value - self.initial_balance) / self.initial_balance,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.current_step is None:
            return
        portfolio_value = self._get_portfolio_value()
        current_price = self.prices[self.current_step]
        total_return = (portfolio_value - self.initial_balance) / self.initial_balance * 100
        print(
            f"Step: {self.current_step:4d} | "
            f"Price: ${current_price:,.2f} | "
            f"Shares: {self.shares_held:,.2f} | "
            f"Cash: ${self.cash:,.2f} | "
            f"Portfolio: ${portfolio_value:,.2f} | "
            f"Return: {total_return:+.2f}%"
        )

    def get_episode_metrics(self) -> dict:
        """Calculate episode-level performance metrics."""
        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)

        num_trades = len(self.trade_history)
        total_fees = sum(t["fee"] for t in self.trade_history)

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "num_trades": num_trades,
            "total_fees": total_fees,
            "final_portfolio_value": portfolio_values[-1],
        }
