"""Evaluation module for the PPO stock trading agent."""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment import TradingEnv


def _load_obs_normalize(vec_normalize_path: str):
    """Load observation normalization stats from a saved VecNormalize file.

    Returns (obs_mean, obs_var, clip_obs) or None if path is None.
    """
    if vec_normalize_path is None:
        return None
    with open(vec_normalize_path, "rb") as f:
        vec_norm = pickle.load(f)
    return vec_norm.obs_rms.mean, vec_norm.obs_rms.var, vec_norm.clip_obs


def _normalize_obs(obs: np.ndarray, stats):
    """Normalize an observation using saved running-mean/var stats."""
    if stats is None:
        return obs
    mean, var, clip_obs = stats
    normalized = (obs - mean) / np.sqrt(var + 1e-8)
    return np.clip(normalized, -clip_obs, clip_obs).astype(np.float32)


def evaluate_agent(
    model,
    test_df,
    num_episodes: int = 5,
    base_seed: int = 0,
    vec_normalize_path: str = None,
    **env_kwargs,
):
    """Evaluate the trained agent on test data.

    Uses the raw TradingEnv (no DummyVecEnv) so that episode metrics
    are not wiped by auto-reset.  Observation normalization is applied
    manually from the saved VecNormalize stats.
    """

    print(f"\n Evaluating agent on {num_episodes} episodes...")

    norm_stats = _load_obs_normalize(vec_normalize_path)

    all_metrics = []
    all_portfolio_values = []

    for episode in range(num_episodes):
        env = TradingEnv(test_df, **env_kwargs)
        obs, _info = env.reset(seed=base_seed + episode)

        # Normalize & reshape to (1, obs_dim) for model.predict
        obs = _normalize_obs(obs, norm_stats)

        done = False
        while not done:
            action, _ = model.predict(obs[np.newaxis], deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            obs = _normalize_obs(obs, norm_stats)
            done = terminated or truncated

        metrics = env.get_episode_metrics()
        all_metrics.append(metrics)
        all_portfolio_values.append(env.portfolio_values.copy())

        print(
            f"   Episode {episode + 1}: "
            f"Return: {metrics['total_return']*100:+.2f}% | "
            f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
            f"MaxDD: {metrics['max_drawdown']*100:.2f}% | "
            f"Trades: {metrics['num_trades']}"
        )

    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()
    }

    print(f"\n Average Metrics:")
    print(f"   Total Return: {avg_metrics['total_return']*100:+.2f}%")
    print(f"   Sharpe Ratio: {avg_metrics['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {avg_metrics['max_drawdown']*100:.2f}%")
    print(f"   Avg Trades:   {avg_metrics['num_trades']:.1f}")

    return avg_metrics, all_portfolio_values


def compare_with_baseline(
    test_df,
    initial_balance: float = 1_000_000,
    transaction_fee: float = 0.001,
):
    """Calculate buy-and-hold baseline performance (fee-adjusted)."""

    if "Close" in test_df.columns:
        prices = test_df["Close"].values.flatten()
    else:
        prices = test_df.iloc[:, 0].values.flatten()

    cash_after_buy_fee = initial_balance * (1 - transaction_fee)
    shares = cash_after_buy_fee / prices[0]
    final_value = shares * prices[-1] * (1 - transaction_fee)
    baseline_return = (final_value - initial_balance) / initial_balance

    baseline_values = shares * prices

    print(f"\n Buy-and-Hold Baseline (fee-adjusted):")
    print(f"   Return: {baseline_return*100:+.2f}%")
    print(f"   Final Value: ${final_value:,.2f}")

    return baseline_return, baseline_values


def plot_results(
    test_df,
    agent_portfolio_values,
    baseline_values,
    save_path: str = None,
):
    """Plot agent performance vs baseline."""

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax1 = axes[0]
    max_len = max(len(pv) for pv in agent_portfolio_values)
    padded_values = []
    for pv in agent_portfolio_values:
        if len(pv) < max_len:
            pv = np.concatenate([pv, [pv[-1]] * (max_len - len(pv))])
        padded_values.append(pv)

    agent_values = np.array(padded_values)
    mean_values = np.mean(agent_values, axis=0)
    std_values = np.std(agent_values, axis=0)

    x = np.arange(len(mean_values))
    ax1.plot(x, mean_values, label="PPO Agent", color="blue", linewidth=2)
    ax1.fill_between(
        x, mean_values - std_values, mean_values + std_values, alpha=0.3, color="blue"
    )
    ax1.plot(baseline_values, label="Buy & Hold", color="orange", linewidth=2, linestyle="--")
    ax1.axhline(y=1_000_000, color="gray", linestyle=":", alpha=0.5, label="Initial")
    ax1.set_xlabel("Trading Day")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.set_title("PPO Trading Agent vs Buy-and-Hold Strategy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    if "Close" in test_df.columns:
        prices = test_df["Close"].values.flatten()
    else:
        prices = test_df.iloc[:, 0].values.flatten()
    ax2.plot(prices, color="green", linewidth=1.5)
    ax2.set_xlabel("Trading Day")
    ax2.set_ylabel("Stock Price ($)")
    ax2.set_title("Stock Price During Test Period")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n Plot saved to: {save_path}")

    plt.close(fig)
