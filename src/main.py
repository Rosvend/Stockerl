"""
Stock Trading Agent - PPO Training Pipeline

Trains a PPO agent to trade Amazon (AMZN) stock using historical data.
"""

import random
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment import TradingEnv
from data.market_data import get_stock_data
from train import train_agent, TensorboardCallback
from evaluate import evaluate_agent, compare_with_baseline, plot_results


class NormSyncCallback(BaseCallback):
    """Sync VecNormalize stats from training env to eval env before each eval."""

    def __init__(self, train_env, eval_env, verbose=0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        sync_envs_normalization(self.train_env, self.eval_env)
        return True


def load_and_prepare_data(
    ticker: str = "AMZN",
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    train_ratio: float = 0.8,
):
    """Load stock data and split into train/test sets."""
    print(f"Loading {ticker} data from {start_date} to {end_date}...")
    df = get_stock_data(ticker, start_date, end_date)

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"   Training samples: {len(train_df)}")
    print(f"   Testing samples:  {len(test_df)}")

    return train_df, test_df


def create_env(df, **env_kwargs):
    """Create a Monitor-wrapped trading environment."""
    def _init():
        env = TradingEnv(df, **env_kwargs)
        env = Monitor(env)
        return env
    return _init


def main():
    """Main training and evaluation pipeline."""

    # Reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 60)
    print(" Stock Trading Agent - PPO Training Pipeline")
    print("=" * 60)

    # Configuration
    TICKER = "AMZN"
    START_DATE = "2020-01-01"
    END_DATE = "2024-12-31"
    INITIAL_BALANCE = 1_000_000
    TOTAL_TIMESTEPS = 500_000

    env_kwargs = {
        "initial_balance": INITIAL_BALANCE,
        "window_size": 10,
        "transaction_fee": 0.001,
        "hold_penalty": 0.0,
        "bankruptcy_threshold": 0.1,
        "random_start": True,
    }

    # Load data
    train_df, test_df = load_and_prepare_data(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        train_ratio=0.8,
    )

    # --- Build training env with VecNormalize ---
    train_env = DummyVecEnv([create_env(train_df, **env_kwargs)])
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=True, clip_obs=10.0
    )

    # --- Build eval env for EvalCallback ---
    # Uses training=False so it doesn't update its own stats
    eval_env = DummyVecEnv([create_env(test_df, **env_kwargs)])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
        training=False,
    )

    # Sync stats from train → eval before each evaluation
    norm_sync = NormSyncCallback(train_env, eval_env)
    eval_callback = EvalCallback(
        eval_env,
        callback_after_eval=norm_sync,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/eval/",
        eval_freq=10_000,
        n_eval_episodes=3,
        deterministic=True,
    )

    tb_callback = TensorboardCallback()
    callback_list = CallbackList([tb_callback, eval_callback])

    # Train agent
    model = train_agent(
        env=train_env,
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback_list,
        seed=SEED,
    )

    # Save model and VecNormalize stats
    model_path = f"ppo_trading_{TICKER.lower()}"
    model.save(model_path)
    vec_norm_path = f"{model_path}_vecnormalize.pkl"
    train_env.save(vec_norm_path)
    print(f"\n Model saved to: {model_path}.zip")
    print(f" VecNormalize stats saved to: {vec_norm_path}")

    # Evaluate agent
    agent_metrics, agent_portfolios = evaluate_agent(
        model,
        test_df,
        num_episodes=5,
        base_seed=SEED,
        vec_normalize_path=vec_norm_path,
        **env_kwargs,
    )

    # Compare with fee-adjusted baseline
    baseline_return, baseline_values = compare_with_baseline(
        test_df,
        initial_balance=INITIAL_BALANCE,
        transaction_fee=env_kwargs["transaction_fee"],
    )

    # Plot results
    plot_results(
        test_df,
        agent_portfolios,
        baseline_values,
        save_path="trading_results.png",
    )

    # Summary
    print("\n" + "=" * 60)
    print(" FINAL SUMMARY")
    print("=" * 60)
    print(f"Agent Return:    {agent_metrics['total_return']*100:+.2f}%")
    print(f"Baseline Return: {baseline_return*100:+.2f}%")
    print(f"Outperformance:  {(agent_metrics['total_return'] - baseline_return)*100:+.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
