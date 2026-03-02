"""Training module for the PPO stock trading agent."""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """Custom callback for logging additional metrics to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]
            if "total_return" in info:
                self.logger.record("trading/total_return", info["total_return"])
            if "portfolio_value" in info:
                self.logger.record("trading/portfolio_value", info["portfolio_value"])
        return True


def linear_schedule(initial_value: float):
    """Linear learning rate schedule: decays from initial_value to 0."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def train_agent(
    env,
    total_timesteps: int = 500_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 128,
    n_epochs: int = 5,
    gamma: float = 0.96,
    ent_coef: float = 0.01,
    target_kl: float = 0.02,
    clip_range: float = 0.15,
    callback=None,
    seed: int = None,
):
    """
    Train a PPO agent on a pre-built (Vec)environment.

    Parameters
    ----------
    env : VecEnv
        A vectorised environment (e.g. DummyVecEnv or VecNormalize-wrapped).
    callback : BaseCallback or CallbackList, optional
        SB3 callback(s) to use during training.
    seed : int, optional
        Random seed for PPO reproducibility.
    """

    print("\n Training PPO Agent...")
    print(f"   Total timesteps: {total_timesteps:,}")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(learning_rate),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=ent_coef,
        target_kl=target_kl,
        clip_range=clip_range,
        verbose=1,
        seed=seed,
        tensorboard_log="./logs/ppo_trading",
    )

    if callback is None:
        callback = TensorboardCallback()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    print("Training complete!")
    return model
