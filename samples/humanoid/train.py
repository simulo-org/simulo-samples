"""Train the Humanoid walking task."""

from __future__ import annotations

from functools import partial
from typing import Any

import simulo

from task import HumanoidTask, app

with app.runtime.imports():
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.resources.schedulers.torch import KLAdaptiveLR


def _ppo_overrides() -> dict[str, Any]:
    """Return the PPO settings of the standard humanoid recipe that the trainer supports."""
    return {
        "discount_factor": 0.99,
        "lambda": 0.95,
        "learning_epochs": 5,
        "mini_batches": 4,
        "learning_rate": 5e-4,
        # Bound with partial: the trainer's config merge rejects keys inside the
        # empty learning_rate_scheduler_kwargs section, so kl_threshold rides here.
        "learning_rate_scheduler": partial(KLAdaptiveLR, kl_threshold=0.008),
        "state_preprocessor": RunningStandardScaler,
        "value_preprocessor": RunningStandardScaler,
        "grad_norm_clip": 1.0,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "clip_predicted_values": True,
        "entropy_loss_scale": 0.0,
        "value_loss_scale": 2.0,
        "kl_threshold": 0.0,
        "rewards_shaper": _scale_rewards,
        "time_limit_bootstrap": False,
    }


def _scale_rewards(rewards, timestep: int, timesteps: int):
    """Scale rewards by 0.01 before PPO updates, as the recipe does."""
    del timestep, timesteps
    return rewards * 0.01


_MAX_SEED = 2**32


def _check_seed(seed: int) -> None:
    """Reject a seed NumPy cannot use before the simulation starts, not after it boots."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < _MAX_SEED:
        raise ValueError(f"seed must be a whole number from 0 to {_MAX_SEED - 1}, got {seed!r}.")


@app.job(
    type="train",
    system=simulo.SystemType.TIER_2,
    timeout=8 * 60 * 60,
    retries=2,
)
def train_humanoid(
    num_envs: int = 1024, max_iterations: int = 2500, seed: int = 42
) -> dict[str, Any]:
    """Train the humanoid-walking policy with PPO.

    Args:
        num_envs: Number of parallel environments to simulate. The default is 1024.
        max_iterations: Number of PPO policy-update iterations. The default is 2500.
            With 1024 environments, one measured run took about 17 minutes on a
            Tier 2 (A10G) GPU, startup included.
        seed: The same seed repeats a run; a different seed gives an independent run.
            It must be a whole number from 0 to 2**32 - 1.

    Returns:
        A JSON-serialisable dict with training ``stats``.
    """
    _check_seed(seed)
    env = simulo.LearningEnv(
        task=HumanoidTask(),
        num_envs=num_envs,
        device="cuda",
        env_spacing=4.0,
        headless=True,
        seed=seed,
    )
    trainer = simulo.RLTrainer(
        env=env,
        algorithm="PPO",
        device="cuda",
        seed=seed,
        agent_cfg=_ppo_overrides(),
    )
    try:
        stats = trainer.train(max_iterations=max_iterations)
        return {"num_envs": num_envs, **stats}
    finally:
        trainer.close()
        env.close()
