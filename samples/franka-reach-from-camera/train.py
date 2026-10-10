# ruff: noqa: I001
"""Train a Franka Panda to reach a cube it can only see through a camera."""

from __future__ import annotations

from typing import Any

from task import FrankaReachFromCameraTask, app

import simulo

with app.runtime.imports():
    from skrl.resources.preprocessors.torch import RunningStandardScaler


def _ppo_overrides() -> dict[str, Any]:
    """Return the PPO settings for this task.

    The numbers start from a published vision PPO recipe for a Panda cube task
    (discount 0.97, entropy bonus 0.01, wider ratio clip 0.3, gradient clip 1.0)
    and were tuned for this reach task.
    """
    return {
        "rollouts": 24,
        "learning_epochs": 5,
        "mini_batches": 8,
        "discount_factor": 0.97,
        "lambda": 0.95,
        "learning_rate": 3e-4,
        "state_preprocessor": RunningStandardScaler,
        "value_preprocessor": RunningStandardScaler,
        "grad_norm_clip": 1.0,
        "ratio_clip": 0.3,
        "value_clip": 0.3,
        "clip_predicted_values": True,
        "entropy_loss_scale": 0.01,
        "value_loss_scale": 1.0,
        # Every episode ends at its time limit, never on failure, so the critic
        # estimates what the last step would have earned had the episode gone on.
        "time_limit_bootstrap": True,
        "experiment": {"write_interval": 24},
    }


_MAX_SEED = 2**32


def _check_seed(seed: int) -> None:
    """Reject a seed NumPy cannot use before the simulation starts, not after it boots."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < _MAX_SEED:
        raise ValueError(f"seed must be a whole number from 0 to {_MAX_SEED - 1}, got {seed!r}.")


@app.job(
    type="train",
    system=simulo.SystemType.TIER_1,
    timeout=45 * 60,
    retries=2,
)
def train_franka_reach_from_camera(
    num_envs: int = 128, max_iterations: int = 600, seed: int = 42
) -> dict[str, Any]:
    """Train a camera reaching policy with PPO.

    Args:
        num_envs: Number of parallel environments to simulate. The default is 128.
            Each one renders its own camera image every step, which costs host
            memory.
        max_iterations: Number of PPO policy-update iterations. The default is 600,
            24 steps from every environment per iteration.
        seed: The same seed repeats a run; a different seed gives an independent
            run. It must be a whole number from 0 to 2**32 - 1.

    Returns:
        A JSON-serialisable dict with training ``stats``.

    Raises:
        ValueError: If an argument is out of range. Checking here stops a
            submitted job before the simulation starts.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be a positive integer, got {num_envs}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be a positive integer, got {max_iterations}")
    _check_seed(seed)

    env = simulo.LearningEnv(
        task=FrankaReachFromCameraTask(),
        num_envs=num_envs,
        device="cuda",
        env_spacing=3.0,
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
        stats = trainer.train(max_iterations=max_iterations, checkpoint_every=50)
        return {"num_envs": num_envs, "seed": seed, **stats}
    finally:
        trainer.close()
        env.close()
