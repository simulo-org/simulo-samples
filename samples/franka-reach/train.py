"""Train this sample with PPO."""

from __future__ import annotations

from typing import Any

import simulo

from task import FrankaReachTask, app


@app.job(
    type="train",
    # Tier 1: T4 GPU, 16 GB VRAM. Run `simulo systems` for the full four-tier catalog.
    system=simulo.SystemType.TIER_1,
    timeout=4 * 60 * 60,
    retries=2,
)
def train_franka_reach(num_envs: int = 2048, max_iterations: int = 300) -> dict[str, Any]:
    """Train a task-space reaching policy with PPO. Simulo saves its checkpoints automatically.

    Args:
        num_envs: Number of parallel environments to simulate.
        max_iterations: Number of PPO policy-update iterations.

    Returns:
        A JSON-serialisable dict of training statistics, such as ``iterations`` and
        ``best_reward``.

    Raises:
        ValueError: If ``num_envs`` or ``max_iterations`` is not a positive
            integer. Checked here, at the argument boundary, because the
            alternative is a GPU job that starts the simulation and then fails
            somewhere less legible.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be a positive integer, got {num_envs}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be a positive integer, got {max_iterations}")

    env = simulo.LearningEnv(
        task=FrankaReachTask(),
        num_envs=num_envs,
        device="cuda",
        dt=1.0 / 120.0,
        physics_steps_per_action=2,
        env_spacing=2.5,
        headless=True,
        seed=42,
    )
    trainer = simulo.RLTrainer(env=env, algorithm="PPO", device="cuda", seed=42)

    stats = trainer.train(max_iterations=max_iterations)

    # Close the trainer before the environment so the RL library releases its
    # resources first.
    trainer.close()
    env.close()

    return {"num_envs": num_envs, **stats}
