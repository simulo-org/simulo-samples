"""Train a Franka Panda arm to reach moving goals with PPO."""

from __future__ import annotations

from typing import Any

import simulo

from task import FrankaReachTask, app


@app.job(
    type="train",
    system=simulo.SystemType.TIER_1,
    timeout=4 * 60 * 60,
    retries=2,
)
def train_franka_reach(num_envs: int = 2048, max_iterations: int = 1500) -> dict[str, Any]:
    """Train a task-space reaching policy with PPO.

    Args:
        num_envs: Number of parallel environments to simulate. The default is 2048.
        max_iterations: Number of PPO policy-update iterations. The default is 1500.
            With 2048 environments, one measured run took about 12 minutes on a
            Tier 1 GPU, startup included. Shorter runs of 300 or 1000 iterations
            sometimes stopped before the hand was accurate.

    Returns:
        A JSON-serialisable dict with training ``stats``.

    Raises:
        ValueError: If ``num_envs`` or ``max_iterations`` is not a positive
            integer. Checking here prevents a submitted job from starting the
            simulation with invalid arguments.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be a positive integer, got {num_envs}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be a positive integer, got {max_iterations}")

    env = simulo.LearningEnv(
        task=FrankaReachTask(),
        num_envs=num_envs,
        device="cuda",
        env_spacing=2.5,
        headless=True,
        seed=42,
    )
    trainer = simulo.RLTrainer(env=env, algorithm="PPO", device="cuda", seed=42)
    try:
        stats = trainer.train(max_iterations=max_iterations)
        return {"num_envs": num_envs, **stats}
    finally:
        trainer.close()
        env.close()
