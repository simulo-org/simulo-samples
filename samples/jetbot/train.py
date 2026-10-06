"""Train a JetBot direction-following policy with PPO."""

from __future__ import annotations

from typing import Any

import simulo

from task import JetbotTask, app


@app.job(
    type="train",
    system=simulo.SystemType.TIER_1,
    timeout=8 * 60 * 60,
    retries=2,
)
def train_jetbot(num_envs: int = 16, max_iterations: int = 700) -> dict[str, Any]:
    """Train the Jetbot direction-following policy with PPO.

    Args:
        num_envs: Number of parallel environments to simulate. More environments give
            the trainer more experience per iteration and use more memory. Increase this
            value only after confirming that the selected system has enough capacity.
        max_iterations: Number of PPO policy-update iterations.

    Returns:
        A JSON-serialisable dict with training ``stats``.
    """
    env = simulo.LearningEnv(
        task=JetbotTask(),
        num_envs=num_envs,
        device="cuda",
        env_spacing=2.0,
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
