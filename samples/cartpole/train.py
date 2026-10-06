"""Train a Cartpole balancing policy with PPO."""

from __future__ import annotations

from typing import Any

import simulo

from task import CartpoleTask, app


@app.job(
    type="train",
    system=simulo.SystemType.TIER_1,
    timeout=8 * 60 * 60,
    retries=2,
)
def train(num_envs: int = 4096, max_iterations: int = 200) -> dict[str, Any]:
    """Train Cartpole with PPO."""
    env = simulo.LearningEnv(
        task=CartpoleTask(),
        num_envs=num_envs,
        device="cuda",
        env_spacing=4.0,
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
