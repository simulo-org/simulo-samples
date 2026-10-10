# ruff: noqa: I001
"""Train a cartpole policy that sees the pole through a camera."""

from __future__ import annotations

from typing import Any

from task import CartpoleFromCameraTask, app

import simulo


@app.job(type="train", system=simulo.SystemType.TIER_1, timeout=55 * 60, retries=2)
def train(num_envs: int = 256, max_iterations: int = 550, seed: int = 42) -> dict[str, Any]:
    """Train with PPO; use seed 43 for an independent run."""
    env = simulo.LearningEnv(
        task=CartpoleFromCameraTask(),
        num_envs=num_envs,
        device="cuda",
        env_spacing=20.0,
        headless=True,
        seed=seed,
        enable_cameras=True,
    )
    trainer = simulo.RLTrainer(
        env=env,
        algorithm="PPO",
        device="cuda",
        seed=seed,
        agent_cfg={
            "rollouts": 32,
            "learning_epochs": 4,
            "mini_batches": 16,
            "learning_rate": 1e-4,
            "value_loss_scale": 1.0,
            "time_limit_bootstrap": True,
            "experiment": {"write_interval": 32},
        },
    )
    try:
        stats = trainer.train(max_iterations=max_iterations, checkpoint_every=50)
        return {"num_envs": num_envs, "seed": seed, **stats}
    finally:
        trainer.close()
        env.close()
