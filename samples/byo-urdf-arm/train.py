"""Train a policy that moves the published three-joint arm to randomized targets."""

from __future__ import annotations

from typing import Any

import simulo

from task import ByoArmTask, app


@app.job(
    type="train",
    system=simulo.SystemType.TIER_1,
    timeout=8 * 60 * 60,
    retries=2,
)
def train(num_envs: int = 256, max_iterations: int = 1200) -> dict[str, Any]:
    """Train the arm-reaching policy with PPO.

    Args:
        num_envs: Number of parallel environments to simulate.
        max_iterations: Number of PPO policy-update iterations.

    Returns:
        A JSON-serializable dictionary with training statistics and the arm asset reference.
    """
    env = simulo.LearningEnv(
        task=ByoArmTask(),
        num_envs=num_envs,
        device="cuda",
        env_spacing=1.5,
        headless=True,
        seed=42,
    )
    trainer = simulo.RLTrainer(env=env, algorithm="PPO", device="cuda", seed=42)
    try:
        stats = trainer.train(max_iterations=max_iterations)
        return {
            "num_envs": num_envs,
            "robot_asset": "robot/byo-urdf-arm:v1",
            **stats,
        }
    finally:
        trainer.close()
        env.close()
