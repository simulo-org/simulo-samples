"""Train this sample with PPO."""

from __future__ import annotations

from typing import Any

import simulo

from task import ByoArmTask, app


# retries=2 reruns the job after a failure. A training job saves its latest
# checkpoint automatically every 50 iterations, and a rerun picks up from it
# instead of starting over.
@app.job(
    type="train",
    # Tier 1: T4 GPU, 16 GB VRAM. Run `simulo systems` for the full four-tier catalog.
    system=simulo.SystemType.TIER_1,
    timeout=8 * 60 * 60,
    retries=2,
)
def train(num_envs: int = 256, max_iterations: int = 150) -> dict[str, Any]:
    """Train the reaching policy with PPO. Simulo saves its checkpoints automatically.

    Everything here runs in the Simulo cloud: ``simulo.LearningEnv`` builds ``num_envs``
    parallel copies of ``ByoArmTask`` on the GPU and ``simulo.RLTrainer`` trains a PPO
    policy against them.

    Args:
        num_envs: Number of parallel environments to simulate.
        max_iterations: Number of PPO policy-update iterations.

    Returns:
        A JSON-serialisable dict: the catalog reference the run trained against and
        training statistics such as ``iterations`` and ``best_reward``.
    """
    env = simulo.LearningEnv(
        task=ByoArmTask(),
        num_envs=num_envs,
        device="cuda",
        dt=1.0 / 120.0,
        physics_steps_per_action=2,
        env_spacing=1.5,
        headless=True,
        seed=42,
    )
    trainer = simulo.RLTrainer(env=env, algorithm="PPO", device="cuda", seed=42)

    stats = trainer.train(max_iterations=max_iterations)

    # Close the trainer before the environment so the RL library releases its resources.
    trainer.close()
    env.close()

    return {
        "num_envs": num_envs,
        "robot_asset": "robot/byo-urdf-arm:v1",
        **stats,
    }
