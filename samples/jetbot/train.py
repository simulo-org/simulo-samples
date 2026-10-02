"""Train this sample with PPO."""

from __future__ import annotations

from typing import Any

import simulo

from task import JetbotTask, app


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
def train_jetbot(num_envs: int = 16, max_iterations: int = 700) -> dict[str, Any]:
    """Train the JetBot direction-following policy with PPO.

    Simulo saves the policy's checkpoints automatically.

    Args:
        num_envs: Number of parallel environments to simulate. More environments give
            the trainer more experience per iteration and use more memory. The default
            is the figure measured for this app; a run that asks for more memory than
            the job is given is stopped rather than run slowly. 32 has been measured to
            complete and 64 has not.
        max_iterations: Number of PPO policy-update iterations.

    Returns:
        A JSON-serialisable dict of training statistics, such as ``iterations`` and
        ``best_reward``.
    """
    env = simulo.LearningEnv(
        task=JetbotTask(),
        num_envs=num_envs,
        device="cuda",
        dt=1.0 / 120.0,
        physics_steps_per_action=2,
        env_spacing=2.0,
        headless=True,
        seed=42,
    )
    trainer = simulo.RLTrainer(env=env, algorithm="PPO", device="cuda", seed=42)

    stats = trainer.train(max_iterations=max_iterations)

    # Close the trainer before the environment so the RL library releases its resources.
    trainer.close()
    env.close()

    return {"num_envs": num_envs, **stats}
