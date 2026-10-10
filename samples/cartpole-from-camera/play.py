# ruff: noqa: I001
"""Play a camera cartpole policy and optionally record each episode."""

from __future__ import annotations

from task import CartpoleFromCameraTask, app

import simulo


@app.job(type="play", system=simulo.SystemType.TIER_1, timeout=15 * 60)
def play(policy: simulo.PolicyCheckpoint, episodes: int = 3, seed: int = 42, record: bool = True):
    """Run deterministic actions with the camera setup used during training."""
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be a whole number from 0 to 2**32 - 1")
    if not isinstance(record, bool):
        raise ValueError("record must be a boolean")

    env = simulo.LearningEnv(
        task=CartpoleFromCameraTask(),
        num_envs=1,
        device="cuda",
        env_spacing=20.0,
        headless=True,
        seed=seed,
        enable_cameras=True,
    )
    try:
        player = simulo.RLPlayer(env=env, policy=policy, device="cuda")
        try:
            results = []
            for _ in range(episodes):
                results.append(
                    player.play(
                        num_episodes=1,
                        deterministic=True,
                        real_time=True,
                        record=simulo.RecordConfig(include_video=True, env_indices=(0,))
                        if record
                        else None,
                    )
                )
            return {"episodes": sum(result["episodes"] for result in results), "results": results}
        finally:
            player.close()
    finally:
        env.close()
