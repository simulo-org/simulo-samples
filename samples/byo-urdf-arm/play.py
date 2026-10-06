"""Play a trained arm policy and record its rollout."""

from __future__ import annotations

import simulo

from task import ByoArmTask, app


@app.job(type="play", system=simulo.SystemType.TIER_1, timeout=15 * 60)
def play(policy: simulo.PolicyCheckpoint, episodes: int = 3) -> None:
    """Play or record a saved policy checkpoint."""
    env = simulo.LearningEnv(
        task=ByoArmTask(with_camera=True),
        num_envs=1,
        device="cuda",
        env_spacing=1.5,
        headless=True,
        seed=42,
        enable_cameras=True,
    )
    player = simulo.RLPlayer(env=env, policy=policy, device="cuda")
    try:
        player.play(num_episodes=episodes, record=simulo.RecordConfig(include_video=True))
    finally:
        player.close()
        env.close()
