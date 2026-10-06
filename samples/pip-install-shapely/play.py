"""Play a trained Shapely target-zone policy and record it."""

from __future__ import annotations

import os

import simulo

from task import ShapelyZoneTask, app


@app.job(type="play", system=simulo.SystemType.TIER_1, timeout=15 * 60)
def play(policy: simulo.PolicyCheckpoint, episodes: int = 3) -> None:
    """Watch or record a policy: simulo run play.py --policy policy_x:best."""
    zone_center_x = float(os.environ.get("DEMO_ZONE_CENTER_X", "2.5"))
    env = simulo.LearningEnv(
        task=ShapelyZoneTask(zone_center_x=zone_center_x, with_camera=True),
        num_envs=1,
        device="cuda",
        env_spacing=8.0,
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
