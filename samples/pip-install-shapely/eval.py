"""Evaluate a Shapely target-zone policy by its final position."""

from __future__ import annotations

import os

import simulo

from task import ShapelyZoneTask, app


@app.job(type="eval", system=simulo.SystemType.TIER_1, timeout=8 * 60 * 60)
def evaluate(
    policy: simulo.PolicyCheckpoint,
    episodes: int = 100,
    num_envs: int = 100,
    actions: str = "best",
):
    zone_center_x = float(os.environ.get("DEMO_ZONE_CENTER_X", "2.5"))
    # Use the same 8 m spacing as training, so each copy's zone and robot sit
    # where training put them.
    return simulo.evaluate(
        ShapelyZoneTask(zone_center_x=zone_center_x),
        policy,
        episodes=episodes,
        num_envs=num_envs,
        actions=actions,
        env_spacing=8.0,
    )


@app.success
def reached_target_zone(task):
    """Count an episode when the robot finishes inside the target zone."""
    local_xy = task.robot.state.pose[:, :2] - task.env_origin_xy
    to_zone = task.target_centroid_xy.unsqueeze(0) - local_xy
    return task.check(
        "robot inside the 1 m target zone at the end",
        (to_zone.abs() <= 0.5).all(dim=-1),
    )
