# ruff: noqa: I001
"""Evaluate a camera reaching policy from the fingertips' final distance to the cube."""

from __future__ import annotations

from task import FrankaReachFromCameraTask, app

import simulo


@app.job(type="eval", system=simulo.SystemType.TIER_1, timeout=15 * 60)
def evaluate(policy: simulo.PolicyCheckpoint, episodes: int = 100, num_envs: int = 100):
    """Evaluate deterministic actions on numbered starts, beginning with episode 0."""
    return simulo.evaluate(
        FrankaReachFromCameraTask(),
        policy,
        episodes=episodes,
        num_envs=num_envs,
        actions="best",
        env_spacing=3.0,
    )


@app.success
def reached(task):
    """Count an episode when the fingertips finish within 5 cm of the point above the cube."""
    distance = (task.fingertip_position() - task.target_position()).norm(dim=-1)
    return task.check("fingertips within 5 cm of the target above the cube", distance < 0.05)
