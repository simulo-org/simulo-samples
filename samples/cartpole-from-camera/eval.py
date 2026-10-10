# ruff: noqa: I001
"""Evaluate deterministic actions from a saved camera cartpole policy."""

from __future__ import annotations

import math

from task import CartpoleFromCameraTask, app

import simulo


@app.job(type="eval", system=simulo.SystemType.TIER_1, timeout=15 * 60)
def evaluate(policy: simulo.PolicyCheckpoint, episodes: int = 100, num_envs: int = 100):
    """Evaluate deterministic actions on numbered starts, beginning with episode 0."""
    return simulo.evaluate(
        CartpoleFromCameraTask(),
        policy,
        episodes=episodes,
        num_envs=num_envs,
        actions="best",
        env_spacing=20.0,
    )


@app.success
def balanced(task):
    """Count episodes that survive the time limit and end within 10 degrees of upright."""
    pole_angle = task.robot.state.joint_positions[:, task.robot.find_joints("cart_to_pole")[0]]
    lasted = task.check("lasted to the time limit", task.survived())
    upright = task.check("pole within 10 degrees of upright", pole_angle.abs() < math.radians(10))
    return lasted & upright
