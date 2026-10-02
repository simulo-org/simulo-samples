"""Evaluate a Cartpole policy by balancing the pole to the time limit."""

from __future__ import annotations

import math

import simulo

from task import CartpoleTask, app


@app.job(type="eval", system=simulo.SystemType.TIER_1, timeout=8 * 60 * 60)
def evaluate(
    policy: simulo.PolicyCheckpoint,
    episodes: int = 100,
    num_envs: int = 100,
    actions: str = "best",
):
    return simulo.evaluate(
        CartpoleTask(), policy, episodes=episodes, num_envs=num_envs, actions=actions
    )


@app.success
def balanced(task):
    """Count an episode when the pole remains upright until the time limit."""
    pole_angle = task.robot.state.joint_positions[:, task.robot.find_joints("cart_to_pole")[0]]
    lasted = task.check("lasted to the time limit", task.survived())
    upright = task.check("pole within 10 degrees of upright", pole_angle.abs() < math.radians(10))
    return lasted & upright
