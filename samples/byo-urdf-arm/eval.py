"""Evaluate a three-joint arm policy through a stable complete episode."""

from __future__ import annotations

import simulo

from task import ByoArmTask, app


@app.job(type="eval", system=simulo.SystemType.TIER_1, timeout=8 * 60 * 60)
def evaluate(
    policy: simulo.PolicyCheckpoint,
    episodes: int = 100,
    num_envs: int = 100,
    actions: str = "best",
):
    return simulo.evaluate(
        ByoArmTask(), policy, episodes=episodes, num_envs=num_envs, actions=actions
    )


@app.success
def completed_stably(task):
    """Count an episode when the arm reaches the time limit without moving quickly."""
    lasted = task.check("lasted to the 5 s time limit", task.survived())
    settled = task.check(
        "arm settled below 0.1 rad/s",
        task.robot.state.joint_velocities.norm(dim=-1) < 0.1,
    )
    return lasted & settled
