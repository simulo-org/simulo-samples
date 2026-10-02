"""Evaluate a Humanoid policy for upright forward walking."""

from __future__ import annotations

import simulo

from task import HumanoidTask, app


@app.job(type="eval", system=simulo.SystemType.TIER_1, timeout=8 * 60 * 60)
def evaluate(
    policy: simulo.PolicyCheckpoint,
    episodes: int = 100,
    num_envs: int = 100,
    actions: str = "best",
):
    return simulo.evaluate(
        HumanoidTask(), policy, episodes=episodes, num_envs=num_envs, actions=actions
    )


@app.success
def walked_forward(task):
    """Count an episode when the robot stays upright and moves forward."""
    upright = task.check(
        "stayed upright to the time limit",
        task.survived() & ~task.ended_by("robot fell over"),
    )
    forward = task.check(
        "moved forward at least 0.1 m/s at the end",
        task.robot.state.linear_velocity[:, 0] >= 0.1,
    )
    return upright & forward
