"""Evaluate a drifting policy for staying on the track while moving."""

from __future__ import annotations

import simulo

from task import F1TenthDriftTask, app


@app.job(type="eval", system=simulo.SystemType.TIER_1, timeout=8 * 60 * 60)
def evaluate(
    policy: simulo.PolicyCheckpoint,
    episodes: int = 100,
    num_envs: int = 100,
    actions: str = "best",
):
    return simulo.evaluate(
        F1TenthDriftTask(), policy, episodes=episodes, num_envs=num_envs, actions=actions
    )


@app.success
def stayed_on_track(task):
    """Count an episode when the car stays on the track and keeps moving."""
    on_track = task.check(
        "stayed on the track for 5 s",
        task.survived() & ~task.ended_by("car left the track"),
    )
    moving = task.check(
        "moving at least 1 m/s at the end",
        task.robot.state.linear_velocity[:, :2].square().sum(dim=-1) >= 1.0,
    )
    return on_track & moving
