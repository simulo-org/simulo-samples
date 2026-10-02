"""Evaluate a Shapely target-zone policy through a complete episode."""

from __future__ import annotations

import simulo

from task import ShapelyZoneTask, app


@app.job(type="eval", system=simulo.SystemType.TIER_1, timeout=8 * 60 * 60)
def evaluate(
    policy: simulo.PolicyCheckpoint,
    episodes: int = 100,
    num_envs: int = 100,
    actions: str = "best",
):
    return simulo.evaluate(
        ShapelyZoneTask(), policy, episodes=episodes, num_envs=num_envs, actions=actions
    )


@app.success
def completed_episode(task):
    """Record whether the policy completed the full episode."""
    return task.check("lasted to the 10 s time limit", task.survived())
