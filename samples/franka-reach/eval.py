"""Evaluate a reaching policy from the hand's final distance to its goal."""

from __future__ import annotations

import simulo

from task import FrankaReachTask, app


@app.job(type="eval", system=simulo.SystemType.TIER_1, timeout=4 * 60 * 60)
def evaluate(
    policy: simulo.PolicyCheckpoint,
    episodes: int = 100,
    num_envs: int = 100,
    actions: str = "best",
):
    return simulo.evaluate(
        FrankaReachTask(),
        policy,
        episodes=episodes,
        num_envs=num_envs,
        actions=actions,
        env_spacing=2.5,
    )


@app.success
def reached(task):
    """Count an episode when the hand finishes close to its goal."""
    hand = task.hand_position()
    return task.check("hand within 5 cm of the goal", (hand - task.goal_pos).norm(dim=-1) < 0.05)
