"""Evaluate whether a JetBot policy moves in its commanded direction."""

from __future__ import annotations

import simulo

from task import JetbotTask, app


@app.job(type="eval", system=simulo.SystemType.TIER_1, timeout=8 * 60 * 60)
def evaluate(
    policy: simulo.PolicyCheckpoint,
    episodes: int = 100,
    num_envs: int = 16,
    actions: str = "best",
):
    return simulo.evaluate(
        JetbotTask(),
        policy,
        episodes=episodes,
        num_envs=num_envs,
        actions=actions,
        env_spacing=2.0,
    )


@app.success
def kept_moving(task):
    """Count an episode when the robot moves in its commanded direction."""
    command = task.commands
    return task.check(
        "moved at least 0.1 m/s in the commanded direction",
        (task.robot.state.linear_velocity[:, :2] * command[:, :2]).sum(dim=-1) >= 0.1,
    )
