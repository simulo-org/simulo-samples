"""Evaluate whether a Humanoid policy stays upright and moves forward."""

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
        HumanoidTask(),
        policy,
        episodes=episodes,
        num_envs=num_envs,
        actions=actions,
        env_spacing=4.0,
    )


@app.success
def walked_forward(task):
    """Count an episode when the robot stays upright and moves forward."""
    upright = task.check("stayed upright to the time limit", task.survived())
    forward = task.check(
        "covered at least 1 m forward over the episode",
        task.robot.state.pose[:, 0] - task.env.scene.env_origins[:, 0] >= 1.0,
    )
    return upright & forward
