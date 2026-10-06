"""Evaluate a three-joint arm policy by its target accuracy and stability."""

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
        ByoArmTask(),
        policy,
        episodes=episodes,
        num_envs=num_envs,
        actions=actions,
        env_spacing=1.5,
    )


@app.success
def completed_stably(task):
    """Count an episode when every joint reaches its target and settles."""
    joint_ids = (
        task.robot.find_joints("shoulder_pan")
        + task.robot.find_joints("shoulder_lift")
        + task.robot.find_joints("elbow")
    )
    target = task.target
    reached_target = task.check(
        "every joint within 0.1 rad of its target",
        (task.robot.state.joint_positions[:, joint_ids] - target).abs().amax(dim=-1) < 0.1,
    )
    settled = task.check(
        "arm settled below 0.1 rad/s",
        task.robot.state.joint_velocities[:, joint_ids].norm(dim=-1) < 0.1,
    )
    return reached_target & settled
